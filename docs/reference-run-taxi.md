# Reference run: hand-written Dockerfile for taxi-trip-regression

Date: 2026-10-08. Written by hand, no LLM loop. This is the baseline the Builder's
generated Dockerfile should match or beat.

## Result

| Step | Outcome |
|---|---|
| `docker build` | passed first try, 2 min 22 s (pipenv install 102 s) |
| Build context | 8.3 MB (data/ excluded; mostly the 7.8 MB model) |
| Image | `taxi-ref`, 1.77 GB, Python 3.9.25, xgboost 2.0.0, Flask 3.0.1, pandas 2.0.3, gunicorn 21.2.0 |
| `data/` in image | absent (checked with `ls /app/data`) |
| `POST /predict` with row 1 of `data/processed/test.csv` | HTTP 200, `0:8:51` (531 s) vs actual 700 s |

Taxi repo not modified: `git status` clean before and after.

## Files (in `../reference/`, outside both repos)

- `Dockerfile.taxi` — the hand-written Dockerfile
- `Dockerfile.taxi.dockerignore` — keeps data/, notebooks/, reports/, docs/, .git/ out of the context
- `build.log` — full `docker build --progress=plain` output
- `request.json` — row 1 of test.csv minus `trip_duration`; `request_sorted.json`,
  `request_reversed.json`, `request_missing.json` are the probes below
- `Dockerfile` — the original author's Dockerfile, moved here when the taxi repo removed it
  (commit e228acc). Left untouched.

## Commands

```
# from builder-agent/
docker build -f ../reference/Dockerfile.taxi -t taxi-ref ../taxi-trip-regression
docker run -d --name taxi-ref-run -p 9696:9696 taxi-ref
curl -X POST -H "Content-Type: application/json" --data @../reference/request.json http://127.0.0.1:9696/predict
docker rm -f taxi-ref-run
```

## Problems hit and how they were fixed

1. **`../reference/Dockerfile` already existed.** It is the original author's Dockerfile
   (same content as `d789667:Dockerfile`, only CRLF vs LF differs).
   Fix: kept it, wrote the new one as `Dockerfile.taxi`.

2. **Keeping `data/` out without touching the taxi repo.** A `.dockerignore` must sit at
   the context root, which is inside the taxi repo.
   Fix: BuildKit's per-Dockerfile ignore file, `<Dockerfile name>.dockerignore` next to the
   Dockerfile. Context dropped from about 1.4 GB to 8.3 MB. Needs BuildKit (default in
   Docker 23+).

3. **No documented request format.** The repo has no API docs or example payload.
   Fix: notebook `taxi-trip-duration.ipynb` cell 140 posts
   `df_test.sample(n=1).to_dict(orient='records')[0]`, i.e. a test.csv row without
   `trip_duration`. Built `request.json` from row 1 with the same 24 keys in CSV column
   order, booleans as JSON `true`/`false`.

The build itself and gunicorn startup had no errors. The only build warning is pip's
"running pip as root" notice, which is harmless in a container.

## Found during testing, NOT fixed (bugs in the taxi app, not in the Dockerfile)

Fixing these means changing taxi repo code, which is out of scope here.

4. **Predictions depend on JSON key order.** The model was trained on `df.values`, so the
   booster has `feature_names: None` and matches features by position. The app builds the
   DataFrame from the request dict in whatever key order the client sent.
   Same trip, same values:

   | Payload | Prediction |
   |---|---|
   | CSV column order | 0:8:51 |
   | keys sorted A–Z | 0:6:51 |
   | keys reversed | 0:15:20 |

   Every response is HTTP 200, so a client can't tell. Fix belongs in `predict_model.py`
   (reorder to the training column list) or in retraining with feature names.

5. **Missing features are accepted silently.** Dropping `bearing` (23 of 24 features)
   still returns HTTP 200 with `0:8:22`.

6. **Errors return HTTP 200.** `predict()` catches every exception and returns
   `{"error": ...}` with status 200, so health checks and the Deployer's error rate won't
   see failures. Confirmed: `{"geodesic_distance": "abc"}` returns HTTP 200 with
   `{"error": "DataFrame.dtypes for data must be int, float, bool or category. ..."}`.

7. **Image is 1.77 GB.** `Pipfile.lock` has notebook-only packages (geopandas, matplotlib,
   seaborn, ipykernel, Cython) that the serving app never imports. Kept as is so the image
   matches the lock; a serving-only install would need a separate requirements list.

## Lessons for the Builder

- The scan facts were enough to write this Dockerfile: Python 3.9 (4 agreeing hints),
  Pipfile + lock, `src.models.predict_model:app`, port 9696, `models/xgb_taxi_trip.json`
  referenced by path, data/ folder to exclude.
- Use a per-Dockerfile `.dockerignore` when the target repo must not be modified.
- A smoke test needs a sample payload. The scanner doesn't find one yet; here it came from
  a notebook cell and the test.csv header.
- Checking only for HTTP 200 is not enough: items 4–6 all return 200. The sandbox check
  should parse the body and fail on an `error` key.
- Items 4 and 5 are training–serving skew issues for the Reviewer and the Inspector.
