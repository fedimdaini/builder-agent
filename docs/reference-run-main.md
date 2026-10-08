# Reference run: main project (mlops-zoomcamp-project), by hand

Date: 2026-10-08. Roadmap item 1. No Builder, no LLM. The main project is a held-out test:
nothing here is used to tune prompts.

## Result

The full pipeline works: download → train (4 models in MLflow) → deploy the best model to
S3 → `/reload` → `/predict` returns a prediction.

| Step | Outcome | Time |
|---|---|---|
| `docker compose build` | ok, 8 images | 326 s |
| `docker compose up -d` (7 services + dependencies) | ok | 187 s, Airflow healthy ~20 s later |
| `dag_01_download_nyc_data` | 2 runs, both success | ~37 s |
| `dag_02_training` | 2 runs, both success | 334 s (from unpause) |
| `dag_03_deploy` | 2 runs, both success | 46 s |
| `POST /reload` | `{"reloaded": true, "result": "success"}`, HTTP 200 | |
| `POST /predict` | `{"predictions": [14.267329740152285], "result": "success"}`, HTTP 200 | |
| Peak memory (all containers) | **8.39 GiB** of the 9.71 GiB Docker limit | at 20:12:43 UTC |

Each DAG ran twice: unpausing creates a scheduled run on top of the manual trigger (problem 2).

## Setup

- Repo: https://github.com/jelambrar96-datatalks/mlops-zoomcamp-project at `e157717`, cloned into
  `C:/dev/mlops-zoomcamp-project` with `git clone -c core.autocrlf=false` (the two `.sh` files are LF).
- Machine: Windows 11, Docker Desktop (Compose v5.1.4), 9.71 GiB memory limit, 32 CPUs.
  Ollama was not running.
- `docker-compose.yaml`, only these four lines changed (the team guide's fixes):
  ```
  -    image: localstack/localstack:stable      +    image: localstack/localstack:4.12
  -    image: postgres:latest   (x3)            +    image: postgres:16
  ```
- `.env` created with the team's values (gitignored). It differs from the README's example in
  `_PIP_ADDITIONAL_REQUIREMENTS=""` (README: a pinned package list; the Airflow image already
  installs them) and `AIRFLOW_START_TIME="2024-01-01"` (README: 2023-01-01).

## Commands

```bash
docker compose build
docker compose up -d airflow-webserver airflow-scheduler airflow-worker airflow-triggerer \
                     localstack_s3_client mlflow flask-app
# per DAG, in order, waiting until all its runs finished:
docker compose exec -T airflow-webserver airflow dags unpause <dag_id>
docker compose exec -T airflow-webserver airflow dags trigger <dag_id>
docker compose exec -T airflow-webserver airflow dags list-runs -d <dag_id> -o json   # poll
```

Compose also started the dependencies: postgres, redis, airflow-init (exits 0), localstack,
postgresmlflow. `localstack_s3_client` creates the buckets and an IAM role, then exits 0.

## What the run produced

**LocalStack** (`s3://mlops-zoomcamp-bucket`, also `mlflow-bucket` exists, empty):

```
nyc-taxi-data/type=yellow/year=2026/month=06/raw-data.parquet     96.2 MiB   (scheduled dag_01 run)
nyc-taxi-data/type=yellow/year=2026/month=07/raw-data.parquet     90.0 MiB   (manual dag_01 run)
nyc-taxi-data/datasets/year=2026/month=09/{train,val,test}.parquet, dict_vectorizer.pkl
nyc-taxi-data/datasets/year=2026/month=10/{train,val,test}.parquet, dict_vectorizer.pkl
models/skmodels/best_model/model.pkl            152.6 KiB
models/skmodels/best_model/dict_vectorizer.pkl   10.2 KiB
```

`dag_01` downloads the month 3 months before the run's logical date (`MONTH_RELATIVE_DELTA = 3`):
the manual run (2026-10-08) fetched 2026-07, the scheduled one (2026-09-01) fetched 2026-06.

**MLflow**, experiment `mlops-zoomcamp-experiment` (id 1): 8 runs, the same 4 models twice.
Target: `duration` in minutes (dropoff − pickup), kept between 1 and 60.

| model_name | rmse | mae | r2 |
|---|---|---|---|
| sklearn-gradient-boosting-regression | **6.33** | 4.64 | 0.640 |
| sklearn-random-forest-regression | 6.41 | 4.63 | 0.631 |
| sklearn-lasso | 7.33 | 5.61 | 0.517 |
| sklearn-linear-regression | 5.9 × 10⁸ | 1.3 × 10⁷ | −3.1 × 10¹⁵ |

The two training runs (datasets `month=09` and `month=10`) gave identical metrics.

## Facts the Builder needs

| | Value |
|---|---|
| MLflow inside the Docker network | `http://mlflow:5000` (service `mlflow`, container port 5000) |
| MLflow on the host | `http://localhost:5001` (`MLFLOW_PORT`) |
| Experiment | `mlops-zoomcamp-experiment` |
| MLflow backend / artifacts | Postgres `postgresmlflow`; artifact root `/tmp/artifacts` (local path, not S3) |
| S3 (LocalStack) inside the network | `http://localstack:4566`, bucket `mlops-zoomcamp-bucket` |
| Model used by the API | S3 `models/skmodels/best_model/model.pkl` + `dict_vectorizer.pkl` (pickle) |
| Services | airflow-webserver/-scheduler/-worker/-triggerer, localstack, localstack_s3_client, mlflow, postgresmlflow, flask-app, postgres, redis |
| Host ports | Airflow 8081, MLflow 5001, Flask 8000, LocalStack 4566–4597 |
| DAG ids | `dag_01_download_nyc_data`, `dag_02_training`, `dag_03_deploy` |

### The API (`flask-app`, port 8000)

- `POST /reload` (no body): loads the model and vectorizer from S3.
  Response `{"reloaded": true, "result": "success"}` or `{"reloaded": false, "result": "failed"}`, always HTTP 200.
  It must be called after `dag_03`; the app doesn't load a model at startup.
- `POST /predict`: a **batch**. Request `{"data": [record, ...]}`; the model uses
  `PULocationID`, `DOLocationID`, `trip_distance` and `pickup_datetime`; other fields are ignored.
  Response `{"predictions": [float, ...], "result": "success"}`, or `{"predictions": null, "result": "failed"}`
  with HTTP 200.
- `GET /` returns the text `Zoomcamp application`.
- **There is no `/health`** (`GET /health` → 404).

Example request used (from `flask/test_app.py`, the repo's own test of `/predict`; the README's
example `{'feature1': 1, 'feature2': 2}` is a placeholder):

```json
{"data": [{"VendorID": 2, "pickup_datetime": "2024-01-01 00:57:55", "dropoff_datetime": "2024-01-01 01:17:43",
  "passenger_count": 1.0, "trip_distance": 1.72, "RatecodeID": 1.0, "store_and_fwd_flag": "N",
  "PULocationID": 186, "DOLocationID": 79, "payment_type": 2, "fare_amount": 17.7, "extra": 1.0,
  "mta_tax": 0.5, "tip_amount": 0.0, "tolls_amount": 0.0, "improvement_surcharge": 1.0,
  "total_amount": 22.7, "congestion_surcharge": 2.5, "Airport_fee": 0.0, "execution_date": "2024-01",
  "type_tripdata": "yellow"}]}
```

Response: `{"predictions": [14.267329740152285], "result": "success"}` (minutes; this trip took 19.8 min).

### Gaps between the API and contracts.yaml (the Builder has to bridge them)

| contracts.yaml | this project |
|---|---|
| `GET /health` | missing |
| one record per request | batch `{"data": [...]}` |
| response field `prediction` (float) + `model_version` | `predictions` (list) + `result`, no model version |
| errors return 4xx/5xx, never 200 | failures return 200 with `"result": "failed"`; a missing `data` key returns 500 |
| model from MLflow via `MODEL_URI` | model pickle from S3, loaded only on `POST /reload` |
| MLflow `http://mlflow:5000`, experiment `taxi-duration` | `http://mlflow:5000`, experiment `mlops-zoomcamp-experiment` |
| `serving.port` 8000 | 8000 (same) |

## Peak memory (`docker stats` every 5 s, 589 s from startup to the end)

Total peak **8.39 GiB** (86% of the 9.71 GiB limit) at 20:12:43 UTC, while **two `dag_01` runs
ran at the same time** (scheduled + manual), each loading a month of TLC parquet into pandas.

| Container | Peak |
|---|---|
| airflow-worker | 6953 MiB |
| airflow-webserver | 1078 MiB |
| airflow-scheduler | 597 MiB |
| mlflow | 551 MiB |
| localstack | 370 MiB |
| airflow-triggerer | 305 MiB |
| flask-app | 159 MiB |
| postgresmlflow | 89 MiB |
| airflow-init | 86 MiB |
| postgres | 73 MiB |
| redis | 22 MiB |

At rest after the run: 3.11 GiB in total (worker 0.8 GiB).

## Problems and fixes

1. **The guide's setup fixes were applied before starting** (`localstack:stable` → `4.12`,
   `postgres:latest` → `16`, `core.autocrlf=false`), so their failures did not occur in this run.
   They become fault cases 2–4 (roadmap item 2).

2. **Unpausing a DAG also starts a scheduled run.** The DAGs have a monthly schedule with
   `catchup=False`, so `airflow dags unpause` immediately creates a run for the latest interval
   (`scheduled__2026-09-01`), and the manual trigger adds a second one. Result: 2 runs per DAG,
   8 MLflow runs instead of 4, two raw months in S3, and two concurrent `dag_01` runs, which set
   the memory peak. *Not fixed* (only the guide's changes were allowed). One run per DAG would
   need `schedule=None` in the DAGs; skipping the unpause doesn't work, because a paused DAG never
   executes its triggered runs. The Builder should expect both runs.

3. **LinearRegression diverges** (RMSE 5.9 × 10⁸): unregularized regression on the one-hot
   location features. `dag_03` picks the lowest RMSE (gradient boosting), so the deployed model
   is fine. *Not fixed* (project code).

4. **`/predict` error handling** (seen in the calls):
   - one record without the `{"data": [...]}` wrapper → HTTP 500 (`KeyError: 'data'`, HTML error page);
   - `"trip_distance": "far"` is accepted: HTTP 200, `"success"`, prediction 17.57;
     the DictVectorizer treats the string as an unknown category and drops the feature silently;
   - a failed prediction returns HTTP 200 with `"result": "failed"`.
   *Not fixed* (project code); these are contract gaps for the API adapter (roadmap item 8).

5. **MLflow artifacts are not reachable through MLflow.** The artifact root is the local path
   `/tmp/artifacts`; the Airflow worker writes model files to its own `/tmp/artifacts`
   (host `./airflow/mlflow_artifacts`, 48 files), while the MLflow server's `/tmp/artifacts`
   (host `./mlflow/artifacts`) stays empty. Loading a model by `MODEL_URI` from another container
   would fail; `dag_03` works around it by copying the best model to S3. *Not fixed.*

6. **No healthcheck on `mlflow` and `flask-app`**, and `flask-app` doesn't depend on `localstack`.
   Everything still came up in order in this run. *Not fixed;* a verify-mode check (roadmap item 7).

7. **Feature bug (from reading the code, not seen in outputs):** `prepare_features()` computes
   `pickup_minutes = hour + 60 * minute`; minutes since midnight would be `60 * hour + minute`.
   `dag_02_training.py` line 233 has the same formula, so training and serving agree, but the
   feature is wrong. *Not fixed.*

8. Tooling only (not the project): my first polling command had a Python quoting error; replaced
   by a small script. No effect on the run.

## State left behind

- The stack is running: airflow-webserver, -scheduler, -worker, -triggerer, localstack, mlflow,
  postgresmlflow, flask-app, postgres, redis (`localstack_s3_client` and `airflow-init` exited 0).
  The three DAGs are unpaused, so they will also run on their monthly schedules.
  Stop with `docker compose down` in `C:/dev/mlops-zoomcamp-project`.
- Repo: `git status` shows only `docker-compose.yaml` modified (the four image lines). The run
  created gitignored files: `.env`, `airflow/logs/`, `airflow/mlflow_artifacts/`,
  `airflow/dags/__pycache__/`.
