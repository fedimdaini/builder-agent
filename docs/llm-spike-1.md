# LLM spike 1: can a local 7B model write the taxi adapters?

Date: 2026-10-08. Throwaway experiment, not part of the agent.
Script: `experiments/llm_spike.py`. Outputs and full logs: `experiments/output/`.

## Setup

- Model: `qwen2.5-coder:7b` via Ollama HTTP API, digest `dae161e2…f4364`, Q4_K_M.
  Options: `temperature 0, seed 42, num_ctx 8192`.
- One call per adapter. The prompt holds only what the Builder would have:
  - the BuildPlan's needs_llm item and its context (`target:train`, `serving`),
  - the repo file it points at (`src/models/train_model.py`, `src/models/predict_model.py`),
  - the relevant contracts.yaml values (paths, MLflow, sample mode, serving endpoints and formats).
- System prompt: adapters live in `pipeline/`, only call existing repo functions, never modify
  repo files; answer with one Python code block.
- The generated code was parsed (`ast.parse`) but never run.

## Results

| | `pipeline/train.py` | `pipeline/serve.py` |
|---|---|---|
| Latency | 143.3 s (first call; likely includes loading the model) | 6.8 s |
| Prompt / output tokens | 553 / 423 | 704 / 314 |
| Syntax (`ast.parse`) | ok | ok |
| Would it work? | no | no |

What it got right: the right MLflow URI default and experiment name, a `__main__` block,
`mlflow.xgboost.log_model`; in serve, a `GET /health` route, the `prediction` field, and errors
returned with status 500 instead of 200.

## The 10 problems (found by reading the code)

### pipeline/train.py

1. **Wrong target column.** Guesses `"duration"`; the real column is `trip_duration`, so it
   would fail with `KeyError`. The plan doesn't contain the target column.
2. **`SAMPLE=1` doesn't shrink training.** It samples 1 % into `df_train`, then calls
   `train_xgboost(train_path, ...)`, which re-reads the full CSV. Only the metrics use the sample.
3. **Missing imports.** Uses `np` and `xgb` without importing them (`NameError`).
4. **Metrics on training data.** rmse/mae/r2 are computed on the rows it trained on, so they
   are optimistic.
5. **Picks `train.csv`.** The notebook trains the saved model on `train_large.csv`.

### pipeline/serve.py

6. **Predictions silently wrong.** Drops the `np.expm1` inverse transform even though it was
   in the source it was given. The model predicts `log1p(duration)`, so it would return about
   6.3 instead of about 531 seconds, with HTTP 200.
7. **Breaks the adapter rule.** Rewrites `single_prediction` instead of calling the repo's.
   (Importing the repo module isn't clean either: it loads `models/xgb_taxi_trip.json` at import.)
8. **Still positional.** `pd.DataFrame([features])` keeps the key-order bug from the reference
   run; the contract's "features by name" isn't met.
9. **Missing import.** Uses `os` without importing it, so it crashes at startup.
10. **Every `/predict` returns 500.** Reads `model.metadata.model_version`; MLflow's `Model` has
    no such attribute (checked on mlflow 3.17.0: attributes are artifact_path, flavors,
    mlflow_version, model_id, model_uuid, prompts, run_id, utc_time_created). The
    `AttributeError` is caught by the `try/except` and returned as a 500.

## Conclusions

- **A syntax check proves nothing.** Both files parse. pyflakes would catch problems 3 and 9
  (undefined names) in milliseconds, before any sandbox build.
- **Missing facts become guesses.** The target column, the feature list and the target
  transform (`log1p` in training, `expm1` in serving) must be facts in the context. They belong
  in the scanner, like the sample request.
- **The model ignores source it was given.** Dropping `expm1` (problem 6) from code that was in
  the prompt means a 7B model should not write whole files.
- **Wrong numbers pass a shape-only smoke test.** Problem 6 still returns a `prediction` field
  with HTTP 200. The smoke test must also compare a known prediction from the reference run
  (row 1 of `data/processed/test.csv` → about 531 s; see docs/reference-run-taxi.md).
- **Design change.** Adapters become Jinja2 templates. The LLM only fills slots (which function
  to call, target column, inverse transform, ...) as strict JSON, choosing from candidate lists
  the scanner provides.
