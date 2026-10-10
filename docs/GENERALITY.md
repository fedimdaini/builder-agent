# How general is the Builder?

An audit of `builder_agent/`, its templates and prompts, done by reading the code only (2026-10-10).
Nothing was run. Every Builder result so far comes from one dev repo (`../taxi-trip-regression`);
the main project has only been through verify mode. This file says what would carry over to
another repo and what wouldn't.

Verdicts used below:

- **contract**: a team agreement in `contracts.yaml`. It applies to every repo by design.
- **default**: a Builder-internal choice that is reasonable for most repos and easy to change.
- **taxi-shaped**: an assumption that holds for taxi and would break, or silently mislead, on
  many other repos.

## 0. Supported scope (decided 2026-10-10)

**In scope:** Python tabular machine learning, **regression or classification**, with **CSV data** in
the repo (or made by a data step in the repo), trained by **either** an importable function that
returns the model **or** a training script run with arguments. For a script, the LLM reads it and
fills in how to run it: the command, its arguments, and where the model and metrics end up. A
separate slot set and prompt version do this, and the sandbox validates the answers.

**Out of scope, explicitly:** deep learning (PyTorch, TensorFlow/Keras, transformers), text or image
data, notebooks-only repos, pipeline frameworks that own the training run (Airflow, DVC, Kedro,
Prefect), non-CSV data (Parquet, Excel, databases), and data downloaded at run time from outside
the repo. An out-of-scope repo still gets scan facts and verify-mode findings; create mode reports
why it stops instead of generating files that can't work.

## 0.1 The main project against this scope

`mlops-zoomcamp-project` (checked at commit `e157717`, read only) is **outside the create-mode scope**:

| Aspect | Main project | In scope? |
|---|---|---|
| Training | `train_sklearn_model(download_date, sklearn_model, model_name)` in `airflow/dags/dag_02_training.py`, run by an Airflow `PythonOperator`; it logs to MLflow and saves the model and a `DictVectorizer` as pickles to S3; it does not return the model | no: orchestrated by Airflow, and the function takes a date, not data |
| Data | `dag_01` downloads NYC taxi Parquet from the internet into LocalStack S3; `dag_02` reads Parquet from S3 | no: downloaded at run time, Parquet, S3 |
| Target | `duration`, computed from two timestamp columns | not supported by the scanner yet (roadmap item 5) |
| Task / framework | regression with scikit-learn (LinearRegression, Lasso, RandomForest, GradientBoosting) | yes |
| Dependencies | root `Pipfile` (Python 3.9, `apache-airflow = "*"` unpinned), plus per-service `requirements.txt` | yes, but the root Pipfile isn't what the services use |
| Serving | Flask app with a batch `POST /predict`, no `/health`, `POST /reload` from S3 | needs the contract bridge (roadmap item 8) |
| Existing pipeline files | `docker-compose.yaml`, Dockerfiles, `Makefile` | verify mode, not create |

**What create mode can do there today:** nothing end to end. Every file the project already has goes
to verify mode (5 findings, `experiments/verify/mlops-zoomcamp-project/`). The missing ones (CI
workflow, `pipeline/` adapters) can't be filled, because no slot can point at a function that
returns a model from CSV data.

**What would be needed** (a separate mode, not a widening of the tabular scope):

1. An **orchestrated-repo mode**: the make targets call the repo's own orchestrator inside its own
   stack. For example, `make data` = `airflow dags test dag_01_download_nyc_data <date>` and
   `make train` = `airflow dags test dag_02_training <date>`, each run in the Airflow container
   of the repo's compose, with DAG ids and the date argument as slots. This is the script-training
   mode one level up: a command with arguments, validated by the sandbox.
2. A sandbox that brings up the **repo's** compose (Airflow, LocalStack, Postgres; about 589 s
   and the memory peak measured in `docs/reference-run-main.md`), not the Builder's.
3. **Network access** in the sandbox for the NYC data download, and a sample mode (one month).
4. The **contract bridge** for the API (roadmap item 8).
5. Scanner support for **Parquet** columns and for a **target computed from two columns**
   (roadmap item 5).

Until then, the main project is the held-out test for **verify mode**, and create mode is
evaluated on in-scope repos.

## 1. Specific values and assumptions

### Hard-coded literals

A search of `builder_agent/` and `prompts/` for taxi names (`taxi`, `trip_duration`, `9696`,
`pickup`, `mlops-zoomcamp`) finds them only in docstrings, usage examples and the dev-tool defaults
of `llm/eval.py` (`--repo ../taxi-trip-regression`, `--gold tests/gold/taxi_slots.json`). The
prompts contain none. The pipeline logic has no taxi literals. The taxi-specific parts are in
assumptions about the repo's shape, listed next.

### Item by item

| What | Where | Verdict | Why |
|---|---|---|---|
| Port 8000, `POST /predict`, `GET /health`, `prediction` field, `MODEL_URI`, errors never 200 | `contracts.yaml` serving | contract | The API every agent calls. |
| Make target names, `SAMPLE=1`, fraction 0.01 | `contracts.yaml` | contract | |
| Paths: `data/raw`, `data/processed`, `models`, `reports/eval.json`, `logs/predictions.jsonl`, `pipeline/` | `contracts.yaml` paths | contract | |
| `mlflow_uri: http://mlflow:5000` | `contracts.yaml` | contract | |
| `mlflow_experiment: mlops-zoomcamp-experiment` | `contracts.yaml` | contract, but **project-named** | Every repo the Builder handles would log into the main project's experiment. It should be per repo (e.g. derived from the repo name). |
| `eval_report` with `rmse`, `mae`, `r2` and `per_group` | `contracts.yaml` formats, `evaluate.py.j2` | contract, but **regression-only** | A classification repo gets meaningless metrics. `per_group` needs a categorical column present in every data file that has the target (`_group_columns`); a repo without one can't fill the slot. |
| MLflow server `ghcr.io/mlflow/mlflow:v2.17.2`, client pinned to `mlflow==2.17.2` | `render/configs.py` `MLFLOW_IMAGE` | default | Needed to avoid fault-001. A repo that pins mlflow 3.x, or deps that conflict with mlflow 2.17.2's own requirements, would break at install (not tested; Python 3.13 support of 2.17.2 is unchecked). |
| Base image `python:<version>-slim`, `WORKDIR /app` | `Dockerfile.j2`, `configs.py` | default | |
| System packages: only `make` | `configs.py` `apt_packages` | default, **thin** | Taxi needs none. LightGBM wheels need `libgomp1` on slim images; packages built from source need a compiler. Today the fix loop has to discover these one failure at a time. |
| `pipenv==2023.12.1` | `decide/rules.py` `PIPENV_VERSION` | default | Known to work on taxi only. |
| Adapter extras: `flask`, `gunicorn`, `pandas`, `numpy`, `mlflow` | `configs.py` `ADAPTER_PACKAGES` | default | The generated service is Flask + gunicorn. |
| Folder names: `data`, `raw`, `processed`, `models`, `artifacts`, ... | `scan/layout.py` | default | Heuristics; data under other names isn't mounted. |
| Script roles from file names (`train`, `predict`, `features`, ...) | `scan/code.py` `ROLE_PATTERNS` | default | Works for cookiecutter-style layouts (taxi is one). Fails for `main.py` or `pipeline.py` that does everything. |
| Python version needs an exact hint (Pipfile, `.python-version`, `runtime.txt`, lock, conda, Dockerfile, CI, `.pyc`) or a constraint with a lower bound | `decide/rules.py` `choose_python` | **taxi-shaped** | Many requirements.txt-only repos have no hint. Then `python` is `needs_llm`, and `config_context` raises `Python version not decided`. The only workaround is to force it in `contracts.yaml`, which affects every repo. |
| Train data read with `pd.read_csv`; columns read only from `.csv`/`.tsv` | `train.py.j2`, `evaluate.py.j2`, `scan/layout.py` | **taxi-shaped** | Parquet, Excel, databases, or data downloaded at run time can't fill the data slots. |
| Training = one repo function `module:function` that **returns** the model | `render/slots.py` `train_function` | **taxi-shaped** | Taxi has `train_xgboost(train_path, target_col, ...)`. Fails for top-level training scripts, argparse `main()`, notebooks-only repos, functions that save the model instead of returning it, and framework nodes (Kedro, DVC stages). |
| Model flavors: `xgboost`, `sklearn`, `lightgbm`, `catboost`; inputs `dmatrix`/`numpy`/`dataframe` | `render/slots.py` | **taxi-shaped** (tabular only) | PyTorch, TensorFlow/Keras, transformers, statsmodels and prophet are detected by the scanner (`scan/known.py`) but can't be trained, logged or served. |
| Target transform: only `log1p`/`expm1` and `log`/`exp` | `scan/targets.py` `TRANSFORM_PAIRS` | **taxi-shaped** | Taxi's log1p. A scaler, Box-Cox or a sklearn `TransformedTargetRegressor` isn't recognised. |
| One target column in the data | `render/slots.py` `target_column` | taxi-shaped | No multi-output targets, and no target computed from two columns (roadmap item 5 notes the main project needs that). |
| Smoke test against a known prediction | `tests/gold/taxi_expected.json` | taxi-specific **data** | Each new repo needs a reference run for its expected value. Without one, only the response shape is checked. |
| Gold slot answers | `tests/gold/taxi_slots.json` | taxi-specific data | Used by the evals and the sandbox runs; a new repo needs the LLM, or a new gold file. |
| Fix menu and validator | `fix/models.py` | default | The actions are general. The rejection reasons were added after taxi faults (apt patterns, Python packages as apt names, the version already in the image), but they are general rules. |
| Fault memory (fault-001 + 13 generated faults) | `tests/faults/` | **taxi-specific data** | Every memory document is a taxi failure (mlflow 2.17.2 server, Pipfile on Python 3.9, libffi8, xgboost, the taxi port). Retrieval on another repo would return taxi fixes. |
| Repo app used as is only if Flask/FastAPI, with `/predict` and `/health`, and importing mlflow | `decide/rules.py` `plan_serving` | contract + default | Otherwise the Flask adapter is generated. Streamlit/Gradio apps are ignored, which is right. |
| Verify rules (images, CRLF, MLflow artifacts, healthchecks) | `verify/rules.py` | default | Written from main-project faults, but they check general compose/Dockerfile mistakes. |

## 2. Which repo shapes are supported today

"Supported" means scan → decide → render handles it with rules or slots. Nothing outside taxi has
been run end to end.

| Aspect | Supported | Not supported |
|---|---|---|
| Dependency format | Pipfile + Pipfile.lock; Pipfile alone; poetry (`pyproject.toml` + `poetry.lock`); `requirements*.txt` / `.in` at the repo root; `pyproject.toml` with `[project]` deps, `setup.py`, `setup.cfg` (`pip install .`, no layer caching) | conda `environment.yml` (scanned for the Python version, but no install rule, so `needs_llm` and render fails); `uv.lock` (detected, no rule); requirements files only in subfolders |
| Python version | an exact hint or a constraint with a lower bound | no hint at all (render fails unless `contracts.yaml` forces a version) |
| ML framework | scikit-learn, XGBoost, LightGBM, CatBoost; regression | PyTorch, TensorFlow/Keras, transformers, statsmodels, prophet, Spark; classification (the metrics are regression-only); clustering, ranking, multi-output |
| Training code | an importable `.py` function that returns a model, args filled from data path, DataFrame, `X`, `y`, target name or literals | notebooks-only repos (notebooks give facts, not callable functions); top-level scripts; argparse CLIs; Kedro/DVC/Airflow pipelines (Airflow is roadmap item 5); models saved to disk instead of returned |
| Data | CSV/TSV files in the repo (also via Git LFS, pulled), mounted read-only | Parquet/Excel/feather headers (detected as data, columns not read); data downloaded at run time (`dvc pull`, Kaggle, URLs); no data folder; databases |
| Data step | `existing` (the processed CSV is already there) or a no-argument function | a step that needs arguments or configuration |
| Serving | the repo's Flask/FastAPI app if it meets the contract, else a generated Flask adapter that loads the model from MLflow | the main project's batch API without the bridge (roadmap item 8) |
| Experiment tracking | none in the repo, or MLflow (the Builder adds its own server) | a repo already on mlflow 3.x, W&B/Neptune calls in the repo's own train function (not handled; untested) |
| Target transform | none, log1p, log | anything else |

## 3. Does the memory learn from real runs?

**No.** `memory_cases()` (`builder_agent/memory/__init__.py`) indexes only `tests/faults/`: the
hand-recorded fault-001, plus the generated faults with split `memory`. Fix-loop runs are exported
to `experiments/fixes/` for analysis, but nothing writes them back into the memory. A fix that worked
on a real run is never retrieved for the next one. CLAUDE.md's design rule "Store both successful and
failed fixes in memory" is not implemented yet.

What it would take: after each fix-loop run, write a memory document per verified fix (stage, error
signature, the fix, and whether it passed). Mark failed fixes as such, so retrieval can show
"tried, didn't work". Keep a held-out set that is never written to, so later evaluations stay
clean. Then tag every document with the repo it came from, so one repo's fixes don't dominate
another's retrieval.

## 4. Candidate repos for a generalization test

Chosen to differ from taxi (Pipfile, XGBoost regression, a train function, CSV in the repo). **These
are from memory and not checked.** Before cloning, confirm each one's current layout,
dependency files and data size.

| Repo (GitHub) | Why it differs | What I expect to break | Size |
|---|---|---|---|
| `mlflow/mlflow-example` | conda env (`conda.yaml`) + `MLproject`; scikit-learn ElasticNet on wine-quality CSV; already logs to MLflow | install (conda: no rule), possibly the Python version; training is a script (`train.py` with `sys.argv`), not a function | tiny |
| `kedro-org/kedro-starters` (the `spaceflights-pandas` starter) | Kedro project, `pyproject.toml` + `requirements.txt`; scikit-learn regression; data includes `.xlsx` in `data/01_raw` | training lives in Kedro nodes that take DataFrames and a parameters dict, run through a data catalog; Excel data; the starter is a template, so it may need `kedro new` first | small |
| `iterative/example-get-started` | DVC pipeline (`dvc.yaml`, `params.yaml`), `requirements.txt`; text classification (random forest on TF-IDF) | data comes from `dvc pull`, not the repo; classification vs the regression metrics; sparse matrices instead of DataFrames | small (data downloaded) |
| `alexeygrigorev/mlbookcamp-code` (`course-zoomcamp/05-deployment`, churn) | close to taxi (Pipfile, Flask) but scikit-learn **classification** with a pickled model and `DictVectorizer` | the metrics; the model is pickled with its vectorizer, not returned by a function; serving reads the pickle | small |
| a small PyTorch or Keras tabular repo (to pick) | deep-learning framework | flavors (no torch/keras support) and image size (a CPU torch wheel is about 700 MB, fine on 16 GB) | medium |

A sensible order: mlbookcamp churn first (one change from taxi: classification), then
mlflow-example (dependency format), then kedro and DVC (pipeline frameworks). Run the deep-learning
repo last, as a known-unsupported case. For each repo, record what scan and decide produce
before any sandbox run. The `needs_llm` items and render errors are the generality measure.
