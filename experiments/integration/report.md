# Integration rehearsal (ROADMAP A4b, 2026-10-12)

**What.** `contracts.rehearsal.yaml` is a copy of `contracts.yaml` with every integration field that has
another valid value changed: 27 of 34. `make_rehearsal_contracts.py` builds it, and the real file is
only read. With it, `python -m builder_agent.create --contracts experiments/integration/contracts.rehearsal.yaml
--no-write-back` ran on a fresh taxi clone (`--expected` the taxi reference, LLM slots) and on
`script-file-demo` (LLM slots, `train_script_v1`).

**Result.**
- **Both runs:** every sandbox stage passed with the changed values. Both committed 12 files on
  `builder/2026-10-12`, including `team/agents.yaml`, and both repos are back on their branch with the
  working tree unchanged (`create_taxi.json`, `create_script-file-demo.json`).
- **`check_rehearsal.py`:** reads the committed files (git show) and the evaluate stage's output. 27 of 27
  checks are ok in each repo: each new value is where it should be, and no old default remains (`check.json`).
- **Negative control:** the same check on a clone built with the real contracts gives 2 of 27 ok. The
  two are the fields with nothing to render: classification metrics on a regression repo, and the
  candidate service.
- **Board:** the generated `compose.base.yml` has a `board-svc` service with `hashicorp/http-echo:1.0.0`.
  `docker compose up -d board-svc` from it started the service, and `curl localhost:5678` answered
  `hello-world`; it was then taken down.
- **`candidate_and_router: true`** is refused before any sandbox run: "contracts.yaml
  integration.deployer.candidate_and_router is true, but the candidate container and router are not
  implemented yet (ROADMAP D)" (`candidate_and_router_on.txt`).
- **`python -m builder_agent.integration`** on the copy lists 27 fields as "changed" and the 7 below as
  "default" (`integration_listing.txt`).

| Field | New value | Where it shows up | taxi | script demo |
|---|---|---|---|---|
| `monitor.predictions_log` | `monitoring/predictions.log.jsonl` | serve.py PREDICTIONS_LOG; compose volume ./monitoring; agents.yaml monitor watch (runtime) | ok | ok |
| `monitor.predictions_log_fields` | `{"time": "ts", "model_version": "model", "features": "inputs", "pre...` | serve.py log line keys (and the response's request id key) | ok | ok |
| `trainer.eval_report` | `metrics_out/evaluation.json` | evaluate.py EVAL_REPORT; compose volume ./metrics_out; agents.yaml trainer watch (runtime) | ok | ok |
| `trainer.eval_report_fields` | `{"model_version": "model", "task": "kind", "metrics": "scores", "pe...` | evaluate.py report keys (task key: classification only) | ok | ok |
| `trainer.eval_metrics` | `["rmse", "mae"]` | evaluate.py keep filter; the evaluate stage printed only these | ok | ok |
| `trainer.eval_classification_metrics` | `["accuracy", "f1_macro"]` | evaluate.py, classification repos only (not rendered for a regression repo (the filter is per task); covered by tests) | ok | ok |
| `trainer.latest_model_uri` | `state/latest_model.txt` | train/evaluate/serve MODEL_URI_FILE; compose volume ./state; evaluate and serve passed reading it | ok | ok |
| `trainer.mlflow_experiment` | `rehearsal-experiment` | train.py EXPERIMENT; agents.yaml trainer.mlflow | ok | ok |
| `inspector.raw_data` | `data/external` | agents.yaml inspector watch (only if the folder exists in the repo) (listed) | ok | ok |
| `inspector.schema_path` | `quality/schema.py` | agents.yaml inspector.schema.path | ok | ok |
| `deployer.serving_port` | `8081` | Dockerfile EXPOSE, gunicorn bind, compose ports and healthcheck, agents.yaml | ok | ok |
| `deployer.endpoints` | `{"predict": {"method": "POST", "path": "/v1/predict"}, "health": {"...` | serve.py routes, smoke_test.py, compose healthcheck, agents.yaml | ok | ok |
| `deployer.prediction_field` | `y_pred` | serve.py response key; smoke_test.py checks it; predict stage passed | ok | ok |
| `deployer.model_version_field` | `model_tag` | serve.py response key | ok | ok |
| `deployer.model_uri_env` | `SERVING_MODEL_URI` | compose environment; the adapters read it | ok | ok |
| `deployer.current_service` | `model-live` | compose service; agents.yaml deployer.service and monitor.health_url | ok | ok |
| `deployer.candidate_service` | `model-next` | nowhere yet: used once the candidate container exists (ROADMAP D) (not used while candidate_and_router is off) | ok | ok |
| `deployer.deploy_file` | `deploy/compose.deploy.yaml` | compose header comment; agents.yaml deployer.files_you_create | ok | ok |
| `deployer.router_config` | `deploy/router.yaml` | agents.yaml deployer.files_you_create | ok | ok |
| `board.url` | `http://board-svc:5678` | agents.yaml board_url; compose board service name and port | ok | ok |
| `board.image` | `hashicorp/http-echo:1.0.0` | compose board service image | ok | ok |
| `board.agents_file` | `team/agents.yaml` | the agents file's path in the commit | ok | ok |
| `board.subscriptions` | `6 agents changed` | agents.yaml subscriptions | ok | ok |
| `reviewer.builder_commit_author` | `Rehearsal Bot <rehearsal@example.com>` | the commit's author | ok | ok |
| `reviewer.ci_workflow` | `.github/workflows/builder-ci.yml` | agents.yaml reviewer.ci_workflow (create doesn't write the workflow yet) | ok | ok |
| `team.mlflow_uri` | `http://tracking:5050` | compose service tracking on 5050 and MLFLOW_TRACKING_URI; train/evaluate/serve default; agents.yaml | ok | ok |
| `team.mlflow_server_image` | `ghcr.io/mlflow/mlflow:v2.17.1` | compose mlflow image; Dockerfile client pin mlflow==2.17.1 | ok | ok |

The evaluate stage printed only the listed metrics: taxi `{"rmse": 353.13, "mae": 227.44}`, script demo
`{"rmse": 2.04, "mae": 1.45}`. `eval_report_fields.task` appears only in a classification report.

**Not changed (no other valid value for these repos):**
- `inspector.processed_data`: names where the repo keeps its processed data, which is `data/processed`
  in both repos.
- `trainer.train_command`, `trainer.evaluate_command`, `deployer.serve_command`: each must be
  `make <target>`, and each role has one target. Renaming a target is a contract change.
- `deployer.request_format`, `deployer.error_status`: the only format and error rule the service
  implements.
- `deployer.candidate_and_router`: off on purpose, and checked separately (above).

## Bugs found and fixed (fields that needed a code change)

1. **Runtime output folders were not volumes.**
   - *Before:* compose mounted only `models/` and the predictions log's folder. A `latest_model_uri` or
     an eval report anywhere else stayed inside the container that wrote it, so the next stage (its own
     container) couldn't read the model URI.
   - *Also with the defaults:* `reports/eval.json` never reached the host, where the Trainer reads it.
   - *Fix:* every runtime output's folder is now a volume (`render/configs.py` `RUNTIME_OUTPUTS`). With
     the real contracts, taxi's `compose.base.yml` gains `./reports:/app/reports`, so its snapshot hash
     in `tests/gold/taxi_rendered_sha256.json` changed on purpose; the other rendered files are
     byte-identical.
   - Found by the static render with the rehearsal contracts, before any sandbox run.
2. **`trainer.eval_metrics` had no effect.** The report kept every supported metric whatever the list
   said. A shorter list now renders a filter that keeps only those metrics, overall and per group (with
   `n`). The full default list renders nothing new, so taxi's `evaluate.py` is unchanged.
3. **The smoke test ignored `deployer.prediction_field`.**
   - *Before:* `smoke_context` read `smoke_test` from the untyped extras, where it never is, and fell
     back to `"prediction"`.
   - *Effect in the first taxi run:* the service answered `y_pred` correctly, but the predict stage
     failed with `FAIL response has no "prediction"`. The fix loop couldn't help, create wrote nothing,
     and the clone was left unchanged (`create_taxi_first_try_smoke_bug.json`).
   - *Fix:* it now reads the typed field, which the loader fills from the prediction field.

Each bug has a test in `tests/test_integration.py`.

Also added: `--contracts` on `python -m builder_agent.create`, so a rehearsal never touches the real
contracts.
