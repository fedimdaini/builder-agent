# Integration: what the Builder needs from the team

The Builder runs on its own: every value that depends on another agent or on the team has a working
default in the `integration` section of `contracts.yaml`. This page lists the questions to ask each
teammate, and for each one the exact field the answer changes.

**How to record an answer.** Every field has the shape `{value, default, confirmed}`. Set `value` to the
answer and `confirmed: true`, even when the answer equals the default. Leave `default` as it is.
`python -m builder_agent.integration` lists the fields nobody has confirmed yet; add `--strict` to make it
exit 1 while any remain.

**Rule for new features** (ROADMAP): a feature that touches another agent puts its team-dependent values
in the `integration` section, adds its questions here, and uses a documented format. A test fails if one
of these defaults is hard-coded in the Builder's code (`tests/test_integration.py`).

## Monitor

| Question | Field | Default |
|---|---|---|
| Where should the model service write the prediction log? | `integration.monitor.predictions_log` | `logs/predictions.jsonl` |
| Which keys should each log line use for time, model version, features, prediction and request id? | `integration.monitor.predictions_log_fields` (Builder's name to your key) | the same names |

Each log line is one JSON object: the time (ISO-8601, UTC), the model version, the input record by
feature name, the prediction, and a request id. The API response returns the same request id under the
same key, so late labels can be joined back to it. Do you also need anything else in the line?
That would be a new field, not a rename.

## Trainer

| Question | Field | Default |
|---|---|---|
| Where should `make evaluate` write its report? | `integration.trainer.eval_report` | `reports/eval.json` |
| Which top-level keys should the report use (model version, task, metrics, per group)? | `integration.trainer.eval_report_fields` | the same names |
| Which regression metrics do you read? | `integration.trainer.eval_metrics` | `[rmse, mae, r2]` |
| Which classification metrics do you read? | `integration.trainer.eval_classification_metrics` | `[accuracy, f1_macro, roc_auc]` |
| Where do you read the URI of the model `make train` just trained? | `integration.trainer.latest_model_uri` | `models/latest_model_uri.txt` |
| Which MLflow experiment? One shared experiment, or one per repo (`auto`)? | `integration.trainer.mlflow_experiment` | `mlops-zoomcamp-experiment` |
| Which commands do you run to train and to evaluate? | `integration.trainer.train_command`, `evaluate_command` | `make train`, `make evaluate` |

Limits today:
- **Metrics:** only `rmse`, `mae`, `r2` and `accuracy`, `f1_macro`, `roc_auc` are supported. Asking for
  another one stops rendering with a clear error. The report always contains all the supported metrics
  of its task: a shorter list is accepted but doesn't remove any.
- **Commands:** each must be `make <target>` with a target of `make_targets`. Renaming a target is a
  contract change.

## Inspector

| Question | Field | Default |
|---|---|---|
| Which raw-data folder do you watch? | `integration.inspector.raw_data` | `data/raw` |
| Which processed-data folder do you watch? | `integration.inspector.processed_data` | `data/processed` |
| Where should the Builder put the first Pandera schema draft? | `integration.inspector.schema_path` | `schema.py` (not generated yet: ROADMAP B7) |

## Deployer

| Question | Field | Default |
|---|---|---|
| Which port does every agent call the model on? | `integration.deployer.serving_port` | `8000` |
| Which predict and health endpoints (method and path)? | `integration.deployer.endpoints` | `POST /predict`, `GET /health` |
| Is the request one JSON record with features by name? | `integration.deployer.request_format` | `json` |
| Which response key holds the prediction, and which the model version? | `integration.deployer.prediction_field`, `model_version_field` | `prediction`, `model_version` |
| Do errors always come back as 4xx or 5xx? | `integration.deployer.error_status` | `4xx_or_5xx` |
| Which environment variable gives the service its model URI? | `integration.deployer.model_uri_env` | `MODEL_URI` |
| Which command starts the service? | `integration.deployer.serve_command` | `make serve` |
| What are the compose service names of the live and the candidate model? | `integration.deployer.current_service`, `candidate_service` | `serving-current`, `serving-candidate` |
| What are your compose file and router config called? | `integration.deployer.deploy_file`, `router_config` | `compose.deploy.yml`, `router.yaml` |
| Should the Builder add the candidate container and the router, or do you? | `integration.deployer.candidate_and_router` | `false` (true is refused until ROADMAP D is done) |

## Board owner

| Question | Field | Default |
|---|---|---|
| What is the board's URL inside the compose network? | `integration.board.url` | `http://board:8000` |
| Which image runs the board? (the Builder adds the service once this is set) | `integration.board.image` | empty: no board service |
| What should the agents file be called? | `integration.board.agents_file` | `agents.yaml` |
| Which topics does each agent subscribe to? | `integration.board.subscriptions` | the architecture page's table |
| Is the agents.yaml format below what the board reads? | (format version 1, below) | |

## Reviewer

| Question | Field | Default |
|---|---|---|
| Who should the Builder's commits on `builder/<date>` come from? | `integration.reviewer.builder_commit_author` | `Builder agent <builder@localhost>` |
| Where does the CI workflow go? (create doesn't write it yet: PIPELINE_GAPS.md item 4) | `integration.reviewer.ci_workflow` | `.github/workflows/ci.yml` |

The `builder/` branch prefix is a hard rule (CLAUDE.md), not an integration field.

## Team

| Question | Field | Default |
|---|---|---|
| What is the MLflow tracking URI inside the containers? | `integration.team.mlflow_uri` | `http://mlflow:5000` |
| Which MLflow server image does the team run? The client is pinned to its tag. | `integration.team.mlflow_server_image` | `ghcr.io/mlflow/mlflow:v2.17.2` |

## agents.yaml format (version 1)

Written once by create, at `integration.board.agents_file`; the board owns it afterwards.

```yaml
format_version: 1
repo: <repo name>
board_url: <integration.board.url>
agents:
  <agent>:                       # builder, reviewer, inspector, trainer, deployer, monitor (all six, always)
    subscriptions: [<topic>, ...]  # integration.board.subscriptions.<agent>
    watch:                       # what this agent should look at in this repo
    - {path: <repo-relative path>, kind: file | folder | runtime}
    # agent-specific keys:
    #   reviewer:  ci_workflow {path, status}
    #   inspector: schema {path, status}
    #   trainer:   commands {train, evaluate}, mlflow {uri, experiment}
    #   deployer:  commands {serve}, service {name, port, endpoints, model_uri_env}, files_you_create [...]
    #   monitor:   health_url, fields {prediction, model_version, log_line {...}}
```

- **`kind: file` or `folder`:** exists in the repo, or is a file this Builder run writes.
- **`kind: runtime`:** appears only after the first pipeline run. Only three paths can have it: the
  predictions log (Monitor), the eval report and the latest model URI file (Trainer).
- **The static check** (`render/agents.py` `check_agents`, part of the sandbox's render stage) refuses a
  file that:
  - doesn't parse as YAML;
  - doesn't list all six agents;
  - has an agent without subscriptions;
  - watches a non-runtime path that doesn't exist, or a path outside the repo;
  - marks any other path as runtime.
