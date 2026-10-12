# What the Builder should generate vs what it generates today (create mode)

Checked on 2026-10-12 at commit `1cee92c`, by reading the code only; no code was changed.
The target list is the Builder card of `docs/architecture.html` ("Files it generates from
templates") plus the lineage map, which the same page assigns to the Builder ("The Builder creates
it; the Reviewer and Trainer update it"). `contracts.yaml` and `docs/ROADMAP.md` say who owns what;
`docs/GENERALITY.md` says which repos create mode supports.

Verdicts:

- **done**: generated, statically checked, and run end to end in the sandbox on the taxi repo;
- **partial**: generated, but part of the spec is missing or it has never run where it would really
  run;
- **missing**: not generated at all.

## Summary

| # | Item (architecture page) | Verdict | Where |
|---|---|---|---|
| 1 | Makefile: install, data, train, evaluate, serve, test, all | **done** | `render/templates/Makefile.j2`, `render/configs.py` `_make_rules` |
| 2 | Dockerfile | **done** | `render/templates/Dockerfile.j2` (+ `dockerignore.j2`), `render/configs.py` `config_context` |
| 3a | docker-compose: current serving container | **done** | `render/templates/compose.base.yml.j2` (service `serving-current`) |
| 3b | docker-compose: MLflow | **done** | same file (service `mlflow`, image `ghcr.io/mlflow/mlflow:v2.17.2`) |
| 3c | docker-compose: candidate serving container | **missing** | not generated; `contracts.yaml` puts it in `compose.deploy.yml`, owned by the Deployer |
| 3d | docker-compose: router | **missing** | not generated; `contracts.yaml` puts it in `compose.deploy.yml` / `router.yaml`, owned by the Deployer |
| 3e | docker-compose: message board | **missing** (blocked) | not generated; waits for the board owner's service (CLAUDE.md, `contracts.yaml` `compose.base_file`) |
| 4 | GitHub Actions workflow | **partial** | `render/ci.py`, `render/templates/ci.yml.j2`, `render/actionlint.py` |
| 5 | MLflow config | **partial** | compose service + `MLFLOW_TRACKING_URI` + experiment name + pinned client; no model registry |
| 6 | Pandera `schema.py` for the Inspector | **missing** | only detected (`decide/modes.py`, "schema draft") |
| 7 | `agents.yaml` (paths and subscriptions per agent) | **missing** | only detected (`decide/modes.py`) |
| 8 | Lineage map | **missing** | nothing in `builder_agent/` |

Besides the generated files, the adapters in `pipeline/` (`train.py` or `train_script.py.j2`,
`evaluate.py`, `serve.py`, `data.py`, `sample_request.json`, `smoke_test.py`) are done. They are
how the Makefile targets reach the repo's own code.

So 4 items are done (Makefile, Dockerfile, the current service and MLflow in compose), 2 are
partial (CI, MLflow config), and 6 are missing. 3 of the missing ones (candidate, router, board)
depend on other owners.

## Item by item

### 1. Makefile: done

`Makefile.j2` writes one rule per `contracts.yaml` `make_targets` entry, in the contract's order:
install, data, train, evaluate, serve, test and all. It uses `.NOTPARALLEL` and `SAMPLE ?= 0`, with
`SAMPLE=1` for a 1% sample. `_make_rules` picks each target's command:

- install: the rule-decided install commands (Pipfile, poetry, requirements, `pip install .`);
- data, train, evaluate: the repo's command if a rule decided one, else the `pipeline/` adapter.
  In script mode `make train` always goes through the adapter (`via_adapter`);
- serve: the repo's app if it meets the API contract, else `pipeline/serve.py`;
- test: `pipeline/smoke_test.py`.

A target with no command makes rendering stop (`RenderError`); it never writes an empty rule.

Checked by the sandbox stages render, build, mlflow, data, train, evaluate, serve, health and
predict. The taxi repo passes all of them with the prediction inside the ±50% reference: 472.18 s
against 531 s, unchanged at every regression check (`experiments/regression_checks.jsonl`).

### 2. Dockerfile: done

`python:<version>-slim`, with the version from the repo or `DEFAULT_PYTHON` 3.11, reported as
ASSUMED. Dependencies are installed before the code is copied, so their layer stays cached. System
packages come from rules (`make`, and `SYSTEM_PACKAGES` such as `libgomp1` for lightgbm). The MLflow
client is pinned to the server's version. Data is never copied: `.dockerignore` excludes it and
compose mounts it. A static lint (`lint_configs`) checks EXPOSE and the instructions. Checked by the
sandbox's build stage.

### 3. docker-compose

The file is `compose.base.yml` (the `contracts.yaml` name), not `docker-compose.yml` as the
architecture page says. Checked by `docker compose config` (`compose_config_check`) and used by
every sandbox run.

- **3a, current serving container: done.** The service is `serving-current` (its name comes from
  `contracts.yaml` `router.services.current`). It builds the Dockerfile, mounts data read-only and
  `models/` and `logs/` read-write, and has a healthcheck on `/health`. It passes
  `MLFLOW_TRACKING_URI` and `MODEL_URI`.
- **3b, MLflow: done.** The server image is pinned. The backend store is SQLite and artifacts go to
  a named volume. Port 5000.
- **3c, candidate serving container: missing.** The template's header says it belongs in
  `compose.deploy.yml`, owned by the Deployer. `contracts.yaml` names `serving-candidate` but assigns
  that file to the Deployer, and the architecture page lists it among the Builder's files. The team
  has to settle that conflict first.
  *Estimate: 0.5 day* for a first draft of `compose.deploy.yml` with `serving-candidate`. It would
  use the same image and `MODEL_URI` taken from a `CANDIDATE_MODEL_URI` variable, and would be
  checked by bringing it up in the sandbox next to `serving-current`. This only makes sense if the
  team agrees the Builder writes the first draft, the way it does for `schema.py`.
- **3d, router: missing.** It has the same ownership conflict: `contracts.yaml` gives
  `router.yaml` and the traffic split to the Deployer.
  *Estimate: 1 day* for a draft: a small proxy (for example nginx with a weighted upstream, set to
  100/0 at first) plus a sandbox stage that sends `/predict` through it. It also needs the team to
  agree on the format of `router.yaml`.
- **3e, message board: missing, blocked.** CLAUDE.md: "added to compose.base.yml once the board
  owner provides it". There is no image or schema to add yet.
  *Estimate: 0.5 day once the image exists:* the service, its healthcheck and a `BOARD_URL`
  variable for the other services, checked by a sandbox stage that waits for it to be healthy.

### 4. GitHub Actions workflow: partial

**Done:**

- `render/ci.py` plans `.github/workflows/ci.yml` from scanner facts, with no LLM:
  - install;
  - the repo's unit tests, with tests that need live services left out (`classify_tests`);
  - the smoke test with `SAMPLE=1` on a local MLflow file store;
  - pre-commit on the changed files;
  - `docker compose config`;
  - a docker build of each Dockerfile whose inputs changed.
- The runner and the actions are pinned.
- Whatever the rules can't decide is listed as needs_llm in a comment at the top of the file.
- Validated by a YAML parse and actionlint. An invalid workflow is never written
  (`tests/test_ci.py`).

**Missing:**

- The sandbox doesn't render it: `sandbox/__init__.py` calls only `render_adapters` and
  `render_configs`. So it has never been rendered together with the other files for taxi.
- It has never run on GitHub Actions.
- The needs_llm items (`ci:service-tests`, `ci:pre-commit`, `ci:tests`) have no slots or prompt.

*Estimate: 1 day.* About 0.5 day to render it in the sandbox run, so it is written with the other
files and linted every time, and 0.5 day for one real run on a fork of the taxi repo, recorded with
its run URL. Filling the needs_llm items with an LLM would be a separate step.

### 5. MLflow config: partial

**Done:**

- The server in compose (3b).
- `MLFLOW_TRACKING_URI` comes from `contracts.yaml` `paths.mlflow_uri` and is set in compose for the
  model service.
- The experiment is set by the contract, or "auto" names it after the repo (`experiment_name()`).
- The client is pinned to the server's version (fault-001).
- The adapters log params, the schema (`builder/schema.json`) and the model, and they write
  `models/latest_model_uri.txt` for evaluate and serve.

**Missing:**

- **No model registry.** Models are referred to as `runs:/<run_id>/model`, and `model_version` in the
  API and the predictions log is derived from that URI. Nothing calls `register_model`, so the
  "training command and model registry location" the Builder should give the Trainer
  (architecture page, first-setup case) is a run URI, not a registered model.
- No config file of its own. Everything is in compose environment variables and the adapters. That
  may be enough; the architecture page doesn't name a file.

*Estimate: 1 day* for registration: in `train.py.j2` and `train_script.py.j2`, register the logged
model under a name derived from the repo, then point `latest_model_uri.txt` and `model_version` at
`models:/<name>/<version>`. SQLite supports the registry. It would be checked by the existing
evaluate, serve and predict stages. Because the train adapter changes, the taxi
rendered-file hashes (`tests/gold/taxi_rendered_sha256.json`) would change on purpose.

### 6. Pandera schema.py for the Inspector: missing

`decide/modes.py` only looks for an existing `schema.py` to choose verify mode. Nothing generates
one. The scanner reads only column names (`scan/layout.py`): no types, null counts or ranges. The
only schema the Builder writes today is `builder/schema.json` in MLflow: feature names and target.

*Estimate: 2 days.*

- Read a sample of the train file (the data slot already names it) and infer per column: dtype,
  nullable, and min/max or the allowed categories, with a margin.
- A `schema.py.j2` that emits a `pandera.DataFrameSchema` marked as a draft. `contracts.yaml` says
  the Inspector owns it after this.
- Check it in the sandbox: validate the eval file with it. A first draft that rejects the repo's
  own data is a bug.
- pandera has to be added to the adapter extras (or installed only in the check), and its version
  pinned.

### 7. agents.yaml: missing

`decide/modes.py` only detects an existing one. Every value it needs exists already:

- the subscriptions table on the architecture page (`builder: ["pipeline:*"]`, ...);
- the paths from `contracts.yaml` (`eval_report`, `predictions_log`, `latest_model_uri`, data
  folders, the MLflow URI and experiment);
- the plan's commands (make targets, the serve command, port and endpoints).

*Estimate: 1 day:* a template and a check that every path it names exists after a sandbox run. One
decision is open: `contracts.yaml` says the board owns `agents.yaml` ("Builder writes it once"), so
its format should be agreed with the board owner first.

### 8. Lineage map: missing

Nothing in `builder_agent/` builds a graph of code, features, dataset, model and deployment. Most
of the inputs exist already:

- the commit: HEAD of the target repo;
- the features: the adapters' `FEATURES` from the data header;
- the dataset: the train and eval files (a sha256 would be added);
- the model: the MLflow run of `make train`;
- the deployment: `serving-current`.

Which code feeds which feature is harder. The scanner knows the repo's feature functions only by
file role (`scan/code.py` `ROLE_PATTERNS`), not per column.

*Estimate: 2 days* for a file-level map: commit, train and eval files with their hashes, the feature
list, the model run and the serving service, written as JSON after a passing sandbox run. Per-column
links from code to feature (an AST pass over the feature functions) would be another 2-3 days.
First, the team has to agree on the format and where it lives (`contracts.yaml` has an owner,
`lineage_map: [reviewer, trainer]`, but no path).

## Outside the list, but needed for create mode to deliver these files

- **No create command.** The render functions write through `RepoWriter` (new files only, on
  `builder/<date>`), but only the sandbox (into a temp copy), the evals and the tests call them. No
  command takes a repo through scan, decide, LLM slots, sandbox and fix loop, and then writes the
  passing files into the repo's branch. The architecture page's "commits once every sandbox stage
  passes" is not implemented.
  *Estimate: 1.5 days:* a `python -m builder_agent.create <repo>` that reuses those pieces, writes
  only after an all-stages-ok run, and commits on `builder/<date>`, with `tests/test_repo_rule.py`
  extended to cover it.
- **"Tells each agent where to watch":** the first-setup case's findings to the Reviewer, Inspector
  and Trainer need the board (3e) and `agents.yaml` (7).

## Total, roughly

| Group | Items | Estimate |
|---|---|---|
| Builder alone | CI in the sandbox and one real run (4), model registry (5), Pandera draft (6), create command | about 5.5 days |
| Needs a format agreed with the team first | agents.yaml (7), lineage map (8, file level) | about 3 days |
| Needs the Deployer or the board owner | candidate (3c), router (3d), board (3e) | about 2 days, after their decision or service |

The estimates are for the taxi repo with a sandbox check each, like the steps so far. They don't
include testing on other repos (`docs/GENERALITY.md` section 4).
