# Builder Agent — ADSP Topic 4 (six-agent MLOps pipeline)

My part: the Builder agent. It reads an existing Python ML repo and generates
the pipeline files (Makefile, Dockerfile, .dockerignore, docker-compose, CI
workflow, MLflow config, Pandera schema draft, agents.yaml), verifies them with
a sandbox build, and fixes failures in a loop.
Full team architecture: docs/architecture.html

## Design rules
- Three layers: scan (facts, no LLM) -> decide (facts + contracts.yaml) -> render (Jinja2 templates).
- Team decisions live in contracts.yaml, never hard-coded in Python.
- LLM only for planning and failure diagnosis; strict JSON output (pydantic), chosen from an action menu.
- The LLM never writes whole files. It only fills template slots as JSON, choosing from
  candidate lists the scanner provides (see docs/llm-spike-1.md).
- Free/local LLM via Ollama. Few calls, small context. All state stays in Python, not in the LLM.
- Sandbox: docker build + `make all SAMPLE=1` in a throwaway container. Max 3 fixes, every fix verified.
- Store both successful and failed fixes in memory.
- Log every LLM call and build attempt (prompt, response, model digest, latency, outcome).
- Never touch: rollout/router config (Deployer), runtime schema checks (Inspector), PR gating (Reviewer).
- Never modify the target repo's existing files; only add generated files.

## Test repos
- Dev repo: ../taxi-trip-regression (fork; its Dockerfile was removed on purpose)

## Lessons from the reference run
Details: docs/reference-run-taxi.md
- Smoke test must fail on an "error" key in the response body, not only on HTTP status
  (the taxi app returns errors with HTTP 200).
- Dockerfiles install dependencies before copying code, so the dependency layer stays cached.
- Scanner should later find or build a sample request for the smoke test
  (for taxi it came from a notebook cell and the test.csv header).

## Lessons from LLM spike 1
Details: docs/llm-spike-1.md
- Run pyflakes on generated code before the sandbox (both spike adapters parsed but had undefined names).
- Smoke test must compare a known prediction from the reference run, not only the response shape
  (taxi: row 1 of data/processed/test.csv -> about 531 s); a dropped inverse transform still returns 200.

## Decisions
- needs_llm items are filled with thin adapter scripts in the target repo's
  `pipeline/` folder (contracts.yaml paths.adapters_dir, owned by builder):
  train.py, evaluate.py, serve.py, data.py, sample_request.json.
  Adapters only call functions the repo already has; they never modify the repo's files.
- Adapters are Jinja2 templates. The LLM only fills their slots (e.g. which function to call,
  target column, inverse transform) as strict JSON, choosing from candidate lists the scanner provides.
- The message board service will be added to compose.base.yml once the board owner provides it
  (contracts.yaml compose.base_file lists it; not there yet).
- The MLflow client is pinned to the server image's version, derived from its tag at render time
  (v2.17.2 -> mlflow==2.17.2). Fault case 1 (tests/faults/mlflow_client_server_mismatch): the
  unpinned 3.x client got 404 from the 2.17.2 server when logging the model.

## Fault cases
Recorded failures live in tests/faults/<name>/case.json (stage, error tail, cause, fix,
how to reproduce); tests/test_faults.py checks their shape. They will seed the incident memory.

## Current step
Step 1 done: repo scanner in builder_agent/scan (`python -m builder_agent.scan <repo>`), returns RepoContext + summary().
Step 2 done: decide layer in builder_agent/decide (`python -m builder_agent.decide <repo>`), RepoContext + contracts.yaml -> BuildPlan, no LLM; marks what rules can't settle as needs_llm.
Step 3 done: adapters in builder_agent/render: SlotAnswers + slot_candidates() + validate_slots() (slots.py),
  Jinja2 templates for pipeline/{train,evaluate,serve,data}.py and sample_request.json, pyflakes on output.
  Gold answers for taxi: tests/gold/taxi_slots.json (validate and render cleanly; adapters not run yet).
Step 4 done: config templates in builder_agent/render/configs.py: Dockerfile, .dockerignore, Makefile,
  compose.base.yml, with static lint + `docker compose config`.
Step 5 done: sandbox runner in builder_agent/sandbox (`python -m builder_agent.sandbox <repo> --slots ... --expected ...`):
  temp copy (real data mounted read-only), render, build, mlflow, data/train/evaluate with SAMPLE=1, serve,
  health, predict; stops at the first failure; attempts in logs/sandbox_attempts.jsonl.
  pipeline/smoke_test.py runs the contract's smoke_test (in-process for `make test`, --url over HTTP).
  Taxi with the gold slots: all stages ok (prediction 472 s for the reference row; expected 531 s +-50%).
