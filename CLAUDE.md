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
- Free/local LLM via Ollama. Few calls, small context. All state stays in Python, not in the LLM.
- Sandbox: docker build + `make all SAMPLE=1` in a throwaway container. Max 3 fixes, every fix verified.
- Store both successful and failed fixes in memory.
- Log every LLM call and build attempt (prompt, response, model digest, latency, outcome).
- Never touch: rollout/router config (Deployer), runtime schema checks (Inspector), PR gating (Reviewer).
- Never modify the target repo's existing files; only add generated files.

## Test repos
- Dev repo: ../taxi-trip-regression (fork; its Dockerfile was removed on purpose)

## Current step
Step 1: repo scanner (pure Python, no LLM). Output a RepoContext pydantic model.
