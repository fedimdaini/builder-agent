# Builder roadmap

Next phase: from the taxi dev repo to the team's main project
(`C:/dev/mlops-zoomcamp-project`). Done so far: steps 1–5 in CLAUDE.md, plus the
slot-filling loop and its evals (`experiments/evals/`).

**Rule for the whole phase: the main project is a held-out test. Don't tune prompts on it.**

## 1. Reference run of the main project

- Run it by hand, following the team guide. Quit Ollama first (RAM).
- Record the steps, problems, build time and peak memory in `docs/reference-run-main.md`.

## 2. Fault cases from the guide's setup fixes

Save the guide's setup fixes as fault cases in `tests/faults/`:

| Case | Fix |
|---|---|
| 2 | `localstack:stable` → `4.12` |
| 3 | `postgres:latest` → `16` |
| 4 | `core.autocrlf=false` |

## 3. contracts.yaml with the project's real values

- MLflow URI inside Docker, experiment `mlops-zoomcamp-experiment`, service names.
- Keep the API contract (`/health`, one record per request, `"prediction"` field).
  The Builder bridges the difference (see 8).

## 4. Builder modes

| Mode | When | What it does |
|---|---|---|
| create | no pipeline files | generates them all |
| complete | some pipeline files | generates the missing ones |
| verify | pipeline files exist | checks them and proposes fixes as pull requests; never overwrites teammates' files |

The mode is chosen from the scanner's `existing_pipeline_files`.

## 5. Scanner extensions for the main project

- Airflow DAGs as entry points.
- Parquet headers.
- A computed target (dropoff minus pickup).

## 6. CI workflow template

GitHub Actions workflow: the task the team guide gives the Builder.

## 7. Verify-mode static checks

- Unpinned or `:latest` image tags.
- CRLF in `.sh` files.
- Missing healthchecks.
- Contract gaps.

Goal: catch the guide's setup fixes (fault cases 2–4) automatically.

## 8. API adapter

Bridge the project's batch `/predict` and `/reload` to the contract.

## 9. LLM work, each step measured

- **Prompt versions:** v3 (a reasoning field, if needed); v1 with better feedback; a rerun at
  temperature 0.7 with 5 seeds for variance (at temperature 0 the seed has no effect).
- **RAG fix memory:** compare dense, BM25, and hybrid + reranking retrieval, with a metadata
  filter by stage.
- **Fine-tuning:** LoRA vs QLoRA on the logged slot decisions; distillation from a bigger free model.
- **DPO** on preference pairs from sandbox pass/fail.
- No multimodal in the Builder.
