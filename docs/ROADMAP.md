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

## 9. LLM work: one comparison per technique family

Each technique is measured against a baseline on the same faults, with the same scoring
(exact / loose fix, sandbox pass, tokens, time). The main project stays a held-out test.

**First: a fault generator.** RAG, fine-tuning and DPO need many faults, not six. Generate them
(and their known fixes) before comparing those techniques.

1. **Prompting.** Zero-shot (`diagnose_v1`, the baseline) vs Chain-of-Thought. Self-consistency
   (several answers at a temperature above 0, majority vote) only if the CoT answers vary.
2. **RAG over the fault memory** (`tests/faults/` and the generated faults). Naive: vectors only.
   Advanced: hybrid (BM25 + vectors), reranking, and a filter on the failed stage. Agentic RAG
   (a PyPI version lookup as a tool) only if the version problem remains after that.
3. **Fine-tuning.** Base model vs QLoRA. Single-task vs multi-task (slots + diagnosis) only if
   there is enough data.
4. **DPO from preference pairs.** Automatic pairs from sandbox pass/fail, plus about 50 pairs
   labeled by hand between two passing fixes. Optionally an LLM judge, checked against those
   hand labels.
5. **Already done:** slots v1 vs v2, zero-shot vs few-shot (`experiments/evals/`).

**Not used, with reasons:**

- Tree of Thoughts: each branch would need its own sandbox run.
- Generated knowledge and prompt chaining: RAG and the small action menu cover them.
- Full fine-tuning and prompt tuning: QLoRA is enough for a small local model.
- PPO: DPO gets the preference signal without a reward model.
