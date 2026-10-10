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

*Plan* (`builder_agent/faults/`, taxi dev repo first):

- A fault is injected as a change to what the Builder generates, never to the repo's files: build
  settings (pins, Python version, environment variables) plus two injection hooks the LLM's menu
  can't use (commands before the system-package install, and after the repo's install). Fixes are
  applied on top, so a correct fix undoes the fault.
- Each case: run the sandbox with the fault (it must fail), keep the failed stage, the last 50 lines
  and the error line; then run the sandbox again with the expected fix and record whether it passes.
  Saved in the same shape as `tests/faults/`, under `tests/faults/generated/`, with its family,
  variant, injection and expected fix from the action menu.
- Families and variants (several per family; one per family first):

  | Family | Variants | Expected fix |
  |---|---|---|
  | 1 client/server version mismatch | mlflow 3.1.4 pinned, 3.0.x, unpinned (fault-001) | `pin_package mlflow <server version>` |
  | 2 missing Python dependency | xgboost uninstalled; others MLflow doesn't depend on (pandas, flask, gunicorn, numpy come back with `pip install mlflow`) | `add_dependency <name>` |
  | 3 incompatible pin | numpy 2.0.2 with pandas built for numpy 1.x; xgboost 3.x on Python 3.9; old flask with new werkzeug | `pin_package <name> <working version>` |
  | 4 wrong Python version | 3.12, 3.13, 3.8 for a Pipfile that asks for 3.9 | `set_python_version 3.9` |
  | 5 missing environment variable | tracking URI: wrong host, wrong port, empty | `set_env_var <name> <value>` |
  | 6 missing system library | libffi8 (ctypes), libsqlite3-0 (sqlite3) removed; not libgomp1 (xgboost wheels bundle it) and not libssl3/libbz2 (apt needs them) | `add_system_package <name>` |

- Generated so far (`tests/faults/generated/`, gen-001..013, taxi dev repo): every saved case breaks the
  sandbox and passes with its expected fix. Injections before `pip install mlflow==...` are undone
  when they touch MLflow's dependencies, so version faults are injected as Builder pins.
- Tried and not saved (second batch, 2026-10-09):

  | Variant | Why not saved |
  |---|---|
  | `flask_uninstalled` | didn't break: `pip install mlflow==2.17.2` reinstalls flask (an MLflow dependency) |
  | `gunicorn_uninstalled` | inconclusive: the build failed on a PyPI read timeout, not on the fault, so the case was removed; MLflow also requires gunicorn, so it likely doesn't break |
  | `python_3_8` | didn't break: every stage passed on python:3.8-slim |
  | `libsqlite3_removed` | bad injection: removing libsqlite3-0 breaks apt (liblastlog2-2 depends on it), so the fault is an apt error, and the fix run failed at evaluate |

  On the taxi path, family 2 has only one variant that breaks (xgboost): everything else it imports is
  in MLflow's dependency closure. Family 6 has only libffi8: no other removable library breaks
  the taxi imports without also breaking apt. More variants for these families need another repo or
  a different injection.
- Later: split by **variant** into RAG memory and a held-out test set, with no variant in both, so
  RAG is never tested on a fault it has stored.

1. **Prompting.** Zero-shot (`diagnose_v1`, the baseline) vs Chain-of-Thought. Self-consistency
   (several answers at a temperature above 0, majority vote) only if the CoT answers vary.
   *Result on fault-001* (qwen2.5-coder:7b, `experiments/fixes/`): CoT (`diagnose_v2`) quoted the
   right error line (the 404 on `/logged-models`) in every analysis but didn't fix the fault: not
   fixed after 3 fixes, while zero-shot passed on its third fix with a guessed `mlflow==2.10.0`.
   The model finds the symptom but lacks the knowledge to connect it to the MLflow client/server
   version mismatch: a knowledge gap, not a reasoning-format problem. This is consistent with
   Wei et al. 2022 ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"), where
   CoT helps mainly large models. **So RAG is next** (item 2), to bring that knowledge in.
2. **RAG over the fault memory** (`tests/faults/` and the generated faults). Naive: vectors only.
   Advanced: hybrid (BM25 + vectors), reranking, and a filter on the failed stage. Agentic RAG
   (a PyPI version lookup as a tool) only if the version problem remains after that.
3. **Fine-tuning.** Base model vs QLoRA. Single-task vs multi-task (slots + diagnosis) only if
   there is enough data.
4. **DPO from preference pairs.** Automatic pairs from sandbox pass/fail, plus about 50 pairs
   labeled by hand between two passing fixes. Optionally an LLM judge, checked against those
   hand labels.
5. **Already done:** slots v1 vs v2, zero-shot vs few-shot (`experiments/evals/`).

**Status after the QLoRA run** (`docs/RESULTS.md` sections 4–6). RAG (`diagnose_v3` + the advanced
retriever) is the best configuration: 2 of 4 first fix passes, 3 of 4 faults fixed. QLoRA on 33
examples tied with it. gen-011 (Python 3.13) is unsolved by every model and prompt, and the v3 prompt
never states the image's Python version. Next steps, in order:

- (a) **`diagnose_v5`** = v3 plus one line that states the image's and the Pipfile's Python versions
  side by side, facts only (no hint that they should match). Run it on the base model and contrast
  it with v4: one targeted fact vs a block of build settings.
  *Done* (`docs/RESULTS.md` section 7): fixed 4 of 4 (v3: 3 of 4), gen-011 for the first time, on the
  third fix; first fix passes 1 of 4 (v3: 2 of 4), because gen-010 got worse. A mixed result, so
  whether it "helps" enough for (b) is still to decide.
- (b) **Only if v5 helps:** regenerate the fine-tuning data with v5 prompts and retrain, since the
  training and test prompts must match.
- (c) **One more Python-version memory fault.** The planned Python 3.8 fault didn't break when tried
  (2026-10-09, every stage passed), so it can't be added as is. Try 3.7 first, and drop this step if
  it doesn't break either.
- (d) **DPO** on the existing 34 pairs (`experiments/finetune/dpo.jsonl`).
- (e) **For the final discussion:** a deterministic Python-version check outside the LLM (image vs
  Pipfile), reported separately from the LLM-only results.

**Not used, with reasons:**

- Tree of Thoughts: each branch would need its own sandbox run.
- Generated knowledge and prompt chaining: RAG and the small action menu cover them.
- Full fine-tuning and prompt tuning: QLoRA is enough for a small local model.
- PPO: DPO gets the preference signal without a reward model.
- No multimodal in the Builder.
