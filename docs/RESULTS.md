# LLM experiment results

All LLM experiments of the Builder so far, in one place. Every run used **qwen2.5-coder:7b** through
Ollama (temperature 0, seed 42, context 8192) on the taxi dev repo (`../taxi-trip-regression`). The
team's main project is a held-out test and was not used for any of these.

**Best configuration so far:** the fix loop with **`diagnose_v3` + the advanced retriever** (see 3 and 4).

The LLM does two jobs, and both answer as strict JSON:

- **Slot filling:** it chooses the values the adapter templates need (target column, train function,
  transform, ...) from candidate lists the scanner provides.
- **Diagnosis:** when a sandbox stage fails, it picks one fix from a small menu (`pin_package`,
  `add_dependency`, `add_system_package`, `set_python_version`, `set_env_var`, `give_up`). The fix
  is applied and checked with a new sandbox run, at most 3 fixes per fault.

---

## 1. Slot filling: `slots_v1` vs `slots_v2`

**What changed.** `slots_v2` is `slots_v1` with wording that matches what the validator checks,
clearer slot definitions and **one worked example** of an `arg_map` (zero-shot vs one-shot). In
between, the scanner and validator were fixed after the first v1 run ("fixed system").

| Run | Slots right (strict / lenient, of 11) | Valid on first try | Tries to a valid answer | Tokens | Sandbox end to end |
|---|---|---|---|---|---|
| v1, original system | 8 / 9 | no | 3 (valid at the end) | 8,836 | not run |
| v1, fixed system | 8 / 9 | no | 3 (**never valid**) | 9,049 | not run |
| v2, fixed system | 9 / 10 | no | 2 | 5,889 | all stages ok |
| v2, better feedback, 5 seeds | 9 / 10 in every run | 0 of 5 | 2 in every run | 6,047 | all stages ok, 5 of 5 |

"Lenient" also accepts an alternative the gold file allows (`train.csv` for the training file). The
one remaining miss is `evaluate.eval_file`: `validation.csv` instead of `test.csv`.

**Lesson.** The worked example and clearer wording helped more than anything else: fewer tries, a
third fewer tokens, and an answer that runs end to end. At temperature 0 the seed changes nothing (5
seeds, one distinct answer). Clear rejection reasons from the validator matter: the model fixes
what the reason names.

Logs: `experiments/evals/` (one JSON report per run, plus the 5-seed summary).

---

## 2. Diagnosis on fault-001: zero-shot (`diagnose_v1`) vs Chain-of-Thought (`diagnose_v2`)

**What changed.** `diagnose_v1` asks for the fix from the error alone. `diagnose_v2` (the user's
Chain-of-Thought prompt) asks for a step-by-step `analysis` field **before** the fix.

Fault-001: the MLflow client 3.x can't log models to the 2.17.2 tracking server (HTTP 404 on
`/logged-models`). Expected fix: `pin_package mlflow 2.17.2`.

| Prompt | Fix 1 | Fix 2 | Fix 3 | Result |
|---|---|---|---|---|
| v1 zero-shot | `set_env_var GIT_PYTHON_REFRESH` | `set_env_var MLFLOW_TRACKING_URI` | `pin_package mlflow 2.10.0` | **passed** after 3 (a lucky guess) |
| v2 CoT | `set_env_var GIT_PYTHON_REFRESH` | `set_env_var MLFLOW_TRACKING_URI` | `add_system_package mlflow` (broke the build) | not fixed |

**Lesson.** CoT quoted the right error line (the 404) in every analysis but still didn't link it to a
client/server version mismatch. The model can find the symptom; it lacks the knowledge of what
causes it. That is a knowledge gap, not a reasoning-format problem (consistent with Wei et al. 2022:
CoT helps mainly large models). So the next step was to bring the knowledge in with RAG. CoT was
not used further.

A validator rule came out of it: `add_system_package` is rejected for a name that is a Python
package (`mlflow` is not a Debian package).

Logs: `experiments/fixes/fault-001__diagnose_v1__qwen2.5-coder-7b/`, `…__diagnose_v2__…/`.

---

## 3. Retrieval only: naive vs advanced retriever

**Setup.** A fix memory of 10 past failures: fault-001 plus the 9 generated faults in the
**memory** split (`tests/faults/generated/`, split by variant). Each document holds the failed stage,
the error line, the cause and the verified fix. Embeddings: `nomic-embed-text` through Ollama;
vector store: Chroma. The 4 **test**-split faults are never indexed (a test fails if one is).

- **Naive:** the full error output as the query, vectors only, top 3.
- **Advanced:** the extracted error line as the query, BM25 + vectors merged with reciprocal rank
  fusion, and a small boost (not a filter) for past failures from the same stage, top 3.

| Test fault | Family | Naive: same family in top 3 | Advanced: same family in top 3 |
|---|---|---|---|
| gen-001 MLflow client 3.1.4 | version mismatch | yes (rank 1) | yes (rank 1) |
| gen-010 flask 2.0.3 | incompatible pin | no | yes (rank 3) |
| gen-011 Python 3.13 | Python version | no | yes (rank 2) |
| gen-013 empty tracking URI | env var | yes (rank 3) | yes (rank 2) |
| **Total** | | **2 of 4** | **4 of 4** |

**Lesson.** The full error output is long and noisy (warnings, tracebacks, pip chatter), so naive
vector search often matched the noise. Querying with the extracted error line plus keyword search
found a relevant case every time.

Logs: `experiments/rag/retrieval.md` and `retrieval.json`.

---

## 4. Fix loop on the 4 held-out faults: v1, v3 naive, v3 advanced, v4 advanced

**What changed.**

- `diagnose_v1`: the baseline (error output only).
- `diagnose_v3`: v1 plus the 3 retrieved past failures with their verified fixes (naive or advanced).
- `diagnose_v4`: v3 plus the current build settings (Python version of the image, system packages,
  pins, environment), rebuilt for every attempt.

**Main score:** does the first fix pass the sandbox, and how many fixes until it passes (max 3).
"True reasons" counts the fixes whose stated cause matches the real error, even when the chosen
package is wrong. Exact/loose match with the expected fix is kept in each run's `score.json`; it
undercounts, because a different fix can also be right (gen-010: pinning werkzeug instead of flask).

Each cell: first fix passes / fixes to pass / true reasons.

| Fault | Baseline v1 | Naive v3 | Advanced v3 | Advanced v4 |
|---|---|---|---|---|
| gen-001 MLflow client 3.1.4 | no / 3 / 2 of 3 | yes / 1 / 1 of 1 | yes / 1 / 1 of 1 | yes / 1 / 1 of 1 |
| gen-010 flask 2.0.3 | yes / 1 / 1 of 1 | yes / 1 / 1 of 1 | yes / 1 / 1 of 1 | no / 3 / 1 of 3 |
| gen-011 Python 3.13 | no / not fixed / 3 of 3 | no / not fixed / 3 of 3 | no / not fixed / 3 of 3 | no / not fixed / 2 of 3 |
| gen-013 empty tracking URI | no / not fixed / 0 of 3 | no / 2 / 0 of 2 | no / 2 / 0 of 2 | no / 2 / 0 of 2 |
| **First fix passes** | 1 of 4 | 2 of 4 | 2 of 4 | 1 of 4 |
| **Fixed** | 2 of 4 | 3 of 4 | 3 of 4 | 3 of 4 |
| **True reasons** | 6 of 10 | 5 of 7 | 5 of 7 | 4 of 9 |

The gen-011 cells for v1 and v3 are reruns after two system fixes (below); before them, both v3
runs also ended "not fixed".

### What happened, per fault

- **gen-001.** With memory, the model copied the verified fix (`mlflow 2.17.2`) on the first try.
  Without it, it needed 3 fixes and guessed `2.10.0`.
- **gen-010.** Fixed on the first try with a werkzeug pin in v1 and v3: the model already knows this
  fault. In v4 it first set Python 3.9 on an image that already had 3.9, then copied
  `add_system_package libffi8` from a retrieved case, and passed only on the third fix.
- **gen-011.** Not fixed in any condition. The build fails because Python 3.13 has no wheels for the
  repo's locked packages, so pip compiles them. The model chases each compile error in turn (C++
  compiler, then GDAL for fiona) and never links them to the Python version, even with the right past
  failure retrieved (python_3_12, fix `set_python_version 3.9`) and, in v4, the base image
  `python:3.13-slim` in the prompt. No reason in any run says the image's Python version is the
  problem.
- **gen-013.** With memory, fixed on the second try with the right `set_env_var`, but the stated
  reason ("the URI uses port 5001") was copied from the retrieved wrong-port case. In v4 the prompt
  showed `MLFLOW_TRACKING_URI=""`, and the reason still said port 5001.

### Lessons

- **RAG helps where the model lacks knowledge.** v3 fixed 3 of 4 faults instead of 2, and halved
  the fixes on gen-001. The two retrievers tied in the fix loop, even though the advanced one
  retrieves better (section 3).
- **v4 is a negative result.** Adding the current build settings made the 7B model worse: first
  fix passes 1 of 4 (v3: 2 of 4), true reasons 4 of 9 (v3: 5 of 7). More context **distracted** it
  (gen-010, which it fixed first time without the settings), and **reasons copied from memory
  overrode the evidence** in the prompt (gen-013: "port 5001" while the settings showed an empty
  URI). For this model size, less context is better.
- **The model copies the reason as well as the fix.** RAG raised the number of right fixes but not
  the number of true reasons. A right fix with a wrong reason is a risk: it only works when the
  retrieved case happens to need the same fix.
- **Capacity limit.** Linking a chain of build errors back to their root cause (gen-011) is beyond
  this model with these prompts. That is the case for fine-tuning (roadmap item 9 step 3).

### System fixes that came out of these runs

These change what the validator accepts; they don't depend on the model.

- `add_system_package` rejects names with `+ * ? [ ]`: apt reads them as patterns (gen-011:
  `clang++` installed conflicting clang packages and broke apt for the rest of the loop).
- Regression guard: a fix that makes an earlier stage fail is undone and counted as a failed attempt.
  It did not fire in the gen-011 reruns, because every failure stayed in the build stage.
- `set_python_version` is rejected when the image already has that version (gen-010 in v4). Added
  after the v4 runs, so the v4 numbers above don't include it.
- The fix loop records the injected fault separately from the fixes, and every attempt's retrieved
  cases and model call are logged.

Logs: `experiments/fixes/gen-NNN-<variant>__<prompt>[-<retriever>]__qwen2.5-coder-7b/`
(`summary.txt`, `fix_attempts.jsonl` with every model call, `sandbox_attempts.jsonl`, `score.json`);
the gen-011 reruns end in `__guard`. Prompts: `prompts/diagnose_v1.md` to `diagnose_v4.md`.

---

## 5. Fine-tuning data from the memory faults (sampling, then STaR rationalization)

**What was done.** Training data for QLoRA and DPO, from the 9 memory faults only (the 4 test faults
are never used; a test checks it).

1. **Sampling.** For each fault, the `diagnose_v3` prompt as the fix loop renders it, with the advanced
   retriever and the fault itself left out of memory (so the model can't copy it). 8 answers at
   temperature 0.7.
2. **Labels.** Correct = the verified fix, the same package with another version, or another fix
   that passed the sandbox. A correct answer is used for training only if its reason is
   **faithful**: it names a term of this fault's error line, and nothing copied from a retrieved
   other fault or false about the image.
3. **Rationalization (STaR, Zelikman et al. 2022)** for the missing reasons. The same prompt plus a
   hint with the verified fix; 4 answers; kept if faithful and not mentioning the hint. If none is
   kept, a reason is written from the recorded case. The training prompt never contains the hint.

| | Sampled answers | Correct | Correct but unfaithful | SFT examples (sampled / rationalized / written) | DPO pairs |
|---|---|---|---|---|---|
| Before rationalization | 72 | 45 | 17 | 28 (28 / 0 / 0) | 4 |
| After rationalization | 72 | 45 | 17 | **53** (28 / 24 / 1) | **34** |

Per family, SFT examples: version mismatch 5, missing dependency 11, incompatible pin 16, Python
version 1, env var 8, system library 12. DPO pairs: 12, 0, 7, 8, 7, 0.

**Lessons.**

- **38% of the correct fixes had copied reasons** (for example "mlflow==3.0.1" or "mlflow-server"
  from a retrieved fault). Fine-tuning on them would teach the habit that made v4 worse (section 4).
- **The hint itself leaks.** 4 rationalized reasons quoted it ("the verified fix is ..."). They are
  filtered out, because the training prompt has no hint.
- **The Python-version blind spot again.** gen-004 (Python 3.12) is the only fault with no reason from
  the model, even when told the fix. Its one example is written, the same failure as the held-out
  gen-011.
- **Small and unbalanced.** 53 examples and 34 pairs from 9 faults. Two families have no DPO pairs
  because the model was always right. Enough to try QLoRA as a direction, not to measure it.

Logs: `experiments/finetune/` (`report.md`, every model call in `samples.jsonl` and
`rationalized.jsonl`, labels, sandbox checks, `sft.jsonl`, `dpo.jsonl`).

---

## Not run yet

- Fine-tuning (QLoRA) and DPO training (roadmap item 9 steps 3 and 4); the data is in section 5.
- Families 2 (missing dependency) and 6 (system library) have no held-out test case on the taxi repo:
  each has only one variant that breaks, and it is in memory.
- 4 test faults are few: one fault can change a total by 25 points. Read the tables as directions,
  not as measurements.
