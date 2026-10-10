# LLM experiment results

All LLM experiments of the Builder so far, in one place. Every run used **qwen2.5-coder:7b** through
Ollama (temperature 0, seed 42, context 8192) on the taxi dev repo (`../taxi-trip-regression`);
section 6 also uses its QLoRA fine-tune, `qwen-builder-sft`. The team's main project is a held-out test and was not used for any of these.

**Best configuration so far:** the fix loop with **`diagnose_v3` + the advanced retriever** (see 3 and 4).
The QLoRA fine-tuned model (section 6) tied with the base model in it, so the base model stays.
`diagnose_v5` (section 7) fixed all 4 held-out faults but passed fewer on the first fix (1 vs 2 of 4).

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
| v3 (v2 + task slot), seed 42 | 9 / 10, task `regression` | no | 2 | 6,699 | all stages ok |

"Lenient" also accepts an alternative the gold file allows (`train.csv` for the training file). The
one remaining miss is `evaluate.eval_file`: `validation.csv` instead of `test.csv`.

**Lesson.** The worked example and clearer wording helped more than anything else: fewer tries, a
third fewer tokens, and an answer that runs end to end. At temperature 0 the seed changes nothing (5
seeds, one distinct answer). Clear rejection reasons from the validator matter: the model fixes
what the reason names.

**v3** (2026-10-10) adds one question, the task (regression or classification), and a TASK line
with the scan rule's decision (for taxi: regression, from 4 agreeing signals). On taxi it scores the
same as v2 (9 strict / 10 lenient of 11, the same eval_file miss) and answers `regression`. It is
valid on the second try, like v2: the first answer had arg_map key and transform flag mistakes, the
same kind v2 made. Tokens rise by about 650, mostly the TASK lines. One seed only. The task question
matters only for repos where the scan signals don't settle the task, and taxi isn't one, so this run
shows that v3 doesn't regress, not that the question helps. The sandbox run used `train.csv` (an
allowed alternative) and predicted 410 s for the reference row (expected 531 ± 50%).

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
- **Small and unbalanced.** 56 examples (33 after the cap) and 34 pairs from 9 faults. Two families
  have no DPO pairs because the model was always right. Enough to try QLoRA as a direction, not to
  measure it.
- **The filter is not a fully independent judge.** It was refined three times after looking at the
  data: (1) the fault's own recorded cause no longer counts as copied; (2) the Python check flags only
  claims about the image, not requirements; (3) the version the fix sets ("set the image back to
  Python 3.9") is not a false claim. Each change was logged and tested, but each was made while
  looking at the answers the filter judges, so its verdicts on this data are partly fitted to it. To
  make it independent: freeze the filter (its code and term lists) before labelling new data, and
  check a random sample of its verdicts by hand, kept and dropped, to measure how often it is wrong.

**Two later additions.**

- **Human reasons:** a hand-written source (`human_reasons.json`), checked by the same filter, used
  for DPO "chosen" before written reasons. 3 reasons for gen-004 (Python 3.12), the fault the model
  couldn't explain; all 3 pass. SFT: **56** examples (28 sampled, 24 rationalized, 3 human, 1 written).
- **A cap of 4 examples per fault for training**, human first. The 56 SFT examples become **33**
  (16 sampled, 13 rationalized, 3 human, 1 written); the Python-version family goes from 1 example to
  4. Without the cap, the 3 faults the model always got right would be two thirds of the data.

**Training run.** `notebooks/qlora_sft.ipynb` on Colab (T4): Qwen2.5-Coder-7B-Instruct in 4-bit,
LoRA r=16, 3 epochs over the 33 capped examples: 27 steps, 974 s, train loss 0.254. Versions:
unsloth 2026.10.3, torch 2.11.0+cu130, transformers 5.17.0, trl 1.13.0, peft 0.21.1,
bitsandbytes 0.50.2. Data: `sft.jsonl`, sha256 `337c80f3…`. Exported as GGUF Q4_K_M and created in
Ollama as `qwen-builder-sft`. The notebook's export cell crashed after the GGUF was written (Unsloth
saves it in `qwen-builder-sft_gguf/`, which the cell didn't search, and the Modelfile came after the
search); the cell is fixed. Results in section 6.

Logs: `experiments/finetune/` (`report.md`, sandbox checks, human reasons) and `experiments/finetune/v3/`
(every model call in `samples.jsonl` and `rationalized.jsonl`, labels, `sft.jsonl`, `dpo.jsonl`).

---

## 6. Fix loop with the fine-tuned model: `qwen-builder-sft` vs the base model

**Setup.** The same as the best configuration in section 4: `diagnose_v3`, the advanced retriever, the
4 held-out faults, temperature 0, seed 42. Only `--model` changes: `qwen-builder-sft` (QLoRA on the
33 capped examples of section 5) instead of `qwen2.5-coder:7b`. The base column is the v3-advanced
column of section 4 (gen-011: the `__guard` rerun).

Each cell: first fix passes / fixes to pass / true reasons. "Copied" = reasons that name a term only a
retrieved other fault has (its error line, or a version, port or library of its cause), and that is
not in this attempt's error output, the fix itself or the fault's own cause: the rule of the
fine-tuning filter (`finetune.foreign_terms`), applied to every fix of the run.

| Fault | Base: result | Base: copied | Fine-tuned: result | Fine-tuned: copied |
|---|---|---|---|---|
| gen-001 MLflow client 3.1.4 | yes / 1 / 1 of 1 | 0 of 1 | yes / 1 / 1 of 1 | 0 of 1 |
| gen-010 flask 2.0.3 | yes / 1 / 1 of 1 | 1 of 1 ("3.9", a false positive) | yes / 1 / 1 of 1 | 0 of 1 |
| gen-011 Python 3.13 | no / not fixed / 3 of 3 | 0 of 3 | no / not fixed / 3 of 3 | 0 of 3 |
| gen-013 empty tracking URI | no / 2 / 0 of 2 | 2 of 2 ("3.0.1"; "5001") | no / 2 / 0 of 2 | 2 of 2 ("3.x"; "5001") |
| **First fix passes** | 2 of 4 | | 2 of 4 | |
| **Fixed** | 3 of 4 | | 3 of 4 | |
| **True reasons** | 5 of 7 | **3 of 7** copied | 5 of 7 | **2 of 7** copied |

The base gen-010 hit is not a real copy: "works with Python 3.9" is true of the image, but 3.9 is not in
the error output, so the rule flags it. Without it, both models copy in the same 2 fixes (gen-013).

### What happened, per fault

- **gen-001, gen-013, gen-011: the same fixes as the base model**, with reworded reasons. gen-013 still
  pins mlflow first ("client 3.x is not compatible with the server", copied from the retrieved MLflow
  client faults), then sets the right URI with the reason "the client connects to port 5001", copied
  from the retrieved wrong-port fault. gen-011 still chases compiler and GDAL errors (fix 2 is now
  `add_dependency gdal` instead of `add_system_package gdal-bin`) and never names the Python version.
- **gen-010: a new mistake, caught by the validator.** The first call answered `set_python_version 3.9`
  ("aligns with the requirements in the Pipfile"), on an image that already runs 3.9. The validator
  rule added after section 4's v4 runs rejected it, and the second call pinned `werkzeug 2.0.3`, which
  passed. That counts as one fix, so the cell reads "yes / 1", but without that rule the first fix
  would have been a no-op. The base model never proposed it. Likely source: 4 of the 33 training
  examples (gen-004) answer `set_python_version 3.9`. The model reused that answer where it doesn't
  fit, and still didn't give it on gen-011, where it is the fix.

### Lessons

- **No measurable gain from SFT on 33 examples.** Same first fix passes (2 of 4), same faults fixed
  (3 of 4), same true reasons (5 of 7). 4 faults can't show a small effect either way.
- **Copied reasons remain.** The training data had only faithful reasons, but in the fix loop the
  model still took "port 5001" and "client 3.x" from retrieved cases on gen-013. 33 examples didn't
  remove the habit.
- **The Python-version blind spot remains.** Training included 4 examples where the fix is
  `set_python_version` (gen-004, 3 of them human); the model learned the answer as a phrase
  (proposed wrongly on gen-010), not when to use it (gen-011).
- **The validator matters more with the fine-tuned model.** The rejection of a version the image
  already has turned a wasted fix into a pass.

### Why it didn't improve

- **It learned surface pairings, not the cause.** It proposed `set_python_version 3.9` where it
  doesn't fit (gen-010, the image is already 3.9) and not where it does (gen-011). With one training
  fault per family, the model can't tell which part of an example matters: the error text or the cause.
- **The evidence was missing from the prompt.** `diagnose_v3`'s repository facts state the Pipfile's
  Python version, as a hint (`PYTHON HINTS: 3.9 (pipfile, pipfile-lock, pyc-cache)`), but never the
  image's. On gen-011 the image version appears only inside file paths in the error output
  (`/usr/local/lib/python3.13/subprocess.py`). So neither the base nor the fine-tuned model was shown
  the mismatch. Fine-tuning can't teach a comparison whose inputs are absent.
- **More data is limited on the taxi repo.** No other image version is known to give a new
  Python-mismatch error. 3.8 was tried on 2026-10-09 and every stage passed (ROADMAP item 9). 3.10
  passed the sandbox checks (section 5). 3.11 has wheels for every heavy pin, so it likely passes.
  3.7 is untested: it might fail on the pins that need Python ≥ 3.9 (geopandas 0.14.1, pyproj 3.6.1,
  scipy 1.11.1), but 3.8 should have failed on the same pins and didn't.
- **Caveat.** 4 test faults, one run each at temperature 0. A tie means no measurable gain, not proof
  of no effect.

Logs: `experiments/fixes/gen-NNN-<variant>__diagnose_v3-advanced__qwen-builder-sft/`; the copied-term
check per fix, for both models: `experiments/finetune/fixloop_copied_terms.json`. Section 7 tests
the missing-evidence point with `diagnose_v5`.

---

## 7. One targeted fact: `diagnose_v5` vs v3 (and v4)

**What changed.** `diagnose_v5` is `diagnose_v3` plus one line at the end of the repository facts,
facts only, no verdict. The image comes from the generated Dockerfile's `FROM` line, the rest from
the repo's own files; it is rebuilt for every attempt:

```
PYTHON: image python:3.13-slim (Python 3.13); Pipfile python_version 3.9; Pipfile.lock built for 3.9
```

Same setup as section 4: base model qwen2.5-coder:7b, advanced retriever, the 4 held-out faults,
temperature 0, seed 42. A first attempt was cut off by a power loss before any sandbox run or model
call was logged; it left no files, and all 4 faults were run again from scratch.

Each cell: first fix passes / fixes to pass / true reasons. "Copied" uses the rule of section 6.

| Fault | v3: result | v3: copied | v5: result | v5: copied |
|---|---|---|---|---|
| gen-001 MLflow client 3.1.4 | yes / 1 / 1 of 1 | 0 of 1 | yes / 1 / 1 of 1 | 0 of 1 |
| gen-010 flask 2.0.3 | yes / 1 / 1 of 1 | 1 of 1 (false positive) | no / 2 / 1 of 2 | 1 of 2 |
| gen-011 Python 3.13 | no / not fixed / 3 of 3 | 0 of 3 | **no / 3 / 3 of 3** | 0 of 3 |
| gen-013 empty tracking URI | no / 2 / 0 of 2 | 2 of 2 | no / 2 / 0 of 2 | 2 of 2 |
| **First fix passes** | 2 of 4 | | 1 of 4 | |
| **Fixed** | 3 of 4 | | **4 of 4** | |
| **True reasons** | 5 of 7 | 3 of 7 (2 real) | 5 of 8 | 3 of 8 |

For comparison, v4 (section 4: the full block of build settings) had 1 of 4 first fix passes,
3 of 4 fixed, 4 of 9 true reasons.

### What happened, per fault

- **gen-011: fixed for the first time, by any model or prompt.** Fixes 1 and 2 still chased the
  compiler and GDAL errors. Fix 3 was `set_python_version 3.9`, with the reason "GDAL ... is not
  compatible with Python 3.13. Setting the Python version to 3.9 aligns with the requirements specified
  in the Pipfile and Pipfile.lock". This is the first reason in any run that names the image's Python
  version as the problem. It came on the last allowed fix; with a budget of 2 it would still be
  "not fixed".
- **gen-010: worse.** Fix 1 copied `add_system_package libffi8` and "a missing shared library" from
  the retrieved libffi8 fault; it failed. In fix 2 the first call proposed `set_python_version 3.9`
  on a 3.9 image (rejected by the validator), and the second pinned `werkzeug 2.3.1`, which passed
  (its reason names Werkzeug, but "Flask 2.0.1" is wrong: the fault pins 2.0.3). With v3 the same model
  pinned werkzeug on the first try.
- **gen-001, gen-013: the same as v3.** gen-013 still copies "3.0.1" and "port 5001" from retrieved
  faults.

### One targeted fact vs a block of build settings (v5 vs v4)

Both add the image's Python version. v4 put it in a block of six build settings (image, apt packages,
install commands, environment, MLflow image, serve command), v5 in one line next to the repo's
declared version.

- **gen-011.** v4 showed `base image: python:3.13-slim` and the model never linked it to the error;
  v5 put `3.13` next to `3.9` and the model got there on the third fix. A comparison placed side
  by side was used; the same fact inside a block was not.
- **Distraction.** Both made gen-010 worse, the same way: a fix copied from a retrieved fault
  (libffi8 in both) and a `set_python_version 3.9` on a 3.9 image. Any Python line seems to invite
  that answer when nothing is wrong with the version. v4 also lost true reasons (4 of 9);
  v5 kept them (5 of 8).
- **Overall.** v5 is the only prompt that fixed all 4 faults, but on the main score (first fix
  passes) it is behind v3 (1 vs 2 of 4), tied with v4.

### Lessons

- **The missing evidence was part of the blind spot.** Once the mismatch was stated, the base model
  found the Python fix for gen-011, which neither more retrieval nor fine-tuning had done. That
  supports section 6: the model couldn't compare versions it wasn't shown.
- **But it's slow and has a cost.** It took the last of 3 fixes, and the same line led the model to
  propose a version change where none was needed (gen-010).
- **A trade-off, not a win.** The main metric was declared before these runs (section 4): first fix
  passes. On it v5 is worse than v3 (1 vs 2 of 4); it is better only on faults fixed (4 vs 3 of 4).
  With one run per fault, a one-fault difference either way is within noise.
- **The line didn't stop a wrong Python change.** On gen-010 the PYTHON line showed matching
  versions (`image python:3.9-slim (Python 3.9); Pipfile python_version 3.9; Pipfile.lock built for
  3.9`), and the model still proposed `set_python_version 3.9`. Only the validator stopped it. The
  model reacts to the presence of a Python line, not only to a mismatch in it. A deterministic Python check outside the LLM (roadmap
  item 9 (e)) would fix gen-011 without the cost on gen-010.

### Fine-tuning data with v5 prompts (roadmap step (b), not trained yet)

The section 5 pipeline again, with only the prompt changed to `diagnose_v5`. 56 SFT examples (25
sampled, 28 rationalized, 3 human, 0 written), 34 after the cap, 37 DPO pairs (v3: 56, 33, 34).
Correct sampled answers fell from 45 to 31 of 72 (gen-008 and gen-009 got worse), and on gen-004 no
sample set the Python version even with 3.12 against 3.9 in the PYTHON line. Rationalization gave
gen-004 its first model reason. Details: `experiments/finetune/report.md`; data in
`experiments/finetune/v5/` (v3 kept in `v3/`).

Logs: `experiments/fixes/gen-NNN-<variant>__diagnose_v5-advanced__qwen2.5-coder-7b/`; copied-term
check: `experiments/finetune/fixloop_copied_terms.json` (base v3, fine-tuned and v5).
Prompt: `prompts/diagnose_v5.md`; the line is `python_text()` in `builder_agent/fix/__init__.py`.

---

## Not run yet

- DPO training (roadmap item 9 step 4); the pairs are in section 5.
- Families 2 (missing dependency) and 6 (system library) have no held-out test case on the taxi repo:
  each has only one variant that breaks, and it is in memory.
- 4 test faults are few: one fault can change a total by 25 points. Read the tables as directions,
  not as measurements.
