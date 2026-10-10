# Fine-tuning data (roadmap item 9 step 3)

Source: the 9 generated faults with split **memory** (`tests/faults/generated/`). The 4 test faults are
never rendered, sampled or written; `tests/test_finetune.py` fails if one appears in `sft.jsonl` or
`dpo.jsonl` (by id, variant, or any line of its error output that no memory fault shares).

**How.** For each fault, the `diagnose_v3` prompt is rendered as the fix loop renders its first
attempt, with the advanced retriever over a memory that leaves the fault itself out (and fault-001 for
gen-008, the same fault), so the model can't copy the answer. 8 answers per fault from
qwen2.5-coder:7b at temperature 0.7 (seeds 1–8), with the fix loop's JSON schema.

**Labels.**

- *correct*: the same fix as the fault's verified fix (*exact*), the same package with another
  version (*loose*, pin_package only), or a different fix that passed the sandbox on top of the fault
  (*sandbox*);
- *invalid*: rejected by the fix loop's validator;
- *wrong*: everything else.

Possible alternatives were checked in the sandbox: 9 distinct fixes, 3 passed (`libffi-dev` for
libffi8, and `set_python_version 3.10` for both python_3_12 and xgboost_3_0_0).

**Faithful reasons (SFT and DPO "chosen").** A correct answer is kept only if its reason:

- names a term of the fault's own error line;
- names no term that only a retrieved other fault has;
- doesn't claim a Python version that is neither the image's nor the fix's.

## Per family

| Family | Faults | Samples | Correct | Dropped (unfaithful) | Wrong | Invalid | SFT examples | DPO pairs |
|---|---|---|---|---|---|---|---|---|
| version mismatch | 2 | 16 | 8 | 7 | 8 | 0 | 1 | 0 |
| missing dependency | 1 | 8 | 8 | 0 | 0 | 0 | 8 | 0 |
| incompatible pin | 2 | 16 | 12 | 3 | 0 | 4 | 9 | 4 |
| Python version | 1 | 8 | 1 | 1 | 7 | 0 | 0 | 0 |
| env var | 2 | 16 | 8 | 6 | 0 | 8 | 2 | 0 |
| system library | 1 | 8 | 8 | 0 | 0 | 0 | 8 | 0 |
| **Total** | **9** | **72** | **45** | **17** | **15** | **12** | **28** | **4** |

## Per fault

| Fault | Labels | Dropped | SFT | DPO |
|---|---|---|---|---|
| gen-002 xgboost uninstalled | 8 correct (exact) | 0 | 8 | 0 |
| gen-003 numpy 2.0.2 | 8 correct (loose: numpy 1.21.x / 1.23.1, not 1.23.5) | 0 | 8 | 0 |
| gen-004 Python 3.12 | 7 wrong (pipenv pins), 1 correct (sandbox: Python 3.10) | 1 | 0 | 0 |
| gen-005 wrong host | 8 invalid (repeated the faulty URI, already applied) | 0 | 0 | 0 |
| gen-006 libffi8 removed | 6 exact + 2 sandbox (`libffi-dev`) | 0 | 8 | 0 |
| gen-007 MLflow client 3.0.1 | 8 wrong (`GIT_PYTHON_REFRESH`, failed in the sandbox) | 0 | 0 | 0 |
| gen-008 MLflow client unpinned | 8 correct (exact) | 7 | 1 | 0 |
| gen-009 xgboost 3.0.0 | 4 correct (sandbox: Python 3.10), 4 invalid | 3 | 1 | 4 |
| gen-012 wrong port | 8 correct (exact) | 6 | 2 | 0 |

## What it shows

- **Copied reasons are common.** 17 of 45 correct fixes (38%) were dropped. In gen-008, 7 of 8
  reasons say "the image installs mlflow==3.0.1", copied from the retrieved gen-007; the image
  installs mlflow unpinned. In gen-012, 6 of 8 say "the URI points at mlflow-server", copied from
  gen-005; the real fault is the wrong port. In gen-009, 3 say the image is Python 3.12; it is 3.9.
- **DPO pairs need a correct and a wrong answer to the same prompt.** Only gen-009 had both, so there
  are 4 pairs, all the same fix against two different wrong fixes. The model is consistent at
  temperature 0.7: per fault it is mostly all right or all wrong.
- **Limits of the filter.** It is a term check. It catches copied names and numbers, but it passes
  vague or muddled reasons that name a right term. Two kept gen-012 reasons say "the service is named
  'mlflow'" without saying what is wrong.
- **Loose matches, now sandbox-verified.** The 8 gen-003 examples pin numpy 1.21.x or 1.23.1 instead
  of the locked 1.23.5. All 4 distinct versions (1.21.2, 1.21.4, 1.21.5, 1.23.1) passed every
  sandbox stage on top of the fault (`checks.json`, kind "loose"), so they stay correct. A loose pin
  that failed would be labelled wrong.

## Filter changes after the first build

These made no difference to the sampled answers (same counts as above). They were needed for
written reasons, which quote the fault's own record:

- The fault's own recorded cause doesn't count as copied. Example: the server image `v2.17.2`
  in gen-007 and gen-008.
- The Python check flags only claims about the image ("the image is Python 3.12"), not
  requirements ("xgboost 3.x requires Python 3.10").

## Rationalization (STaR) and the final datasets

**Rationalization.** For each fault, the same prompt plus a hint: the verified fix, and a request for
a one- or two-sentence reason from the current error output. 4 answers per fault at temperature 0.7
(36 calls, no errors). A reason is kept if:

- the fix is unchanged;
- the reason doesn't mention the hint ("verified", "hint"): the training prompt has no hint;
- the reason passes the faithfulness filter.

If none is kept, a reason is written from the recorded error line and cause. Training records always
use the prompt **without** the hint.

| Fault | Hinted reasons kept (of 4) | Why the others were not kept |
|---|---|---|
| gen-002 xgboost uninstalled | 4 | |
| gen-003 numpy 2.0.2 | 4 | |
| gen-004 Python 3.12 | **0, so written** | 2 mention the hint ("as verified by the past failure"), 2 name no term of the error line |
| gen-005 wrong host | 4 | |
| gen-006 libffi8 removed | 4 | |
| gen-007 MLflow client 3.0.1 | 3 | 1 mentions the hint ("verified to work") |
| gen-008 MLflow client unpinned | 1 | 3 name "3.0.1", copied from the retrieved gen-007 |
| gen-009 xgboost 3.0.0 | 3 | 1 mentions the hint ("the verified fix is") |
| gen-012 wrong port | 2 | 2 name "server", copied from the retrieved gen-005 |

The hint check was added after this run: 4 kept reasons quoted the hint. `build` judges the stored
hinted answers again, so the current filter applies (`rationalized_final.jsonl`).

**SFT** keeps every distinct correct and faithful answer, from all sources.

**DPO.** *rejected* = every distinct sampled answer that is wrong, invalid, or correct with an
unfaithful reason (`rejected_kind`). *chosen* = a faithful correct answer of the best source available
(sampled, then rationalized, then written). Identical rejected answers count once: for example, gen-012's
6 copied reasons are 1 distinct answer.

| Family | SFT examples | sampled | rationalized | written | DPO pairs | rejected wrong | invalid | unfaithful |
|---|---|---|---|---|---|---|---|---|
| version mismatch | 5 | 1 | 4 | 0 | 12 | 8 | 0 | 4 |
| missing dependency | 11 | 8 | 3 | 0 | 0 | 0 | 0 | 0 |
| incompatible pin | 16 | 9 | 7 | 0 | 7 | 0 | 4 | 3 |
| Python version | 1 | 0 | 0 | 1 | 8 | 7 | 0 | 1 |
| env var | 8 | 2 | 6 | 0 | 7 | 0 | 6 | 1 |
| system library | 12 | 8 | 4 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **53** | **28** | **24** | **1** | **34** | **15** | **10** | **9** |

Every memory fault now has at least one SFT example; a test checks it. Before rationalization,
gen-004, gen-005 and gen-007 had none. DPO went from 4 to 34 pairs. Their "chosen" answer comes from
the model for 12 pairs (gen-008, gen-009, gen-012), from rationalization for 14 (gen-005, gen-007)
and from the written reason for 8 (gen-004).

**Only written:** gen-004 (python_3_12). Even when told the fix was `set_python_version 3.9`, the
model couldn't say why from its error output (the `pkgutil.ImpImporter` error of old setuptools on
Python 3.12). This is the same blind spot as the held-out gen-011 (Python 3.13).

**Not balanced.** Missing dependency and system library have 11–12 SFT examples but no DPO pairs:
the model was always right there. Python version has 1 SFT example (written) and 8 pairs. Version
mismatch and env var lean on rationalized answers.

## Files

- `prompts.json`: the rendered prompt per fault, what was retrieved and excluded.
- `samples.jsonl`: every model call (seed, options, model digest, latency, response).
- `labels.jsonl` (before the sandbox checks), `labels_final.jsonl` (after the checks, with the
  faithfulness verdict and its reason).
- `checks.json`: the sandbox checks of possible alternatives and loose pins (attempt ids in
  `logs/sandbox_attempts.jsonl`).
- `rationalized.jsonl`: the hinted samples as run (calls logged as in sampling);
  `rationalized_final.jsonl`: the same, judged with the current filter (including the hint check).
- `sft.jsonl`: `messages` = system, user, assistant (the answer as JSON), with case id, variant,
  family, seed and match type.
- `dpo.jsonl`: `prompt` (system + user messages), `chosen`, `rejected`.
- `stats.json`: the per-family counts above.

Rebuild: `python -m builder_agent.finetune sample|check|build` (see `builder_agent/finetune/__main__.py`).
