# Evaluation protocol: leave-one-out over the 14 faults (ROADMAP B5)

Written 2026-10-12, **before any run**. Nothing below has been run yet. Once approved, the protocol is
frozen: a change after the first run is recorded at the end with its reason, and the runs it affects are
done again.

**Why.** The results in `docs/RESULTS.md` sections 4–9 come from 4 test faults, one run each at
temperature 0. A one-fault difference is 25 points and within noise (section 9). This protocol uses all
14 faults, several samples per fault, and intervals that say how much a difference means.

## 1. Cases

There are 14 cases: the 13 generated faults (`tests/faults/generated/*/case.json`, gen-001 to gen-013)
and fault-001 (`tests/faults/mlflow_client_server_mismatch`, injected as an unpinned MLflow client).

| Case | Variant | Family | Stage | Verified fix | Same-fix sibling in its memory |
|---|---|---|---|---|---|
| fault-001 | mlflow client unpinned (hand-recorded) | version mismatch | train | pin mlflow 2.17.2 | gen-001, gen-007 |
| gen-001 | mlflow client 3.1.4 | version mismatch | train | pin mlflow 2.17.2 | gen-007, gen-008, fault-001 |
| gen-002 | xgboost uninstalled | missing dependency | train | add xgboost | none |
| gen-003 | numpy 2.0.2 | incompatible pin | train | pin numpy 1.23.5 | none |
| gen-004 | Python 3.12 | Python version | build | Python 3.9 | gen-011 |
| gen-005 | tracking URI wrong host | env var | train | MLFLOW_TRACKING_URI | gen-012, gen-013 |
| gen-006 | libffi8 removed | system library | train | add libffi8 | none |
| gen-007 | mlflow client 3.0.1 | version mismatch | train | pin mlflow 2.17.2 | gen-001, gen-008, fault-001 |
| gen-008 | mlflow client unpinned | version mismatch | train | pin mlflow 2.17.2 | gen-001, gen-007 |
| gen-009 | xgboost 3.0.0 | incompatible pin | build | pin xgboost 2.0.0 | none |
| gen-010 | flask 2.0.3 | incompatible pin | health | pin flask (werkzeug also passes) | none |
| gen-011 | Python 3.13 | Python version | build | Python 3.9 | gen-004 |
| gen-012 | tracking URI wrong port | env var | train | MLFLOW_TRACKING_URI | gen-005, gen-013 |
| gen-013 | tracking URI empty | env var | evaluate | MLFLOW_TRACKING_URI | gen-005, gen-012 |

**Leave-one-out memory.**
- Each case is tested with a memory of all the other 13 cases, minus its twin.
- **The one twin pair is gen-008 and fault-001: they are the same fault** (an unpinned client against
  the 2.17.2 server), so each leaves the other out too. That gives 12 memory documents for those two
  cases and 13 for the others.
- The memory is built from the case files only. `memory/learned/` is never used, and nothing is
  written back (evaluation mode).
- Each case's memory snapshot hash is recorded with its runs.

**Change to a standing rule, for your approval.** Until now, split-test faults (gen-001, 010, 011, 013)
were never put into the memory. Leave-one-out replaces the fixed split with a rotating one: every fault
is in the memory for the other cases, and never in the memory when it is being tested. The rule for
training data does not change: the fine-tuned models were trained only on the 9 split-memory faults
(section 6 below).

**Siblings stay in the memory, by design.** 9 of the 14 cases have another fault with the same fix in
their memory: the MLflow client group, the tracking-URI group and the Python pair. That is what
retrieval is for, but it also lets a model copy a neighbour's answer, as on gen-013 in section 9.
So every metric is also reported for the **5 cases with no same-fix sibling** (gen-002, 003, 006, 009,
010). There, a passing fix can't be copied from memory.

## 2. Configurations

All three use the base model `qwen2.5-coder:7b` (its digest is recorded) and the same 14 cases and
seeds:

| Config | Prompt | Retrieval |
|---|---|---|
| v1 | `diagnose_v1` (zero-shot) | none: memory plays no part |
| v3-advanced | `diagnose_v3` | advanced retriever over the leave-one-out memory |
| v5-advanced | `diagnose_v5` (v3 plus the PYTHON line) | advanced retriever over the leave-one-out memory |

Shared settings:
- **`--no-python-rule`**: LLM-only, so gen-004 and gen-011 measure the model, not the rule (RESULTS
  section 10).
- At most 3 fixes per run. At most 3 calls per diagnosis, where a call rejected by the validator is
  retried with the reasons. The validator and the regression guard are those of commit `59162ef`.
- **Temperature 0.7, seeds 101–105**: five samples per case and config. Seeds 1–8 were used to sample
  the fine-tuning data, so they are avoided here.
- `num_ctx` 8192, the Ollama defaults otherwise. The same seed with different prompts doesn't give
  related samples, so seeds fix reproducibility, not pairing.

In total: 14 cases × 5 seeds × 3 configs = **210 fix-loop runs**.

## 3. Metrics

One run is one fix loop on one case, with one config and one seed.

| Metric | Unit | Definition |
|---|---|---|
| **First fix passes** (main metric, declared in section 4 of RESULTS.md) | run | the first applied fix passes every sandbox stage |
| Fixed within 3 | run | some fix passes within the 3 allowed |
| True reasons | fix | the reason passes the faithfulness filter (below) |
| Copied reasons | fix | the reason names a term only a retrieved other fault has: the rule of RESULTS.md section 6 (`finetune.foreign_terms`), applied to every applied fix |

**True reasons, decided by a frozen rule and checked by hand.** 210 runs give several hundred reasons,
too many to judge by hand as in sections 4–9. The faithfulness filter of `experiments/finetune/report.md`
decides it automatically. A reason is true if it:
- names a term of the fault's own error line;
- names no term only another fault has;
- claims no Python version that is neither the image's nor the fix's.

The filter's code is frozen before the first run (git hash recorded). A random sample of **50 reasons**
(stratified by config, with the config hidden) is then labelled by hand, by you rather than by me, since
I wrote the filter. The agreement between filter and hand labels is reported. If it is below 80%, true
reasons are reported as unreliable, and the hand labels are used for the sampled reasons only.

**Rates and intervals.**
- Each metric is a success rate per config: per run for the first two, per fix for the others.
- The 5 samples of a case are not independent. The 95% interval is therefore a **case-cluster
  bootstrap**: 14 cases resampled with replacement, all of a case's runs kept together, 10,000
  resamples, percentile interval, fixed bootstrap seed.
- The Wilson interval over runs is shown next to it for reference only, because it understates the
  uncertainty.

**Paired comparison.** Configurations are compared **on the same cases**. The unit is the case: its
mean over the 5 samples.

| Comparison | |
|---|---|
| Effect | the difference of the two configs' rates, with a 95% paired case-cluster bootstrap interval (cases resampled once and applied to both configs) |
| Test | exact two-sided sign test on the 14 per-case differences (ties dropped), with Holm's correction over the 3 pairs |
| Decision rule (declared now) | a config is called better on a metric only if the paired 95% interval of the difference excludes 0 and the Holm-corrected sign test gives p < 0.05. Otherwise: "no difference shown" |
| Secondary | McNemar on the 70 (case, seed) pairs, reported, but not used to decide (it ignores the clustering) |

Results are also reported per family and for the 5 no-sibling cases, as descriptions, without tests:
5 cases are too few.

## 4. What is recorded

For each run:
- the case, config, seed, model digest and prompt sha256;
- the memory snapshot hash and the retrieved documents;
- every LLM call (the fix log) and every sandbox attempt (the sandbox log);
- the score (first fix passed, fixes to pass);
- per applied fix: true (filter verdict) and copied (terms).

Everything goes under `experiments/eval_loo/<config>/<case>/seed-<n>/`, with one `runs.jsonl` and the
analysis output (`summary.json`, `summary.md`).

## 5. What has to be built before running (nothing exists yet)

1. **Fix-loop options for sampling:** `--temperature`, `--seed`, passed to the Ollama client and recorded.
2. **Leave-one-out memory:** `--memory loo --exclude <case>` builds the memory from the 14 case files minus
   the case and its twin, in evaluation mode. It includes the split-test faults, which is the rule
   change above.
3. **A batch runner** (`experiments/eval_loo/run.py`): resumable, skips finished runs, runs the configs in
   a shuffled but fixed order, and stops on a Docker or Ollama failure instead of recording it as a
   model failure (the Docker-down runs of section 9 were caught by hand).
4. **Caching the initial run (optional).** The first sandbox run of a case is the same in every config
   and seed: the injected fault, before any fix. It can run once per case (14 runs instead of 210),
   with a check that its failed stage and error line match `case.json`. This saves about a third of
   the sandbox time.
5. **The analysis script:** filter verdicts, copied terms, bootstrap, sign tests, tables.

Each comes with tests, and the regression checks run before the first evaluation run.

## 6. The fine-tuned models

`qwen-builder-sft` (v3 data) and `qwen-builder-sft-v5` were trained on examples from the 9 split-memory
faults. Leave-one-out on those 9 would test them on their own training faults, so **only the 4 original
test faults apply to them**: gen-001, 010, 011, 013. Fault-001 is the same fault as gen-008, a
training fault, so it is excluded too.

| Setting | |
|---|---|
| Configs | `qwen-builder-sft` with v3-advanced and `qwen-builder-sft-v5` with v5-advanced, each on its training prompt |
| Runs | 4 cases × the same 5 seeds, with the same leave-one-out memory as the base runs: 40 runs |
| Comparison | paired with the base model on the same prompt over those 4 cases (from the main runs) |
| How it is reported | a separate table, as descriptions only. With 4 cases the sign test can't reach p < 0.05 (the smallest two-sided p is 0.125), and the bootstrap over 4 clusters is coarse. It can show a direction, not a difference |

## 7. Run time

From the 32 fix-loop runs logged in `experiments/fixes/`:

| Measure | Value |
|---|---|
| Sandbox run, all stages ok | median 28 s |
| Sandbox run that fails at train | median 184 s (it trains first) |
| Sandbox run, average | 53 s |
| LLM call | 5.5 s on average |
| Fix-loop run, overall | mean 173 s (median 121 s), with 3.0 sandbox runs per loop on average |

| Part | Runs | Estimate |
|---|---|---|
| Main runs, no caching | 210 | 210 × 173 s ≈ **10 h** |
| Main runs, with the initial run cached | 210 | about **7 h** |
| Main runs, worst case (every run uses 3 fixes and 3 calls per diagnosis, 4 sandbox runs) | 210 | about **15 h** |
| Fine-tuned models | 40 | about **2 h** |
| Hand labels (you) | 50 reasons | about 30–45 min |

Temperature 0.7 likely means more wrong fixes than at temperature 0, so more fixes per run: expect
nearer the upper estimate than the lower one. Plan **three overnight sessions**, one per configuration,
plus one shorter session for the fine-tuned models. The runner resumes, so a power cut costs only the
run in progress.

### Measured 2026-10-13 (no change to the protocol)

**Why some builds are slow.** The Dockerfile installs everything in one `RUN` layer: the repo's pipenv
install, any injected command and the Builder's extras. A fault or a fix that changes that line, or a
layer before it (an `ENV`, apt packages, the Python image), re-runs the whole dependency install, about
230 s with no pip cache. An identical Dockerfile seen before builds in 1–8 s. The build cache wasn't
pruned (38.7 GB); the smoke test's slow builds were the first builds of gen-002's fault and fix lines.

**Speed-ups in the runner, without changing the protocol.**
- The runs are grouped by case.
- The first sandbox run of each case is shared (section 5 item 4). It is stored with a fingerprint of
  its inputs and reused across configs, seeds and both passes.

**Measured on gen-003** (2 seeds × 3 configs, `experiments/eval_loo/_measure/`):

| Step | Time |
|---|---|
| Shared first run | 192 s, once |
| 2 runs whose fix line was new | 251 s and 263 s (a cold build of about 230 s each) |
| 4 runs that reused those builds | 25–38 s |
| LLM calls | about 6 s each |

**New estimate.** The cost is driven by how many different fix Dockerfiles each case produces, each
built once. At temperature 0.7, assume 4–8 per case.

| Pass | Parts | Total |
|---|---|---|
| Main (210 runs) | first runs 14 × about 200 s ≈ 0.8 h; cold fix builds 56–112 × 230 s ≈ 3.6–7.2 h; per-run work (about 1.8 fixes × (25 s sandbox + 9 s LLM)) ≈ 3.5 h | **about 8–12 h** |
| Temperature 0 (42 runs) | first runs reused; most fix builds already cached from the main pass | **about 1–2 h** |

A larger saving would need a Builder change: separate layers for the repo's install and the Builder's
extras, or a pip cache mount. That changes the generated Dockerfile, so it isn't done here.

## 8. How results will be reported

A new RESULTS.md section with:
- per config: the four rates with their bootstrap and Wilson intervals;
- the three paired comparisons, each with its effect, interval, sign test and verdict under the rule above;
- the no-sibling subgroup and the per-family rates, as descriptions;
- the fine-tuned table.

The sections 4–9 single runs stay as they are. They used temperature 0 and a different memory, so they
aren't pooled with these runs.

## Decisions (2026-10-12, approved before any run)

1. **Test faults in the memory.** The split-test faults (gen-001, 010, 011, 013) may go into the memory
   for the other cases, never while they are being tested. This replaces the earlier rule that kept
   them out of the memory. The rule for training data is unchanged.
2. **No-sibling subgroup.** Every metric is also reported for the 5 cases with no same-fix fault in
   their memory (gen-002, 003, 006, 009, 010).
3. **Hand labels by Jack, blind.** Jack labels the 50 sampled reasons himself. He gets a file with, per
   reason, the fault, its error line and the reason, and nothing else: not the filter's verdict, not
   the config.
4. **An extra pass at temperature 0.** 14 cases × 3 configurations, one run each (42 runs, about 2 h),
   with the same leave-one-out memory and `--no-python-rule`. It is reported next to the main results
   and not pooled with them.

## Changes after approval

(none yet)
