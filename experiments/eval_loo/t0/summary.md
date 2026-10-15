# Leave-one-out results: pass t0 (42 runs)

Faithfulness filter frozen at commit c8cb5c8. Interval: case-cluster bootstrap; Wilson in brackets, for reference only.

| Metric | v1 | v3-advanced | v5-advanced |
|---|---|---|---|
| First fix passes (main) | 43% (6/14); 95% 21%–71% [Wilson 21%–67%] | 64% (9/14); 95% 43%–86% [Wilson 39%–84%] | 64% (9/14); 95% 43%–86% [Wilson 39%–84%] |
| Fixed within 3 | 71% (10/14); 95% 50%–93% [Wilson 45%–88%] | 93% (13/14); 95% 79%–100% [Wilson 69%–99%] | 93% (13/14); 95% 79%–100% [Wilson 69%–99%] |
| True reasons | 35% (9/26); 95% 18%–60% [Wilson 19%–54%] | 39% (7/18); 95% 16%–69% [Wilson 20%–61%] | 42% (8/19); 95% 21%–69% [Wilson 23%–64%] |
| Copied reasons | 0% (0/26); 95% 0%–0% [Wilson 0%–13%] | 33% (6/18); 95% 12%–56% [Wilson 16%–56%] | 21% (4/19); 95% 0%–47% [Wilson 9%–43%] |
| Action accuracy (secondary) | 36% (5/14); 95% 14%–64% [Wilson 16%–61%] | 57% (8/14); 95% 29%–86% [Wilson 33%–79%] | 57% (8/14); 95% 29%–86% [Wilson 33%–79%] |

Action accuracy, first LLM action (exact / same target / verified alternative / wrong / none): v1 2/3/0/9/0; v3-advanced 5/1/2/6/0; v5-advanced 6/1/1/6/0. Secondary: not used to call a config better.

## Paired comparisons (later config minus earlier one, per case)

| Metric | Pair | Difference, 95% interval | Sign test (better/worse, p, Holm p) | Verdict |
|---|---|---|---|---|
| First fix passes (main) | v3-advanced - v1 | +21% (+0% to +43%) | 3/0, 0.250, 0.750 | no difference shown |
| First fix passes (main) | v5-advanced - v1 | +21% (+0% to +43%) | 3/0, 0.250, 0.750 | no difference shown |
| First fix passes (main) | v5-advanced - v3-advanced | +0% (+0% to +0%) | 0/0, 1.000, 1.000 | no difference shown |
| Fixed within 3 | v3-advanced - v1 | +21% (+0% to +43%) | 3/0, 0.250, 0.750 | no difference shown |
| Fixed within 3 | v5-advanced - v1 | +21% (+0% to +43%) | 3/0, 0.250, 0.750 | no difference shown |
| Fixed within 3 | v5-advanced - v3-advanced | +0% (+0% to +0%) | 0/0, 1.000, 1.000 | no difference shown |
| True reasons | v3-advanced - v1 | +4% (-24% to +35%) | 3/4, 1.000, 1.000 | no difference shown |
| True reasons | v5-advanced - v1 | +7% (-20% to +36%) | 3/3, 1.000, 1.000 | no difference shown |
| True reasons | v5-advanced - v3-advanced | +3% (+0% to +10%) | 1/0, 1.000, 1.000 | no difference shown |
| Copied reasons | v3-advanced - v1 | +33% (+12% to +56%) | 5/0, 0.062, 0.188 | no difference shown |
| Copied reasons | v5-advanced - v1 | +21% (+0% to +47%) | 3/0, 0.250, 0.500 | no difference shown |
| Copied reasons | v5-advanced - v3-advanced | -12% (-25% to +0%) | 0/2, 0.500, 0.500 | no difference shown |

## Cases with no same-fix case in memory (gen-002, gen-003, gen-006, gen-009, gen-010; descriptive)

| Metric | v1 | v3-advanced | v5-advanced |
|---|---|---|---|
| First fix passes (main) | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] |
| Fixed within 3 | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] |
| True reasons | 100% (5/5); 95% 100%–100% [Wilson 57%–100%] | 80% (4/5); 95% 40%–100% [Wilson 38%–96%] | 80% (4/5); 95% 40%–100% [Wilson 38%–96%] |
| Copied reasons | 0% (0/5); 95% 0%–0% [Wilson 0%–43%] | 20% (1/5); 95% 0%–60% [Wilson 4%–62%] | 20% (1/5); 95% 0%–60% [Wilson 4%–62%] |
| Action accuracy (secondary) | 80% (4/5); 95% 40%–100% [Wilson 38%–96%] | 80% (4/5); 95% 40%–100% [Wilson 38%–96%] | 80% (4/5); 95% 40%–100% [Wilson 38%–96%] |

## First fix passes per family (descriptive)

- v1: env_var 0/3; incompatible_pin 3/3; missing_dependency 1/1; python_version 1/2; system_library 1/1; version_mismatch 0/4
- v3-advanced: env_var 1/3; incompatible_pin 3/3; missing_dependency 1/1; python_version 1/2; system_library 1/1; version_mismatch 2/4
- v5-advanced: env_var 1/3; incompatible_pin 3/3; missing_dependency 1/1; python_version 1/2; system_library 1/1; version_mismatch 2/4
