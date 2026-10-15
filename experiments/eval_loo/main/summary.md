# Leave-one-out results: pass main (210 runs)

Faithfulness filter frozen at commit c8cb5c8. Interval: case-cluster bootstrap; Wilson in brackets, for reference only.

| Metric | v1 | v3-advanced | v5-advanced |
|---|---|---|---|
| First fix passes (main) | 36% (25/70); 95% 16%–57% [Wilson 26%–47%] | 57% (40/70); 95% 37%–77% [Wilson 45%–68%] | 57% (40/70); 95% 37%–77% [Wilson 45%–68%] |
| Fixed within 3 | 61% (43/70); 95% 40%–81% [Wilson 50%–72%] | 86% (60/70); 95% 73%–96% [Wilson 76%–92%] | 81% (57/70); 95% 64%–96% [Wilson 71%–89%] |
| True reasons | 39% (48/122); 95% 27%–56% [Wilson 31%–48%] | 53% (52/98); 95% 39%–71% [Wilson 43%–63%] | 55% (51/93); 95% 41%–73% [Wilson 45%–65%] |
| Copied reasons | 0% (0/122); 95% 0%–0% [Wilson 0%–3%] | 13% (13/98); 95% 3%–27% [Wilson 8%–21%] | 12% (11/93); 95% 2%–26% [Wilson 7%–20%] |
| Action accuracy (secondary) | 29% (20/70); 95% 9%–50% [Wilson 19%–40%] | 50% (35/70); 95% 27%–73% [Wilson 39%–61%] | 51% (36/70); 95% 30%–73% [Wilson 40%–63%] |

Action accuracy, first LLM action (exact / same target / verified alternative / wrong / none): v1 8/12/0/48/2; v3-advanced 25/6/4/35/0; v5-advanced 28/5/3/33/1. Secondary: not used to call a config better.

## Paired comparisons (later config minus earlier one, per case)

| Metric | Pair | Difference, 95% interval | Sign test (better/worse, p, Holm p) | Verdict |
|---|---|---|---|---|
| First fix passes (main) | v3-advanced - v1 | +21% (+1% to +44%) | 5/2, 0.453, 0.906 | no difference shown |
| First fix passes (main) | v5-advanced - v1 | +21% (-1% to +46%) | 6/2, 0.289, 0.867 | no difference shown |
| First fix passes (main) | v5-advanced - v3-advanced | +0% (-7% to +6%) | 2/1, 1.000, 1.000 | no difference shown |
| Fixed within 3 | v3-advanced - v1 | +24% (+4% to +46%) | 8/2, 0.109, 0.328 | no difference shown |
| Fixed within 3 | v5-advanced - v1 | +20% (+1% to +40%) | 8/2, 0.109, 0.328 | no difference shown |
| Fixed within 3 | v5-advanced - v3-advanced | -4% (-13% to +4%) | 2/3, 1.000, 1.000 | no difference shown |
| True reasons | v3-advanced - v1 | +14% (-2% to +33%) | 8/4, 0.388, 0.775 | no difference shown |
| True reasons | v5-advanced - v1 | +15% (+0% to +34%) | 8/2, 0.109, 0.328 | no difference shown |
| True reasons | v5-advanced - v3-advanced | +2% (-5% to +8%) | 6/3, 0.508, 0.775 | no difference shown |
| Copied reasons | v3-advanced - v1 | +13% (+3% to +27%) | 5/0, 0.062, 0.188 | no difference shown |
| Copied reasons | v5-advanced - v1 | +12% (+2% to +26%) | 4/0, 0.125, 0.250 | no difference shown |
| Copied reasons | v5-advanced - v3-advanced | -1% (-5% to +3%) | 3/2, 1.000, 1.000 | no difference shown |

## Cases with no same-fix case in memory (gen-002, gen-003, gen-006, gen-009, gen-010; descriptive)

| Metric | v1 | v3-advanced | v5-advanced |
|---|---|---|---|
| First fix passes (main) | 80% (20/25); 95% 56%–100% [Wilson 61%–91%] | 80% (20/25); 95% 64%–96% [Wilson 61%–91%] | 80% (20/25); 95% 60%–96% [Wilson 61%–91%] |
| Fixed within 3 | 100% (25/25); 95% 100%–100% [Wilson 87%–100%] | 88% (22/25); 95% 72%–100% [Wilson 70%–96%] | 84% (21/25); 95% 60%–100% [Wilson 65%–94%] |
| True reasons | 83% (25/30); 95% 69%–100% [Wilson 66%–93%] | 75% (21/28); 95% 57%–96% [Wilson 57%–87%] | 83% (20/24); 95% 64%–100% [Wilson 64%–93%] |
| Copied reasons | 0% (0/30); 95% 0%–0% [Wilson 0%–11%] | 7% (2/28); 95% 0%–19% [Wilson 2%–23%] | 8% (2/24); 95% 0%–23% [Wilson 2%–26%] |
| Action accuracy (secondary) | 68% (17/25); 95% 28%–100% [Wilson 48%–83%] | 68% (17/25); 95% 32%–96% [Wilson 48%–83%] | 64% (16/25); 95% 28%–96% [Wilson 45%–80%] |

## First fix passes per family (descriptive)

- v1: env_var 0/15; incompatible_pin 13/15; missing_dependency 5/5; python_version 4/10; system_library 2/5; version_mismatch 1/20
- v3-advanced: env_var 5/15; incompatible_pin 12/15; missing_dependency 5/5; python_version 3/10; system_library 3/5; version_mismatch 12/20
- v5-advanced: env_var 5/15; incompatible_pin 11/15; missing_dependency 5/5; python_version 3/10; system_library 4/5; version_mismatch 12/20
