# Leave-one-out results: pass main (2 runs)

Faithfulness filter frozen at commit c8cb5c8. Interval: case-cluster bootstrap; Wilson in brackets, for reference only.

| Metric | v3-advanced |
|---|---|
| First fix passes (main) | 100% (2/2); 95% 100%–100% [Wilson 34%–100%] |
| Fixed within 3 | 100% (2/2); 95% 100%–100% [Wilson 34%–100%] |
| True reasons | 50% (1/2); 95% 0%–100% [Wilson 9%–91%] |
| Copied reasons | 50% (1/2); 95% 0%–100% [Wilson 9%–91%] |

## Paired comparisons (later config minus earlier one, per case)

| Metric | Pair | Difference, 95% interval | Sign test (better/worse, p, Holm p) | Verdict |
|---|---|---|---|---|

## Cases with no same-fix case in memory (gen-002, gen-003, gen-006, gen-009, gen-010; descriptive)

| Metric | v3-advanced |
|---|---|
| First fix passes (main) | 100% (1/1); 95% 100%–100% [Wilson 21%–100%] |
| Fixed within 3 | 100% (1/1); 95% 100%–100% [Wilson 21%–100%] |
| True reasons | 100% (1/1); 95% 100%–100% [Wilson 21%–100%] |
| Copied reasons | 0% (0/1); 95% 0%–0% [Wilson 0%–79%] |

## First fix passes per family (descriptive)

- v3-advanced: env_var 1/1; missing_dependency 1/1
