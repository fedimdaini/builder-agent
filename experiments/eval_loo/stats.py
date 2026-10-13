"""Statistics of docs/EVAL_PROTOCOL.md section 3. Pure functions, no I/O, standard library only.

Data shape: per config, {case_id: [0/1 outcomes]} (one outcome per run, or per applied fix).
"""

from __future__ import annotations

import math
import random
from math import comb

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 2026


def rate(per_case: dict[str, list[int]]) -> tuple[int, int, float | None]:
    k = sum(sum(v) for v in per_case.values())
    n = sum(len(v) for v in per_case.values())
    return k, n, (k / n if n else None)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson score interval over runs (for reference: it ignores the clustering by case)."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    i = (len(values) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def cluster_bootstrap(per_case: dict[str, list[int]], reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """95% percentile interval of the pooled rate, resampling cases (all of a case's outcomes together)."""
    cases = [c for c, v in per_case.items() if v]
    if not cases:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        pick = [rng.choice(cases) for _ in cases]
        n = sum(len(per_case[c]) for c in pick)
        stats.append(sum(sum(per_case[c]) for c in pick) / n)
    return _percentile(stats, 0.025), _percentile(stats, 0.975)


def paired_bootstrap(a: dict[str, list[int]], b: dict[str, list[int]], reps: int = BOOTSTRAP_REPS,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float] | None:
    """Difference of pooled rates a - b over the cases both have, with a 95% paired case-cluster interval
    (the same resampled cases for both configs). Returns (difference, low, high)."""
    cases = [c for c in a if a[c] and b.get(c)]
    if not cases:
        return None

    def diff(pick):
        ra = sum(sum(a[c]) for c in pick) / sum(len(a[c]) for c in pick)
        rb = sum(sum(b[c]) for c in pick) / sum(len(b[c]) for c in pick)
        return ra - rb
    rng = random.Random(seed)
    stats = [diff([rng.choice(cases) for _ in cases]) for _ in range(reps)]
    return diff(cases), _percentile(stats, 0.025), _percentile(stats, 0.975)


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    """Exact two-sided sign test on per-case differences (ties dropped). Returns (positive, negative, p)."""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n
    return pos, neg, min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, kept monotone)."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def mcnemar(pairs: list[tuple[int, int]]) -> tuple[int, int, float]:
    """Exact McNemar on paired 0/1 outcomes (a, b). Returns (a only, b only, two-sided p)."""
    b_ = sum(1 for x, y in pairs if x and not y)
    c_ = sum(1 for x, y in pairs if y and not x)
    n = b_ + c_
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(b_, c_) + 1)) / 2 ** n
    return b_, c_, min(1.0, 2 * tail)


def case_means(per_case: dict[str, list[int]]) -> dict[str, float]:
    return {c: sum(v) / len(v) for c, v in per_case.items() if v}
