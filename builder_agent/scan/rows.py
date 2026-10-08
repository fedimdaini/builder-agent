"""Row overlap between data files, from consistent hash samples (no data is kept).

Every data row (header excluded, line endings stripped) is hashed; a row is in
the sample when its hash falls in one bucket of SAMPLE_MOD. The same row is
picked in every file, so comparing two samples estimates how many rows two
files share and whether one contains the other (train_large.csv contains
validation.csv). Samples are cached outside the target repo, keyed by the
file's path, size and mtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

from .models import DataOverlap

SAMPLE_MOD = 4096
CACHE_FILE = Path(os.environ.get("BUILDER_AGENT_CACHE", Path(__file__).resolve().parents[2] / ".cache")) \
    / "row_samples.json"


def _sample_file(path: Path) -> tuple[int, list[str]]:
    n, keep = 0, set()
    with open(path, "rb") as f:
        f.readline()                                   # header
        for line in f:
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            n += 1
            d = hashlib.blake2b(line, digest_size=8).digest()
            if int.from_bytes(d, "big") % SAMPLE_MOD == 0:
                keep.add(d.hex())
    return n, sorted(keep)


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sample_rows(root: Path, rel_paths: list[str], use_cache: bool = True) -> dict[str, tuple[int, set[str]]]:
    """rel path -> (number of data rows, sampled row hashes)."""
    cache = _load_cache() if use_cache else {}
    out, changed = {}, False
    for rel in rel_paths:
        p = root / rel
        st = p.stat()
        key = f"{p.resolve().as_posix()}|{st.st_size}|{st.st_mtime_ns}|{SAMPLE_MOD}"
        if key not in cache:
            cache[key] = _sample_file(p)
            changed = True
        n, sample = cache[key]
        out[rel] = (n, set(sample))
    if use_cache and changed:
        # drop entries for files that are gone (temp copies, deleted data)
        cache = {k: v for k, v in cache.items() if Path(k.split("|")[0]).exists()}
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass   # the cache is an optimisation only
    return out


def overlaps(samples: dict[str, tuple[int, set[str]]]) -> list[DataOverlap]:
    """Every pair of files whose samples share at least one row."""
    out = []
    for a, b in combinations(sorted(samples), 2):
        sa, sb = samples[a][1], samples[b][1]
        shared = len(sa & sb)
        if shared:
            out.append(DataOverlap(a=a, b=b, shared_in_sample=shared, a_sample=len(sa), b_sample=len(sb),
                                   a_in_b=round(shared / len(sa), 3), b_in_a=round(shared / len(sb), 3)))
    return out
