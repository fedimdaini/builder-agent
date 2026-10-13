"""The blind file of reasons for hand labels (docs/EVAL_PROTOCOL.md, Decisions 3).

    python experiments/eval_loo/make_labels.py --pass main [--n 50] [--out experiments/eval_loo]

Samples n applied LLM fixes from the pass, stratified by config (as equal as n allows), with a fixed seed.
Writes <out>/labels/to_label.csv: item, fault, error_line, reason, label (empty: write true or false).
Nothing else: not the config, not the filter's verdict. Those are in <out>/labels/key.json, which the labeller
doesn't open; analyze.py compares the two once labels are filled in. Refuses to overwrite a sheet that already
has labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import OUT, applied_fixes, load_runs  # noqa: E402

from builder_agent.memory import all_cases  # noqa: E402

SAMPLE_SEED = 2026


def sample(runs: list[dict], n: int, seed: int = SAMPLE_SEED) -> list[dict]:
    cases = {c["id"]: c for c in all_cases()}
    pools: dict[str, list[dict]] = {}
    for r in runs:
        for f in applied_fixes(r, cases):
            pools.setdefault(r["config"], []).append(f | {"config": r["config"], "case": r["case"],
                                                          "variant": r["variant"], "seed": r["seed"]})
    rng = random.Random(seed)
    configs = sorted(pools)
    picked = []
    for i, c in enumerate(configs):
        quota = n // len(configs) + (1 if i < n % len(configs) else 0)
        picked += rng.sample(pools[c], min(quota, len(pools[c])))
    rng.shuffle(picked)                       # configs mixed, so position gives nothing away
    return picked


def write(picked: list[dict], labels_dir: Path) -> None:
    sheet = labels_dir / "to_label.csv"
    if sheet.exists() and any(r["label"].strip() for r in csv.DictReader(sheet.open(encoding="utf-8"))):
        sys.exit(f"{sheet} already has labels: not overwriting it")
    labels_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "fault", "error_line", "reason", "label"])
        for i, x in enumerate(picked, 1):
            item = f"r{i:02d}"
            w.writerow([item, f"{x['case']} ({x['variant']})", x["error_line"], x["reason"], ""])
            key[item] = {"config": x["config"], "case": x["case"], "seed": x["seed"], "fix_n": x["n"],
                         "fix": x["fix"], "filter_true": x["true"], "filter_why": x["why"]}
    (labels_dir / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                         newline="\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    out = Path(args.out)
    picked = sample(load_runs(out, args.pass_), args.n)
    write(picked, out / "labels")
    print(f"{len(picked)} reasons -> {out / 'labels' / 'to_label.csv'} (key: key.json, don't open it before labelling)")


if __name__ == "__main__":
    main()
