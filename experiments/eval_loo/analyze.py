"""Analysis of the leave-one-out runs (docs/EVAL_PROTOCOL.md section 3).

    python experiments/eval_loo/analyze.py --pass main [--out experiments/eval_loo]

Reads <out>/<pass>/*/*/seed-*/result.json, writes <out>/<pass>/summary.json and summary.md. Reads only.

Per run: first fix passes, fixed within 3. Per applied LLM fix (an answer that was applied and checked in the
sandbox): true reason (the faithfulness filter of builder_agent.finetune, frozen: its git hash is recorded) and
copied reason (a term only a retrieved other case has, the rule of RESULTS.md section 6). The error a fix answers
is the case's recorded error line for the first fix, and that attempt's own error output for later ones.
Secondary, per run (protocol "Changes after approval" 1): action accuracy of the first LLM action: exact, same
target, or a sandbox-verified alternative from experiments/finetune/checks.json. Not used in the comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from builder_agent.finetune import _terms, faithful, foreign_terms  # noqa: E402
from builder_agent.scan.deps import normalize  # noqa: E402
from builder_agent.memory import all_cases, loo_docs  # noqa: E402
import stats  # noqa: E402

OUT = ROOT / "experiments" / "eval_loo"
NO_SIBLING = ["gen-002", "gen-003", "gen-006", "gen-009", "gen-010"]   # no same-fix case in their memory
CONFIG_ORDER = ["v1", "v3-advanced", "v5-advanced"]
RUN_METRICS = {"first_fix_passed": "First fix passes (main)", "fixed": "Fixed within 3"}
FIX_METRICS = {"true": "True reasons", "copied": "Copied reasons"}
SECONDARY = {"action": "Action accuracy (secondary)"}
CHECKS = ROOT / "experiments" / "finetune" / "checks.json"
NAMED = {"pin_package", "add_dependency", "add_system_package", "set_env_var"}   # actions with a target


def verified_alternatives(path: Path = CHECKS) -> dict[str, list[dict]]:
    """case id -> fixes that passed the sandbox on top of the fault (checks.json, ok true)."""
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for c in json.loads(path.read_text(encoding="utf-8")).values():
        if c.get("ok"):
            out.setdefault(c["case_id"], []).append(c["fix"])
    return out


def _same(a: dict, b: dict) -> bool:
    return {k: (normalize(str(v)) if k == "name" else v) for k, v in a.items()} == \
        {k: (normalize(str(v)) if k == "name" else v) for k, v in b.items()}


def action_match(run: dict, case: dict, alternatives: dict[str, list[dict]]) -> str:
    """The first LLM action that was applied, against the case's expected fix: "exact", "same_target",
    "alternative", "wrong" (any other action, give_up included) or "none" (no valid answer)."""
    first = next((a["action"] for a in run["attempts"] if a.get("source", "llm") == "llm" and a.get("action")), None)
    if first is None:
        return "none"
    expected = case["expected_fix"]
    if _same(first, expected):
        return "exact"
    if first.get("action") == expected.get("action") in NAMED and \
            normalize(str(first.get("name", ""))) == normalize(str(expected.get("name", ""))):
        return "same_target"
    if any(_same(first, alt) for alt in alternatives.get(case["id"], [])):
        return "alternative"
    return "wrong"
BASE_PYTHON = "3.9"                     # the taxi image's Python before any injection (Pipfile)
ERROR_RE = re.compile(r"error|exception|fail|traceback|no matching|not found", re.I)


ALTERNATIVES = verified_alternatives()


def load_runs(out: Path, pass_: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / pass_).glob("*/*/seed-*/result.json"))]


def error_line(tail: list[str]) -> str:
    """The last line that looks like an error, else the last line."""
    return next((line for line in reversed(tail) if ERROR_RE.search(line)), tail[-1] if tail else "")


def applied_fixes(run: dict, cases: dict[str, dict]) -> list[dict]:
    """The LLM answers that were applied and checked in the sandbox, each with its verdicts."""
    case = cases[run["case"]]
    docs = {d.id: d.model_dump() for d in loo_docs(run["case"])} if run["memory_ids"] else {}
    python = (case.get("injection") or {}).get("python_version") or BASE_PYTHON
    out = []
    for a in run["attempts"]:
        if a.get("source", "llm") == "llm" and a.get("action") and a.get("sandbox_attempt_id"):
            judged = case if a["n"] == 1 else {**case, "error_signature": error_line(a["error_tail"]),
                                               "error_tail": a["error_tail"]}
            retrieved = [docs[r["id"]] for r in (a.get("retrieved") or []) if r["id"] in docs]
            true, why = faithful(a["reason"] or "", judged, retrieved, a["action_text"] or "", python)
            words = _terms(a["reason"] or "")
            copied = sorted(words & (foreign_terms(judged, retrieved) - _terms(a["action_text"] or "")
                                     - _terms(case["cause"]["summary"])))
            out.append({"n": a["n"], "fix": a["action_text"], "outcome": a["outcome"], "reason": a["reason"],
                        "error_line": judged["error_signature"], "true": true, "why": why,
                        "copied": bool(copied), "copied_terms": copied})
        if a.get("action", {}) and a["action"].get("action") == "set_python_version" \
                and not str(a["outcome"]).startswith(("reverted", "not applied")):
            python = a["action"]["version"]
    return out


def per_case(runs: list[dict], cases: dict[str, dict], metric: str, only: list[str] | None = None) -> dict:
    out: dict[str, list[int]] = {}
    for r in runs:
        if only and r["case"] not in only:
            continue
        if metric == "first_fix_passed":
            vals = [int(r["first_fix_passed"])]
        elif metric == "fixed":
            vals = [int(r["final_ok"])]
        elif metric == "action":
            vals = [int(action_match(r, cases[r["case"]], ALTERNATIVES) in ("exact", "same_target", "alternative"))]
        else:
            vals = [int(f[metric]) for f in applied_fixes(r, cases)]
        out.setdefault(r["case"], []).extend(vals)
    return out


def describe(pc: dict) -> dict:
    k, n, p = stats.rate(pc)
    return {"k": k, "n": n, "rate": p, "bootstrap95": stats.cluster_bootstrap(pc), "wilson95": stats.wilson(k, n)}


def analyze(runs: list[dict]) -> dict:
    cases = {c["id"]: c for c in all_cases()}
    configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in runs)]
    by = {c: [r for r in runs if r["config"] == c] for c in configs}
    metrics = list(RUN_METRICS) + list(FIX_METRICS)
    res = {"runs": len(runs), "configs": {}, "comparisons": {}, "no_sibling": {}, "families": {},
           "filter_git": subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--",
                                         "builder_agent/finetune/__init__.py"], capture_output=True, text=True
                                        ).stdout.strip()}
    for c in configs:
        res["configs"][c] = {m: describe(per_case(by[c], cases, m)) for m in metrics + list(SECONDARY)}
        res["no_sibling"][c] = {m: describe(per_case(by[c], cases, m, NO_SIBLING)) for m in metrics + list(SECONDARY)}
        kinds = [action_match(r, cases[r["case"]], ALTERNATIVES) for r in by[c]]
        res["configs"][c]["action_kinds"] = {k: kinds.count(k) for k in
                                             ("exact", "same_target", "alternative", "wrong", "none")}
        fams = sorted({r["family"] for r in by[c]})
        res["families"][c] = {f: describe(per_case([r for r in by[c] if r["family"] == f], cases, "first_fix_passed"))
                              for f in fams}
    for m in metrics:
        raw = {}
        for a, b in combinations(configs, 2):
            pa, pb = per_case(by[b], cases, m), per_case(by[a], cases, m)        # later config minus earlier one
            ma, mb = stats.case_means(pa), stats.case_means(pb)
            pos, neg, p = stats.sign_test([ma[k] - mb[k] for k in ma if k in mb])
            key = f"{b} - {a}"
            res["comparisons"].setdefault(m, {})[key] = {"paired_bootstrap": stats.paired_bootstrap(pa, pb),
                                                          "sign_test": {"better": pos, "worse": neg, "p": p}}
            raw[key] = p
            if m in RUN_METRICS:
                pairs = [(int(rb[m == "fixed" and "final_ok" or m]), int(ra[m == "fixed" and "final_ok" or m]))
                         for rb in by[b] for ra in by[a] if (rb["case"], rb["seed"]) == (ra["case"], ra["seed"])]
                res["comparisons"][m][key]["mcnemar"] = stats.mcnemar(pairs)
        for key, adj in stats.holm(raw).items():
            cmp_ = res["comparisons"][m][key]
            cmp_["sign_test"]["p_holm"] = adj
            pb = cmp_["paired_bootstrap"]
            shown = pb is not None and (pb[1] > 0 or pb[2] < 0) and adj < 0.05
            cmp_["verdict"] = (("better" if pb[0] > 0 else "worse") if shown else "no difference shown")
    return res


def _fmt(d: dict) -> str:
    if d["rate"] is None:
        return "–"
    b, w = d["bootstrap95"], d["wilson95"]
    return f"{d['rate']:.0%} ({d['k']}/{d['n']}); 95% {b[0]:.0%}–{b[1]:.0%} [Wilson {w[0]:.0%}–{w[1]:.0%}]"


def markdown(res: dict, pass_: str) -> str:
    out = [f"# Leave-one-out results: pass {pass_} ({res['runs']} runs)", "",
           f"Faithfulness filter frozen at commit {res['filter_git']}. Interval: case-cluster bootstrap; "
           "Wilson in brackets, for reference only.", "", "| Metric | " + " | ".join(res["configs"]) + " |",
           "|---" * (len(res["configs"]) + 1) + "|"]
    names = RUN_METRICS | FIX_METRICS | SECONDARY
    for m, name in names.items():
        out.append(f"| {name} | " + " | ".join(_fmt(res["configs"][c][m]) for c in res["configs"]) + " |")
    out += ["", "Action accuracy, first LLM action (exact / same target / verified alternative / wrong / none): "
            + "; ".join(f"{c} " + "/".join(str(v) for v in d["action_kinds"].values()) for c, d in res["configs"].items())
            + ". Secondary: not used to call a config better."]
    out += ["", "## Paired comparisons (later config minus earlier one, per case)", "",
            "| Metric | Pair | Difference, 95% interval | Sign test (better/worse, p, Holm p) | Verdict |",
            "|---|---|---|---|---|"]
    for m, pairs in res["comparisons"].items():
        for key, c in pairs.items():
            pb, st = c["paired_bootstrap"], c["sign_test"]
            diff = f"{pb[0]:+.0%} ({pb[1]:+.0%} to {pb[2]:+.0%})" if pb else "–"
            out.append(f"| {names[m]} | {key} | {diff} | {st['better']}/{st['worse']}, {st['p']:.3f}, "
                       f"{st.get('p_holm', st['p']):.3f} | {c.get('verdict', '–')} |")
    out += ["", "## Cases with no same-fix case in memory (" + ", ".join(NO_SIBLING) + "; descriptive)", "",
            "| Metric | " + " | ".join(res["no_sibling"]) + " |", "|---" * (len(res["no_sibling"]) + 1) + "|"]
    for m, name in names.items():
        out.append(f"| {name} | " + " | ".join(_fmt(res["no_sibling"][c][m]) for c in res["no_sibling"]) + " |")
    out += ["", "## First fix passes per family (descriptive)", ""]
    for c, fams in res["families"].items():
        out.append(f"- {c}: " + "; ".join(f"{f} {d['k']}/{d['n']}" for f, d in fams.items()))
    if "labels" in res:
        lab = res["labels"]
        out += ["", f"## Hand labels: {lab['labelled']} labelled, filter agrees on {lab['agree']} "
                f"({lab['agreement']:.0%}); {'reliable' if lab['agreement'] >= 0.8 else 'UNRELIABLE: below 80%'}"]
    return "\n".join(out) + "\n"


def label_agreement(labels_dir: Path) -> dict | None:
    sheet, key = labels_dir / "to_label.csv", labels_dir / "key.json"
    if not (sheet.exists() and key.exists()):
        return None
    verdicts = {k: v["filter_true"] for k, v in json.loads(key.read_text(encoding="utf-8")).items()}
    rows = [r for r in csv.DictReader(sheet.open(encoding="utf-8")) if r["label"].strip().lower() in ("true", "false")]
    if not rows:
        return None
    agree = sum((r["label"].strip().lower() == "true") == verdicts[r["item"]] for r in rows)
    return {"labelled": len(rows), "agree": agree, "agreement": agree / len(rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_", required=True)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    out = Path(args.out)
    runs = load_runs(out, args.pass_)
    if not runs:
        sys.exit(f"no runs in {out / args.pass_}")
    res = analyze(runs)
    if (lab := label_agreement(out / "labels")) is not None:
        res["labels"] = lab
    (out / args.pass_ / "summary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = markdown(res, args.pass_)
    (out / args.pass_ / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
