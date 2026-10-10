"""python -m builder_agent.finetune sample <repo> --slots S      # render prompts, sample 8 answers per case
python -m builder_agent.finetune check  <repo> --slots S [--expected E]   # sandbox-check possible alternatives
python -m builder_agent.finetune check  <repo> --slots S --loose            # sandbox-check the loose pins
python -m builder_agent.finetune rationalize <repo> --slots S   # STaR: 4 hinted reasons per memory fault
python -m builder_agent.finetune build                          # label, filter, write sft.jsonl / dpo.jsonl

Files in experiments/finetune/: prompts.json, samples.jsonl, labels.jsonl, checks.json, rationalized.jsonl,
sft.jsonl, dpo.jsonl, report.md."""

import argparse
import json
import sys
from pathlib import Path

from ..llm.ollama import OllamaClient, OllamaEmbedder
from ..render.slots import load_answer
from ..sandbox import DEFAULT_LOG, run_sandbox
from . import (OUT_DIR, SAMPLES_PER_CASE, TEMPERATURE, alternative_overrides, build_datasets, faithful, label_sample,
               read_jsonl, render_case_prompt, sample_case, split_cases, write_json, write_jsonl, CasePrompt)
from .rationalize import RATIONALIZE_SAMPLES, judge, rationalize_case, training_rows


def cmd_sample(args) -> None:
    slots, _ = load_answer(args.slots)
    embedder = OllamaEmbedder(args.embed_model)
    prompts, samples, labels = {}, [], []
    for case in split_cases("memory"):
        cp = render_case_prompt(case, args.repo, slots, embedder)
        prompts[case["id"]] = cp.model_dump()

        def factory(seed, model=args.model):
            return OllamaClient(model, options={"temperature": TEMPERATURE, "seed": seed, "num_ctx": 8192},
                                keep_alive="5m")
        rows = sample_case(cp, factory, args.n)
        samples += rows
        labels += [r | label_sample(r, case, cp) for r in rows]
        print(f"{case['id']} {case['variant']}: retrieved {[h['id'] for h in cp.retrieved]}; "
              + ", ".join(labels[-k]["label"] for k in range(len(rows), 0, -1)), flush=True)
    OllamaClient(args.model).unload()
    write_json(OUT_DIR / "prompts.json", prompts)
    write_jsonl(OUT_DIR / "samples.jsonl", samples)
    write_jsonl(OUT_DIR / "labels.jsonl", labels)


def _check_key(row: dict) -> str:
    return row["case_id"] + " " + json.dumps(row["fix"], sort_keys=True)


def _to_check(row: dict, loose: bool) -> bool:
    """check: the possible alternatives; check --loose: the loose matches (same package, another version)."""
    return row["label"] == "correct" and row["match"] == "loose" if loose else row["label"] == "candidate"


def cmd_check(args) -> None:
    slots, _ = load_answer(args.slots)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    cases = {c["id"]: c for c in split_cases("memory")}
    path = OUT_DIR / "checks.json"
    checks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = {}
    for row in read_jsonl(OUT_DIR / "labels.jsonl"):
        if _to_check(row, args.loose) and _check_key(row) not in checks:
            todo[_check_key(row)] = row
    print(f"{len(todo)} distinct {'loose matches' if args.loose else 'possible alternatives'} to check", flush=True)
    for key, row in todo.items():
        overrides, mlflow_client = alternative_overrides(cases[row["case_id"]], row["fix"])
        r = run_sandbox(args.repo, slots, expected=expected, log_path=DEFAULT_LOG, overrides=overrides,
                        mlflow_client=mlflow_client)
        checks[key] = {"case_id": row["case_id"], "fix": row["fix"], "ok": r.ok, "failed_stage": r.failed_stage,
                       "attempt_id": r.attempt_id, "kind": "loose" if args.loose else "alternative"}
        write_json(path, checks)                       # after each run, so an interrupted check resumes
        print(f"{row['case_id']} {row['fix_text']}: {'passed' if r.ok else 'failed at ' + str(r.failed_stage)}",
              flush=True)


def cmd_build(args) -> None:
    prompts = json.loads((OUT_DIR / "prompts.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in split_cases("memory")}
    path = OUT_DIR / "checks.json"
    checks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows = []
    for row in read_jsonl(OUT_DIR / "labels.jsonl"):
        if row["label"] == "candidate":
            check = checks.get(_check_key(row))
            if check is None:
                sys.exit(f"unchecked candidate {_check_key(row)}: run the check step first")
            row = row | {"label": "correct" if check["ok"] else "wrong", "match": "sandbox" if check["ok"] else None,
                         "sandbox_attempt_id": check["attempt_id"]}
        elif row["label"] == "correct" and row["match"] == "loose" and _check_key(row) in checks:
            check = checks[_check_key(row)]                # a loose pin that fails the sandbox is wrong
            row = row | {"label": "correct" if check["ok"] else "wrong", "match": "loose" if check["ok"] else None,
                         "sandbox_verified": check["ok"], "sandbox_attempt_id": check["attempt_id"]}
        if row["label"] == "correct":
            cp = CasePrompt(**prompts[row["case_id"]])
            ok, why = faithful(row["reason"], cases[row["case_id"]], cp.retrieved, row.get("fix_text", ""),
                               cp.current_python)
            row = row | {"faithful": ok, "faithful_why": why}
        rows.append(row)
    rationalized = OUT_DIR / "rationalized.jsonl"
    if rationalized.exists():
        cps = {cid: CasePrompt(**p) for cid, p in prompts.items()}
        # judged again, so the current filter applies to the stored hinted answers
        hinted = [r | judge(r, cases[r["case_id"]], cps[r["case_id"]]) for r in read_jsonl(rationalized)]
        write_jsonl(OUT_DIR / "rationalized_final.jsonl", hinted)
        rows += training_rows(hinted, cases, cps)
    sft, dpo, stats = build_datasets(prompts, rows)
    write_jsonl(OUT_DIR / "sft.jsonl", sft)
    write_jsonl(OUT_DIR / "dpo.jsonl", dpo)
    write_jsonl(OUT_DIR / "labels_final.jsonl", rows)
    write_json(OUT_DIR / "stats.json", stats)
    print(json.dumps(stats, indent=2))


def cmd_rationalize(args) -> None:
    prompts = {cid: CasePrompt(**p) for cid, p in
               json.loads((OUT_DIR / "prompts.json").read_text(encoding="utf-8")).items()}
    out = []
    for case in split_cases("memory"):
        cp = prompts[case["id"]]                       # the same prompt as the sampling step

        def factory(seed, model=args.model):
            return OllamaClient(model, options={"temperature": TEMPERATURE, "seed": seed, "num_ctx": 8192},
                                keep_alive="5m")
        rows = rationalize_case(cp, case, factory, args.n)
        out += rows
        print(f"{case['id']} {case['variant']}: kept {sum(r['kept'] for r in rows)} of {len(rows)}", flush=True)
    OllamaClient(args.model).unload()
    write_jsonl(OUT_DIR / "rationalized.jsonl", out)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.finetune", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "check", "rationalize"):
        s = sub.add_parser(name)
        s.add_argument("repo")
        s.add_argument("--slots", required=True)
        s.add_argument("--expected")
        s.add_argument("--model", default="qwen2.5-coder:7b")
        s.add_argument("--embed-model", default="nomic-embed-text")
        s.add_argument("--n", type=int, default=RATIONALIZE_SAMPLES if name == "rationalize" else SAMPLES_PER_CASE)
        s.add_argument("--loose", action="store_true", help="check: sandbox-check the loose matches instead")
    sub.add_parser("build")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"sample": cmd_sample, "check": cmd_check, "rationalize": cmd_rationalize, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
