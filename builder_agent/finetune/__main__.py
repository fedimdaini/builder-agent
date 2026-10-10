"""python -m builder_agent.finetune sample <repo> --slots S      # render prompts, sample 8 answers per case
python -m builder_agent.finetune check  <repo> --slots S [--expected E]   # sandbox-check possible alternatives
python -m builder_agent.finetune build                          # label, filter, write sft.jsonl / dpo.jsonl

Files in experiments/finetune/: prompts.json, samples.jsonl, labels.jsonl, checks.json, sft.jsonl,
dpo.jsonl, report.md."""

import argparse
import json
import sys
from pathlib import Path

from ..llm.ollama import OllamaClient, OllamaEmbedder
from ..render.slots import load_answer
from ..sandbox import DEFAULT_LOG, run_sandbox
from . import (OUT_DIR, SAMPLES_PER_CASE, TEMPERATURE, alternative_overrides, build_datasets, faithful, label_sample,
               read_jsonl, render_case_prompt, sample_case, split_cases, write_json, write_jsonl, CasePrompt)


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


def cmd_check(args) -> None:
    slots, _ = load_answer(args.slots)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    cases = {c["id"]: c for c in split_cases("memory")}
    path = OUT_DIR / "checks.json"
    checks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = {}
    for row in read_jsonl(OUT_DIR / "labels.jsonl"):
        if row["label"] == "candidate" and _check_key(row) not in checks:
            todo[_check_key(row)] = row
    print(f"{len(todo)} distinct possible alternatives to check", flush=True)
    for key, row in todo.items():
        overrides, mlflow_client = alternative_overrides(cases[row["case_id"]], row["fix"])
        r = run_sandbox(args.repo, slots, expected=expected, log_path=DEFAULT_LOG, overrides=overrides,
                        mlflow_client=mlflow_client)
        checks[key] = {"case_id": row["case_id"], "fix": row["fix"], "ok": r.ok, "failed_stage": r.failed_stage,
                       "attempt_id": r.attempt_id}
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
        if row["label"] == "correct":
            cp = CasePrompt(**prompts[row["case_id"]])
            ok, why = faithful(row["reason"], cases[row["case_id"]], cp.retrieved, row.get("fix_text", ""),
                               cp.current_python)
            row = row | {"faithful": ok, "faithful_why": why}
        rows.append(row)
    sft, dpo, stats = build_datasets(prompts, rows)
    write_jsonl(OUT_DIR / "sft.jsonl", sft)
    write_jsonl(OUT_DIR / "dpo.jsonl", dpo)
    write_jsonl(OUT_DIR / "labels_final.jsonl", rows)
    write_json(OUT_DIR / "stats.json", stats)
    print(json.dumps(stats, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.finetune", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "check"):
        s = sub.add_parser(name)
        s.add_argument("repo")
        s.add_argument("--slots", required=True)
        s.add_argument("--expected")
        s.add_argument("--model", default="qwen2.5-coder:7b")
        s.add_argument("--embed-model", default="nomic-embed-text")
        s.add_argument("--n", type=int, default=SAMPLES_PER_CASE)
    sub.add_parser("build")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"sample": cmd_sample, "check": cmd_check, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
