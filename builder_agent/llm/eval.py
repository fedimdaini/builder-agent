"""Eval the slot-filling loop against gold answers.

    python -m builder_agent.llm.eval --prompt slots_v1 --model qwen2.5-coder:7b
        [--repo ../taxi-trip-regression] [--gold tests/gold/taxi_slots.json]

Runs the loop once on the repo and compares the final answer with the gold
answer slot by slot. The report is printed and saved under logs/evals/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..decide import load_contracts, plan_build
from ..scan import scan_repo
from . import DEFAULT_CALL_LOG, OllamaClient, fill_slots, load_prompt

ROOT = Path(__file__).resolve().parents[2]
SLOTS = ["target_column", "target_transform", "transform_inside_train_fn", "model_flavor", "model_input",
         "train.train_function", "train.train_file", "train.arg_map",
         "evaluate.eval_file", "evaluate.group_column", "data.data_step"]
MISSING = object()


class SlotScore(BaseModel):
    slot: str
    gold: Any                      # None when absent
    answer: Any                    # None when absent
    correct: bool


class EvalReport(BaseModel):
    time: str
    repo: str
    prompt_version: str
    prompt_sha256: str
    model: str
    model_digest: str
    slots_correct: int
    slots_total: int
    per_slot: list[SlotScore]
    valid_final: bool
    valid_on_first_try: bool
    attempts: int
    rejection_reasons: list[list[str]]   # per attempt
    total_tokens: int
    llm_s: float                          # sum of call latencies
    wall_s: float                         # whole eval, scan included

    def text(self) -> str:
        out = [f"EVAL {self.prompt_version} x {self.model} on {self.repo}",
               f"  slots correct      {self.slots_correct}/{self.slots_total}",
               f"  valid on 1st try   {'yes' if self.valid_on_first_try else 'no'}",
               f"  valid at the end   {'yes' if self.valid_final else 'no'}",
               f"  attempts           {self.attempts}",
               f"  total tokens       {self.total_tokens}",
               f"  time               {self.llm_s:.1f} s in the LLM, {self.wall_s:.1f} s in total",
               "  per slot:"]
        for s in self.per_slot:
            mark = "ok  " if s.correct else "MISS"
            answer = "(none)" if s.answer is None else json.dumps(s.answer)
            detail = "" if s.correct else f"   gold {json.dumps(s.gold)}"
            out.append(f"    {mark} {s.slot:<27} {answer}{detail}")
        for i, reasons in enumerate(self.rejection_reasons, 1):
            for r in reasons:
                out.append(f"  attempt {i} rejected: {r}")
        return "\n".join(out)


def _get(d, dotted: str):
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def compare(answer: dict | None, gold: dict) -> list[SlotScore]:
    scores = []
    for slot in SLOTS:
        g, a = _get(gold, slot), _get(answer or {}, slot)
        scores.append(SlotScore(slot=slot, gold=None if g is MISSING else g, answer=None if a is MISSING else a,
                                correct=a is not MISSING and a == g))
    return scores


def run_eval(prompt_version: str, model: str, repo: str | Path, gold_path: str | Path,
             contracts_path: str | Path = ROOT / "contracts.yaml", call_log: str | Path | None = DEFAULT_CALL_LOG,
             client=None, prompts_dir: str | Path | None = None) -> EvalReport:
    t0 = time.perf_counter()
    prompt = load_prompt(prompt_version, prompts_dir) if prompts_dir else load_prompt(prompt_version)
    ctx = scan_repo(repo)
    plan = plan_build(ctx, load_contracts(contracts_path))
    client = client or OllamaClient(model)
    result = fill_slots(ctx, plan, prompt, client, log_path=call_log)
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    scores = compare(result.answer, gold)
    return EvalReport(
        time=datetime.now(timezone.utc).isoformat(), repo=ctx.name, prompt_version=prompt.version,
        prompt_sha256=prompt.sha256, model=result.model, model_digest=result.model_digest,
        slots_correct=sum(s.correct for s in scores), slots_total=len(scores), per_slot=scores,
        valid_final=result.valid, valid_on_first_try=result.valid_on_first_try, attempts=len(result.attempts),
        rejection_reasons=[a.reasons for a in result.attempts], total_tokens=result.total_tokens,
        llm_s=result.total_s, wall_s=round(time.perf_counter() - t0, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.llm.eval", description=__doc__)
    parser.add_argument("--prompt", required=True, help="prompt version, e.g. slots_v1 (prompts/<version>.md)")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2.5-coder:7b")
    parser.add_argument("--repo", default=str(ROOT.parent / "taxi-trip-regression"))
    parser.add_argument("--gold", default=str(ROOT / "tests" / "gold" / "taxi_slots.json"))
    args = parser.parse_args()

    report = run_eval(args.prompt, args.model, args.repo, args.gold)
    out_dir = ROOT / "logs" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{args.prompt}__{re.sub(r'[^A-Za-z0-9.-]+', '-', args.model)}__{stamp}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report.text())
    print(f"saved {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
