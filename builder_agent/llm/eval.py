"""Eval the slot-filling loop against gold answers.

    python -m builder_agent.llm.eval --prompt slots_v1 --model qwen2.5-coder:7b --label "v1, fixed system"
        [--repo ../taxi-trip-regression] [--gold tests/gold/taxi_slots.json]
        [--sandbox [--expected tests/gold/taxi_expected.json]]
    python -m builder_agent.llm.eval --compare experiments/evals/a.json experiments/evals/b.json

Runs the loop once on the repo and compares the final answer with the gold
answer slot by slot: strict (the gold value only) and lenient (the gold value
or one of the gold file's "_alternatives"). With --sandbox, the final answer
is also run end to end in the sandbox. Reports are saved in experiments/evals/.
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
from ..render.slots import load_answer
from ..scan import scan_repo
from . import DEFAULT_CALL_LOG, OllamaClient, fill_slots, load_prompt

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "experiments" / "evals"
SLOTS = ["target_column", "target_transform", "transform_inside_train_fn", "model_flavor", "model_input",
         "train.train_function", "train.train_file", "train.arg_map",
         "evaluate.eval_file", "evaluate.group_column", "data.data_step"]
MISSING = object()


class SlotScore(BaseModel):
    slot: str
    gold: Any                      # None when absent
    answer: Any                    # None when absent
    correct: bool                  # strict: equals the gold value
    correct_lenient: bool = False  # equals the gold value or an accepted alternative


class SandboxOutcome(BaseModel):
    ok: bool
    failed_stage: str | None
    stages: dict[str, str]
    prediction: float | None = None
    attempt_id: str


class EvalReport(BaseModel):
    label: str | None = None
    time: str
    repo: str
    prompt_version: str
    prompt_sha256: str
    model: str
    model_digest: str
    slots_correct: int
    slots_correct_lenient: int | None = None
    slots_total: int
    per_slot: list[SlotScore]
    final_answer: dict | None = None
    valid_final: bool
    valid_on_first_try: bool
    attempts: int
    rejection_reasons: list[list[str]]   # per attempt
    total_tokens: int
    llm_s: float                          # sum of call latencies
    wall_s: float                         # loop and scan (sandbox excluded)
    sandbox: SandboxOutcome | None = None

    def text(self) -> str:
        out = [f"EVAL {self.label or ''} ({self.prompt_version} x {self.model} on {self.repo})",
               f"  slots correct      {self.slots_correct}/{self.slots_total} strict, "
               f"{self.slots_correct_lenient}/{self.slots_total} lenient",
               f"  valid on 1st try   {'yes' if self.valid_on_first_try else 'no'}",
               f"  valid at the end   {'yes' if self.valid_final else 'no'}",
               f"  attempts           {self.attempts}",
               f"  total tokens       {self.total_tokens}",
               f"  time               {self.llm_s:.1f} s in the LLM, {self.wall_s:.1f} s in total"]
        if self.sandbox:
            sb = self.sandbox
            out.append("  end to end         " + ("all stages ok" if sb.ok else f"FAILED at {sb.failed_stage}")
                       + (f", prediction {sb.prediction:.0f} s" if sb.prediction is not None else "")
                       + f" (sandbox attempt {sb.attempt_id})")
        out.append("  per slot:")
        for s in self.per_slot:
            mark = "ok  " if s.correct else ("alt " if s.correct_lenient else "MISS")
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


def compare(answer: dict | None, gold: dict, alternatives: dict[str, list] | None = None) -> list[SlotScore]:
    scores = []
    for slot in SLOTS:
        g, a = _get(gold, slot), _get(answer or {}, slot)
        strict = a is not MISSING and a == g
        lenient = strict or (a is not MISSING and a in (alternatives or {}).get(slot, []))
        scores.append(SlotScore(slot=slot, gold=None if g is MISSING else g, answer=None if a is MISSING else a,
                                correct=strict, correct_lenient=lenient))
    return scores


def run_eval(prompt_version: str, model: str, repo: str | Path, gold_path: str | Path,
             contracts_path: str | Path = ROOT / "contracts.yaml", call_log: str | Path | None = DEFAULT_CALL_LOG,
             client=None, prompts_dir: str | Path | None = None, label: str | None = None,
             sandbox: bool = False, expected: dict | None = None, sandbox_runner=None) -> EvalReport:
    t0 = time.perf_counter()
    prompt = load_prompt(prompt_version, prompts_dir) if prompts_dir else load_prompt(prompt_version)
    ctx = scan_repo(repo)
    plan = plan_build(ctx, load_contracts(contracts_path))
    client = client or OllamaClient(model)
    result = fill_slots(ctx, plan, prompt, client, log_path=call_log)
    gold, alternatives = load_answer(gold_path)
    scores = compare(result.answer, gold, alternatives)
    report = EvalReport(
        label=label, time=datetime.now(timezone.utc).isoformat(), repo=ctx.name, prompt_version=prompt.version,
        prompt_sha256=prompt.sha256, model=result.model, model_digest=result.model_digest,
        slots_correct=sum(s.correct for s in scores), slots_correct_lenient=sum(s.correct_lenient for s in scores),
        slots_total=len(scores), per_slot=scores, final_answer=result.answer,
        valid_final=result.valid, valid_on_first_try=result.valid_on_first_try, attempts=len(result.attempts),
        rejection_reasons=[a.reasons for a in result.attempts], total_tokens=result.total_tokens,
        llm_s=result.total_s, wall_s=round(time.perf_counter() - t0, 2),
    )
    if sandbox and result.valid:
        report.sandbox = run_answer_in_sandbox(repo, result.answer, expected, contracts_path, sandbox_runner)
    return report


def run_answer_in_sandbox(repo, answer: dict, expected: dict | None, contracts_path, runner=None) -> SandboxOutcome:
    from ..sandbox import run_sandbox, subprocess_runner
    r = run_sandbox(repo, answer, expected=expected, contracts_path=contracts_path,
                    runner=runner or subprocess_runner)
    prediction = None
    for s in r.stages:
        for line in s.output_tail:
            if m := re.match(r"prediction: (-?[\d.eE+-]+)$", line.strip()):
                prediction = float(m.group(1))
    return SandboxOutcome(ok=r.ok, failed_stage=r.failed_stage, stages={s.name: s.status for s in r.stages},
                          prediction=prediction, attempt_id=r.attempt_id)


def side_by_side(reports: list[EvalReport]) -> str:
    width = 30
    head = f"{'':<29}" + "".join(f"{(r.label or r.prompt_version)[:width - 2]:<{width}}" for r in reports)
    rows = [
        ("slots correct (strict)", lambda r: f"{r.slots_correct}/{r.slots_total}"),
        ("slots correct (lenient)", lambda r: f"{r.slots_correct_lenient}/{r.slots_total}"),
        ("valid on 1st try", lambda r: "yes" if r.valid_on_first_try else "no"),
        ("valid at the end", lambda r: "yes" if r.valid_final else "no"),
        ("attempts", lambda r: str(r.attempts)),
        ("total tokens", lambda r: str(r.total_tokens)),
        ("LLM time", lambda r: f"{r.llm_s:.1f} s"),
        ("end to end", lambda r: "not run" if not r.sandbox else
            ("all stages ok" if r.sandbox.ok else f"failed at {r.sandbox.failed_stage}")),
    ]
    out = [head] + [f"  {name:<27}" + "".join(f"{fn(r):<{width}}" for r in reports) for name, fn in rows]
    out.append("  per slot:")
    for i, slot in enumerate(SLOTS):
        cells = []
        for r in reports:
            s = r.per_slot[i]
            mark = "ok" if s.correct else ("alt" if s.correct_lenient else "MISS")
            cells.append(f"{mark} {json.dumps(s.answer)}"[:width - 2])
        out.append(f"    {slot:<25}" + "".join(f"{c:<{width}}" for c in cells))
    return "\n".join(out)


def save_report(report: EvalReport, out_dir: Path = REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report.time).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{report.prompt_version}__{re.sub(r'[^A-Za-z0-9.-]+', '-', report.model)}__{stamp}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.llm.eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="prompt version, e.g. slots_v1 (prompts/<version>.md)")
    parser.add_argument("--model", help="Ollama model name, e.g. qwen2.5-coder:7b")
    parser.add_argument("--label", help='name of this run in reports, e.g. "v1, fixed system"')
    parser.add_argument("--repo", default=str(ROOT.parent / "taxi-trip-regression"))
    parser.add_argument("--gold", default=str(ROOT / "tests" / "gold" / "taxi_slots.json"))
    parser.add_argument("--sandbox", action="store_true", help="also run the final answer in the sandbox")
    parser.add_argument("--expected", default=str(ROOT / "tests" / "gold" / "taxi_expected.json"))
    parser.add_argument("--compare", nargs="+", metavar="REPORT", help="print saved reports side by side")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.compare:
        reports = [EvalReport.model_validate_json(Path(p).read_text(encoding="utf-8")) for p in args.compare]
        print(side_by_side(reports))
        return
    if not (args.prompt and args.model):
        parser.error("--prompt and --model are required (or use --compare)")
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.sandbox else None
    report = run_eval(args.prompt, args.model, args.repo, args.gold, label=args.label,
                      sandbox=args.sandbox, expected=expected)
    out = save_report(report)
    print(report.text())
    print(f"saved {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
