"""Fine-tuning data (roadmap item 9 step 3): SFT examples and DPO pairs from the memory-split faults.

    python -m builder_agent.finetune sample  ../taxi-trip-regression --slots tests/gold/taxi_slots.json
    python -m builder_agent.finetune check   ../taxi-trip-regression --slots ... --expected ...   # sandbox
    python -m builder_agent.finetune build

Only generated cases with split "memory" are used; a split "test" case is never rendered, sampled or
written (DataError). For each case the diagnose_v3 prompt is rendered as the fix loop renders its
first attempt, with the advanced retriever over a memory that leaves the case itself out (and
fault-001 for gen-008, the same fault), so the model can't copy the answer. Answers are sampled at
temperature 0.7 and labelled against the case's verified fix; non-matching answers that touch
something named in the error are checked in the sandbox as possible alternatives.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from ..decide import load_contracts, plan_build
from ..fix import _installed, _python_packages, prompt_variables
from ..fix.models import AddDependency, AddSystemPackage, Diagnosis, PinPackage, SetEnvVar, SetPythonVersion, \
    answer_model, apply_fix, describe, validate_fix
from ..llm.ollama import inline_refs
from ..llm.prompts import load_prompt
from ..memory import FixMemory, memory_cases, render_retrieved, tokens
from ..render import validate_slots
from ..render.configs import Overrides, config_context
from ..sandbox import DEFAULT_CONTRACTS, SandboxResult, StageResult
from ..scan import scan_repo
from ..scan.deps import normalize

OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "finetune"
GENERATED = Path(__file__).resolve().parents[2] / "tests" / "faults" / "generated"
PROMPT = "diagnose_v3"
SAMPLES_PER_CASE = 8
TEMPERATURE = 0.7
# memory documents that are the same fault as a case, left out with it (gen-008's cause says so)
SAME_FAULT = {"gen-008": {"fault-001"}}
# words that say nothing about which error it is
STOP = {"the", "and", "for", "with", "from", "not", "error", "errors", "failed", "failure", "module", "named",
        "file", "line", "object", "exception", "exceptions", "has", "have", "can", "cannot", "this", "that",
        "into", "was", "are", "any", "found", "such", "api", "request", "code", "type", "mean", "did", "you"}


class DataError(ValueError):
    pass


def split_cases(split: str = "memory") -> list[dict]:
    return [c for c in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(GENERATED.glob("*/case.json")))
            if c.get("split") == split]


def _fault(case: dict) -> tuple[Overrides, str | None]:
    fault = dict(case["injection"])
    mlflow_client = fault.pop("mlflow_client", None)
    return Overrides(**fault), mlflow_client


# --- prompts -----------------------------------------------------------------------------------------

class CasePrompt(BaseModel):
    case_id: str
    variant: str
    family: str
    messages: list[dict]
    schema_: dict
    prompt_version: str
    prompt_sha256: str
    retrieved: list[dict]
    excluded: list[str]
    current_python: str
    installed: list[str]
    python_packages: list[str]


def render_case_prompt(case: dict, repo, slots: dict, embedder, contracts_path=DEFAULT_CONTRACTS) -> CasePrompt:
    """The fix loop's first-attempt prompt for this case, with the case left out of the memory."""
    if case.get("split") != "memory":
        raise DataError(f"{case['id']} has split {case.get('split')!r}: only split=memory cases make training data")
    ctx = scan_repo(repo)
    c = load_contracts(contracts_path)
    plan = plan_build(ctx, c)
    injection, mlflow_client = _fault(case)
    tctx = config_context(ctx, c, plan, validate_slots(ctx, slots).slots, mlflow_client, injection)
    failed = StageResult(name=case["stage"], status="failed", command=case["command"],
                         exit_code=case.get("exit_code"), output_tail=case["error_tail"])
    r = SandboxResult(attempt_id=case["found_by"]["attempt_id"], repo=ctx.name, started_at="", ok=False,
                      failed_stage=case["stage"], total_s=0, workdir="", compose_project="", stages=[failed])
    excluded = {case["id"]} | SAME_FAULT.get(case["id"], set())
    memory = FixMemory.build(embedder, cases=[m for m in memory_cases() if m["id"] not in excluded])
    hits = memory.advanced(case["stage"], case["error_tail"])
    variables = prompt_variables(ctx, r, tctx, 1, []) | {"retrieved_text": render_retrieved(hits)}
    prompt = load_prompt(PROMPT)
    model = answer_model("\n".join(prompt.sections.values()))
    return CasePrompt(case_id=case["id"], variant=case["variant"], family=case["family"],
                      messages=[{"role": "system", "content": prompt.render("system", variables)},
                                {"role": "user", "content": prompt.render("user", variables)}],
                      schema_=inline_refs(model.model_json_schema()), prompt_version=prompt.version,
                      prompt_sha256=prompt.sha256,
                      retrieved=[{"id": h.doc.id, "variant": h.doc.variant, "error": h.doc.error,
                                  "cause": h.doc.cause, "fix": h.doc.fix} for h in hits],
                      excluded=sorted(excluded), current_python=tctx["python_version"],
                      installed=sorted(_installed(ctx, tctx)), python_packages=sorted(_python_packages(ctx)))


# --- sampling ----------------------------------------------------------------------------------------------

def sample_case(cp: CasePrompt, client_factory, n: int = SAMPLES_PER_CASE) -> list[dict]:
    """n answers, one per seed (1..n), at TEMPERATURE. client_factory(seed) -> a chat client."""
    out = []
    for seed in range(1, n + 1):
        client = client_factory(seed)
        t0 = time.perf_counter()
        try:
            resp = client.chat(cp.messages, schema=cp.schema_)
            content, err, latency = resp.content, None, resp.latency_s
            tokens_in, tokens_out = resp.prompt_tokens, resp.output_tokens
        except Exception as e:  # noqa: BLE001 - logged as a failed sample
            content, err, latency = None, f"{type(e).__name__}: {e}", round(time.perf_counter() - t0, 2)
            tokens_in = tokens_out = None
        out.append({"time": datetime.now(timezone.utc).isoformat(), "case_id": cp.case_id, "variant": cp.variant,
                    "family": cp.family, "seed": seed, "model": client.model, "model_digest": client.digest,
                    "options": client.options, "prompt_version": cp.prompt_version,
                    "prompt_sha256": cp.prompt_sha256, "messages_sha256": messages_sha(cp.messages),
                    "response": content, "error": err, "latency_s": latency,
                    "prompt_tokens": tokens_in, "output_tokens": tokens_out})
    return out


def messages_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


# --- labels --------------------------------------------------------------------------------------------------

def _target(fix) -> str | None:
    if isinstance(fix, (PinPackage, AddDependency, AddSystemPackage)):
        return normalize(fix.name)
    if isinstance(fix, SetEnvVar):
        return fix.name
    if isinstance(fix, SetPythonVersion):
        return fix.version
    return None


def match(fix, expected: dict) -> str | None:
    """'exact': same action and arguments; 'loose': pin_package of the same package, another version.
    Anything else (another package, an env var with another value, another Python version): None."""
    exp = Diagnosis.model_validate({"fix": expected, "reason": "x"}).fix
    if type(fix) is not type(exp) or _target(fix) != _target(exp):
        return None
    if fix.model_dump() == exp.model_dump():
        return "exact"
    return "loose" if isinstance(fix, PinPackage) else None


def might_be_alternative(fix, case: dict) -> bool:
    """Worth a sandbox check: a valid fix whose target is named in the case's own error output, or the
    expected action with another value (an env var value, a Python version)."""
    target = _target(fix)
    if target is None:
        return False
    exp = Diagnosis.model_validate({"fix": case["expected_fix"], "reason": "x"}).fix
    same_action = type(fix) is type(exp) and (not isinstance(fix, SetEnvVar) or fix.name == exp.name)
    text = "\n".join(case["error_tail"]).lower()
    return same_action or target.lower() in text


def label_sample(sample: dict, case: dict, cp: CasePrompt) -> dict:
    """correct (exact/loose), candidate (needs a sandbox check), wrong or invalid."""
    base = {"label": "invalid", "match": None, "fix": None, "reason": None, "validation": []}
    if sample["response"] is None:
        return base | {"validation": [sample["error"]]}
    try:
        answer = json.loads(sample["response"])
    except json.JSONDecodeError as e:
        return base | {"validation": [f"not JSON: {e}"]}
    injection, _ = _fault(case)
    v = validate_fix(answer, injection, set(cp.installed), Diagnosis, set(cp.python_packages),
                     current_python=cp.current_python) if isinstance(answer, dict) else None
    if v is None or not v.ok:
        return base | {"validation": v.reasons if v else ["not one JSON object"],
                       "fix": answer.get("fix") if isinstance(answer, dict) else None,
                       "reason": answer.get("reason") if isinstance(answer, dict) else None}
    fix = v.diagnosis.fix
    m = match(fix, case["expected_fix"])
    label = "correct" if m else ("candidate" if might_be_alternative(fix, case) else "wrong")
    return {"label": label, "match": m, "fix": fix.model_dump(), "fix_text": describe(fix),
            "reason": v.diagnosis.reason, "validation": []}


def alternative_overrides(case: dict, fix: dict) -> tuple[Overrides, str | None]:
    injection, mlflow_client = _fault(case)
    return apply_fix(injection, Diagnosis.model_validate({"fix": fix, "reason": "x"}).fix), mlflow_client


# --- faithful reasons ------------------------------------------------------------------------------------------

def _terms(text: str) -> set[str]:
    out = set()
    for t in tokens(text):
        out.add(t)
        out |= set(t.split("."))
    return {t for t in out if t not in STOP and len(t) >= 3}     # drops "2", "17" from split versions


def key_terms(case: dict) -> set[str]:
    """Terms of the case's own error signature."""
    return _terms(case["error_signature"])


def foreign_terms(case: dict, retrieved: list[dict]) -> set[str]:
    """Terms that point to a retrieved other case and appear nowhere in this case's error output:
    its error line's terms, and the versions, ports and library names (terms with a digit) of its cause."""
    own = _terms("\n".join(case["error_tail"])) | key_terms(case)
    out = set()
    for r in retrieved:
        out |= _terms(r["error"]) | {t for t in _terms(r["cause"]) if any(ch.isdigit() for ch in t)}
    return out - own


def faithful(reason: str, case: dict, retrieved: list[dict], fix_text: str = "") -> tuple[bool, str]:
    """The reason names a term of this case's error line and nothing that only a retrieved other case has
    (the answer's own fix, e.g. the version it pins, doesn't count as copied)."""
    words = _terms(reason or "")
    own = words & key_terms(case)
    foreign = words & (foreign_terms(case, retrieved) - _terms(fix_text))
    if foreign:
        return False, f"names {', '.join(sorted(foreign))} from a retrieved case, not from this error"
    if not own:
        return False, "names no term of this case's error line"
    return True, f"refers to {', '.join(sorted(own))}"


# --- datasets ---------------------------------------------------------------------------------------------------

def _assistant(row: dict) -> str:
    return json.dumps({"fix": row["fix"], "reason": row["reason"]}, ensure_ascii=False)


def build_datasets(prompts: dict[str, dict], rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """prompts: case_id -> CasePrompt dump; rows: labelled samples (label in correct/wrong/invalid after the
    sandbox checks). Returns sft examples, dpo pairs and per-family counts."""
    sft, dpo, stats = [], [], {}
    for case_id, cp in prompts.items():
        case_rows = [r for r in rows if r["case_id"] == case_id]
        fam = stats.setdefault(cp["family"], {"cases": 0, "samples": 0, "correct": 0, "correct_dropped_unfaithful": 0,
                                              "wrong": 0, "invalid": 0, "sft_examples": 0, "dpo_pairs": 0})
        fam["cases"] += 1
        fam["samples"] += len(case_rows)
        chosen, rejected, seen_c, seen_r = [], [], set(), set()
        for r in case_rows:
            if r["label"] == "correct":
                fam["correct"] += 1
                if not r["faithful"]:
                    fam["correct_dropped_unfaithful"] += 1
                    continue
                key = _assistant(r)
                if key not in seen_c:
                    seen_c.add(key)
                    chosen.append(r)
            else:
                fam["wrong" if r["label"] == "wrong" else "invalid"] += 1
                key = json.dumps([r.get("fix"), r.get("reason")], sort_keys=True)
                if key not in seen_r and r.get("fix") is not None:
                    seen_r.add(key)
                    rejected.append(r)
        meta = {"case_id": case_id, "variant": cp["variant"], "family": cp["family"],
                "prompt_version": cp["prompt_version"], "prompt_sha256": cp["prompt_sha256"]}
        for r in chosen:
            sft.append(meta | {"seed": r["seed"], "match": r["match"],
                               "messages": cp["messages"] + [{"role": "assistant", "content": _assistant(r)}]})
        pairs = [(chosen[n % len(chosen)], r) for n, r in enumerate(rejected)] if chosen else []
        for c, r in pairs:
            dpo.append(meta | {"prompt": cp["messages"], "chosen": _assistant(c), "rejected": _assistant(r),
                               "chosen_seed": c["seed"], "rejected_seed": r["seed"], "rejected_label": r["label"]})
        fam["sft_examples"] += len(chosen)
        fam["dpo_pairs"] += len(pairs)
    return sft, dpo, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
