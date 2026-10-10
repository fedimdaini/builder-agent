"""Rationalization (STaR, Zelikman et al. 2022): reasons for the verified fix, where sampling found none.

For a memory fault, the diagnose_v3 prompt gets a hint at the end of the user message: the verified
fix, and a request for a short reason based on the current error output. 4 answers are sampled at
temperature 0.7; an answer is kept if it keeps the hinted fix and its reason passes the faithfulness
filter. If none passes, a reason is written from the fault's recorded error signature and cause.
The training record always uses the prompt WITHOUT the hint.

Each answer's source: "sampled" (from builder_agent.finetune sampling, no hint), "rationalized" (hinted)
or "written" (from the recorded case).
"""

from __future__ import annotations

import json
import re

from ..fix.models import Diagnosis, describe
from . import CasePrompt, faithful, sample_case

RATIONALIZE_SAMPLES = 4
# a reason that mentions the hint ("the verified fix is ...") makes no sense without it in the training prompt
HINT_LEAK_RE = re.compile(r"\bverified\b|\bhint(ed)?\b", re.I)
HINT_MARK = "HINT (for this answer only):"
HINT = (HINT_MARK + " the verified fix for this error is `{fix}`. Answer with exactly this fix. In \"reason\", "
        "explain in one or two sentences what in the ERROR OUTPUT above causes the failure and why this fix "
        "addresses it. Base the reason on the current error output, not on the past failures from memory.")


def expected_fix(case: dict):
    return Diagnosis.model_validate({"fix": case["expected_fix"], "reason": "x"}).fix


def hinted(cp: CasePrompt, case: dict) -> CasePrompt:
    """The same prompt with the hint at the end of the user message."""
    system, user = cp.messages
    hint = HINT.format(fix=describe(expected_fix(case)))
    return cp.model_copy(update={"messages": [system, {"role": "user", "content": user["content"] + "\n\n" + hint}]})


def judge(row: dict, case: dict, cp: CasePrompt) -> dict:
    """A hinted sample: kept ("rationalized") if it keeps the fix and its reason is faithful."""
    exp = expected_fix(case)
    out = {"source": "rationalized", "fix": case["expected_fix"], "fix_text": describe(exp), "reason": None,
           "kept": False, "why": ""}
    try:
        answer = json.loads(row["response"] or "")
        d = Diagnosis.model_validate(answer)
    except Exception as e:  # noqa: BLE001 - recorded as not kept
        return out | {"why": f"not a valid answer: {type(e).__name__}"}
    if d.fix.model_dump() != exp.model_dump():
        return out | {"why": f"changed the fix to {describe(d.fix)}", "reason": d.reason}
    leak = HINT_LEAK_RE.search(d.reason)
    if leak:
        return out | {"reason": d.reason, "why": f"refers to the hint ({leak.group(0)!r}), which the training "
                                                   "prompt doesn't have"}
    ok, why = faithful(d.reason, case, cp.retrieved, describe(exp), cp.current_python)
    return out | {"reason": d.reason, "kept": ok, "why": why}


def rationalize_case(cp: CasePrompt, case: dict, client_factory, n: int = RATIONALIZE_SAMPLES) -> list[dict]:
    """n hinted samples, each logged (as sample_case does) and judged."""
    rows = sample_case(hinted(cp, case), client_factory, n)
    return [r | {"hinted": True} | judge(r, case, cp) for r in rows]


def _short(signature: str, limit: int = 160) -> str:
    sig = re.split(r"\s+Response body:|\s+\(Caused by", signature)[0].strip()
    return sig if len(sig) <= limit else sig[:limit].rsplit(" ", 1)[0] + " ..."


def written_reason(case: dict) -> str:
    """Two sentences from the recorded case: the error line, then the first sentence of its cause."""
    cause = re.split(r"(?<=\.)\s", case["cause"]["summary"].strip(), maxsplit=1)[0]
    return f"The {case['stage']} stage fails with: {_short(case['error_signature'])}. {cause}"


def written_row(case: dict, cp: CasePrompt) -> dict:
    exp = expected_fix(case)
    reason = written_reason(case)
    ok, why = faithful(reason, case, cp.retrieved, describe(exp), cp.current_python)
    return {"case_id": case["id"], "variant": case["variant"], "family": case["family"], "seed": None,
            "source": "written", "fix": case["expected_fix"], "fix_text": describe(exp), "reason": reason,
            "kept": ok, "why": why}


def training_rows(rationalized: list[dict], cases: dict[str, dict], prompts: dict[str, CasePrompt]) -> list[dict]:
    """Rows for build_datasets: the kept rationalized answers per fault, or one written answer if none was kept."""
    out = []
    for case_id, case in cases.items():
        kept = [r for r in rationalized if r["case_id"] == case_id and r["kept"]]
        if not kept:
            kept = [written_row(case, prompts[case_id])]
        for r in kept:
            out.append({"case_id": case_id, "seed": r["seed"], "label": "correct", "match": "exact",
                        "source": r["source"], "fix": r["fix"], "fix_text": r["fix_text"], "reason": r["reason"],
                        "faithful": r["kept"], "faithful_why": r["why"]})
    return out
