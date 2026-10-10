"""Hand-written reasons ("human" source) for the fine-tuning data.

experiments/finetune/human_reasons.json maps a memory case id to a list of {"fix": <the verified fix>,
"reason": "..."}. Keys starting with "_" are notes. An empty reason is a placeholder and is skipped.
Each reason must carry the case's verified fix and pass the same faithfulness filter as the model's
answers; one that doesn't is reported and left out.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..fix.models import Diagnosis, describe
from . import CasePrompt, faithful

TEMPLATE = {
    "_how": ("case_id -> list of {fix, reason}. fix must be the case's verified fix (tests/faults/generated/"
             "<variant>/case.json expected_fix). reason: one or two sentences from the case's error output. "
             "Empty reasons are skipped. Only split=memory cases."),
    "gen-004": [{"fix": {"action": "set_python_version", "version": "3.9"}, "reason": ""}],
}


def load_human(path: Path, cases: dict[str, dict], prompts: dict[str, CasePrompt]) -> tuple[list[dict], list[str]]:
    """Rows for build_datasets (source "human"), and the problems with entries left out."""
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows, problems = [], []
    for case_id, entries in data.items():
        if case_id.startswith("_"):
            continue
        if case_id not in cases:
            problems.append(f"{case_id}: not a memory case (only split=memory cases make training data)")
            continue
        expected = Diagnosis.model_validate({"fix": cases[case_id]["expected_fix"], "reason": "x"}).fix
        for n, entry in enumerate(entries, 1):
            reason = (entry.get("reason") or "").strip()
            if not reason:
                continue
            if entry.get("fix") != expected.model_dump():
                problems.append(f"{case_id} #{n}: fix {entry.get('fix')} is not the verified fix "
                                f"{expected.model_dump()}")
                continue
            ok, why = faithful(reason, cases[case_id], prompts[case_id].retrieved, describe(expected),
                               prompts[case_id].current_python)
            if not ok:
                problems.append(f"{case_id} #{n}: not faithful: {why}")
                continue
            rows.append({"case_id": case_id, "seed": None, "label": "correct", "match": "exact", "source": "human",
                         "fix": expected.model_dump(), "fix_text": describe(expected), "reason": reason,
                         "faithful": True, "faithful_why": why})
    return rows, problems
