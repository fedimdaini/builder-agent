"""Recorded fault cases (tests/faults/<name>/case.json) keep one shape, for the incident memory."""

import json
from pathlib import Path

import pytest

from builder_agent.sandbox import STAGES

FAULTS = sorted((Path(__file__).parent / "faults").glob("*/case.json"))
REQUIRED = {"id", "name", "title", "found_by", "stage", "command", "error_signature", "error_tail",
            "cause", "fix", "static_check", "reproduce"}
# where a fault shows up: a sandbox stage, or the target project's own stack failing to start
FAULT_STAGES = STAGES + ["compose-up"]


def test_there_are_fault_cases():
    assert FAULTS


@pytest.mark.parametrize("path", FAULTS, ids=lambda p: p.parent.name)
def test_fault_case_shape(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    assert REQUIRED <= set(case), REQUIRED - set(case)
    assert case["name"] == path.parent.name
    assert case["stage"] in FAULT_STAGES
    assert 0 < len(case["error_tail"]) <= 50
    assert any(case["error_signature"] in line for line in case["error_tail"])
    assert case["cause"]["summary"] and case["fix"]["summary"] and case["reproduce"]["command"]
    assert isinstance(case["static_check"]["catchable_before_run"], bool)
    assert case["static_check"]["rule"] and case["static_check"]["evidence"]


def test_fault_ids_are_unique_and_sequential():
    ids = sorted(json.loads(p.read_text(encoding="utf-8"))["id"] for p in FAULTS)
    assert ids == [f"fault-{i:03d}" for i in range(1, len(ids) + 1)]
