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


# --- generated cases (builder_agent/faults): same shape, plus their family, injection and verified fix ---

GENERATED = sorted((Path(__file__).parent / "faults" / "generated").glob("*/case.json"))


@pytest.mark.parametrize("path", GENERATED, ids=lambda p: p.parent.name)
def test_generated_case_shape(path):
    from builder_agent.faults import CATALOG, FAMILIES
    from builder_agent.fix.models import Diagnosis

    test_fault_case_shape(path)
    case = json.loads(path.read_text(encoding="utf-8"))
    assert case["generated"] and case["family"] in FAMILIES and case["variant"] in CATALOG
    assert case["split"] in (None, "memory", "test")
    Diagnosis.model_validate({"fix": case["expected_fix"], "reason": "x"})     # one menu action
    assert case["expected_fix"] == CATALOG[case["variant"]].expected_fix
    assert case["fix"]["verified_by"]["ok"]


def test_generated_ids_are_unique_and_sequential():
    ids = sorted(json.loads(p.read_text(encoding="utf-8"))["id"] for p in GENERATED)
    assert ids == [f"gen-{i:03d}" for i in range(1, len(ids) + 1)]


def test_generated_split_by_variant():
    """Each variant is in exactly one split (memory or test), and every family has memory cases."""
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in GENERATED]
    variants = [c["variant"] for c in cases]
    assert len(variants) == len(set(variants))
    assert all(c["split"] in ("memory", "test") for c in cases)
    families = {c["family"] for c in cases}
    assert families == {c["family"] for c in cases if c["split"] == "memory"}
