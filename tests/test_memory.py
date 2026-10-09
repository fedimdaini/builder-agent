"""RAG fix memory: what is indexed (never a held-out test case), the two retrievers, the prompt text.
A fake embedder (hashed bag of words) stands in for Ollama."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from builder_agent.fix import run_fix_loop
from builder_agent.llm.prompts import PromptError
from builder_agent.memory import (FixMemory, MemoryCaseError, doc_from_case, memory_cases, render_retrieved,
                                  retriever, tokens)

from test_fix import PIN, Fault001Runner
from test_llm import FakeModel
from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)

GENERATED = Path(__file__).parent / "faults" / "generated"
TEST_CASES = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(GENERATED.glob("*/case.json"))
              if json.loads(p.read_text(encoding="utf-8"))["split"] == "test"]


class FakeEmbedder:
    def __init__(self, dim=256):
        self.dim, self.calls = dim, []

    def embed(self, texts, kind="document"):
        self.calls.append(kind)
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokens(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="module")
def memory():
    return FixMemory.build(FakeEmbedder())


# --- what is indexed ---------------------------------------------------------------------------------

def test_memory_is_fault_001_plus_the_memory_split(memory):
    memory_variants = {json.loads(p.read_text(encoding="utf-8"))["id"] for p in GENERATED.glob("*/case.json")
                       if json.loads(p.read_text(encoding="utf-8"))["split"] == "memory"}
    assert set(memory.docs) == {"fault-001"} | memory_variants
    assert memory.collection.count() == len(memory.docs) == 10


def test_no_held_out_test_case_is_indexed(memory):
    """Fails if a split=test case (by id or by variant) gets into the index."""
    assert TEST_CASES, "there are held-out test cases"
    indexed = memory.collection.get(include=["metadatas"])
    variants = {m["variant"] for m in indexed["metadatas"]}
    for case in TEST_CASES:
        assert case["id"] not in indexed["ids"] and case["variant"] not in variants, case["id"]


def test_a_test_case_is_refused(memory):
    with pytest.raises(MemoryCaseError, match="held-out test case"):
        doc_from_case(TEST_CASES[0])
    with pytest.raises(MemoryCaseError):
        FixMemory.build(FakeEmbedder(), cases=memory_cases() + TEST_CASES[:1])


def test_documents_hold_stage_error_cause_and_verified_fix(memory):
    text = memory.docs["fault-001"].text()
    assert text.startswith("stage: train\nerror: API request to endpoint /api/2.0/mlflow/logged-models")
    assert "cause: Unpinned MLflow client 3.x" in text and text.endswith("fix: pin_package mlflow==2.17.2")


# --- retrievers ------------------------------------------------------------------------------------------

def test_naive_uses_the_full_tail_and_vectors_only(memory):
    case = TEST_CASES[0]
    hits = memory.naive(case["error_tail"])
    assert len(hits) == 3 and all(set(h.ranks) == {"vector"} for h in hits)
    assert [h.ranks["vector"] for h in hits] == [1, 2, 3]


def test_advanced_fuses_bm25_and_vectors_with_a_stage_boost(memory):
    tail = ["Traceback (most recent call last):", "ModuleNotFoundError: No module named 'xgboost'", "make: *** x"]
    hits = memory.advanced("train", tail)
    assert hits[0].doc.variant == "xgboost_uninstalled" and set(hits[0].ranks) == {"vector", "bm25"}
    # the boost is not a filter: a build-stage case can still be retrieved for a train failure
    build = memory.advanced("build", ["ERROR: No matching distribution found for xgboost==3.0.0"])
    assert build[0].doc.variant == "xgboost_3_0_0"
    every = memory.advanced("train", ["ERROR: No matching distribution found for xgboost==3.0.0"], k=10)
    assert len(every) == 10 and any(h.doc.stage == "build" for h in every)


def test_stage_boost_breaks_a_tie():
    cases = memory_cases()
    a, b = (dict(c) for c in cases[:2])
    a |= {"id": "a", "stage": "train", "error_signature": "same words", "cause": {"summary": "x"}}
    b |= {"id": "b", "stage": "build", "error_signature": "same words", "cause": {"summary": "x"},
          "expected_fix": a["expected_fix"]}
    m = FixMemory.build(FakeEmbedder(), cases=[a, b])
    assert m.advanced("build", ["same words"], k=2)[0].doc.id == "b"
    assert m.advanced("train", ["same words"], k=2)[0].doc.id == "a"


def test_render_retrieved(memory):
    text = render_retrieved(memory.advanced("train", ["ImportError: libffi.so.8: cannot open shared object file"]))
    first = text.splitlines()[:4]
    assert first == ["1. stage: train", "   error: ImportError: libffi.so.8: cannot open shared object file: "
                     "No such file or directory", f"   cause: {first[2][10:]}", "   verified fix: add_system_package libffi8"]
    assert render_retrieved([]) == "none"


# --- in the fix loop ---------------------------------------------------------------------------------------

def test_fix_loop_puts_retrieved_cases_into_diagnose_v3(mini, memory, tmp_path):  # noqa: F811
    model = FakeModel(PIN)
    r = run_fix_loop(mini.root, MINI_GOLD, "diagnose_v3", model, mlflow_client="mlflow", runner=Fault001Runner(),
                     sandbox_log=None, fix_log=tmp_path / "fix.jsonl", retriever=retriever(memory, "advanced"),
                     retriever_name="advanced")
    assert r.final_ok and r.retriever == "advanced"
    [a] = r.attempts
    assert a.retrieved[0]["id"] in {"fault-001", "gen-007", "gen-008"}       # the same 404 on /logged-models
    user = model.sent[0][1]["content"]
    assert "SIMILAR PAST FAILURES FROM MEMORY (most similar first):\n1. stage: train" in user
    assert "verified fix: pin_package mlflow==2.17.2" in user
    entry = json.loads((tmp_path / "fix.jsonl").read_text(encoding="utf-8"))
    assert entry["retrieved"] == a.retrieved


def test_diagnose_v3_without_a_retriever_names_the_missing_placeholder(mini, tmp_path):  # noqa: F811
    with pytest.raises(PromptError, match="retrieved_text"):
        run_fix_loop(mini.root, MINI_GOLD, "diagnose_v3", FakeModel(PIN), mlflow_client="mlflow",
                     runner=Fault001Runner(), sandbox_log=None, fix_log=None)
