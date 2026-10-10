"""Memory that learns from real runs, and frozen snapshots for evaluations (step 3)."""

import json

import pytest

from builder_agent.fix import FixAttempt, FixLoopResult
from builder_agent.memory.learned import (BASE_SHA256, SnapshotMismatch, check_snapshot, learned_documents,
                                          memory_docs, snapshot_sha256, write_back)
from builder_agent.render.configs import Overrides

TAIL = ["Traceback (most recent call last):", "ImportError: cannot import name 'url_quote' from 'werkzeug.urls'"]


def attempt(n, outcome, sandbox_id, action=True):
    return FixAttempt(n=n, stage="health", error_tail=TAIL, calls=[],
                      action={"action": "pin_package", "name": "werkzeug", "version": f"2.{n}.0"} if action else None,
                      action_text=f"pin_package werkzeug==2.{n}.0" if action else None,
                      reason="Werkzeug 3 removed url_quote." if action else None,
                      outcome=outcome, sandbox_attempt_id=sandbox_id if action else None)


def result(attempts):
    return FixLoopResult(repo="some-repo", prompt_version="diagnose_v3", prompt_sha256="x", model="m",
                         model_digest="d", initial="failed at health", initial_attempt_id="a0", attempts=attempts,
                         final_ok=True, stopped_because="passed", applied=Overrides(), retriever="advanced")


def test_base_memory_is_the_memory_of_the_past_experiments():
    docs = memory_docs("base")
    assert snapshot_sha256(docs) == BASE_SHA256 and len(docs) == 10
    assert not any(d.id.startswith("learned-") for d in docs)


def test_write_back_stores_every_checked_fix_with_source_and_outcome(tmp_path):
    r = result([attempt(1, "failed at health", "s1"), attempt(2, "not applied: rejected", None, action=False),
                attempt(3, "passed", "s3")])
    written, why = write_back(r, case=None, memory_set="base", memory_sha256=BASE_SHA256, directory=tmp_path)
    assert why is None and [p.name for p in written] == ["learned-s1.json", "learned-s3.json"]
    recs = {d["id"]: d for d in learned_documents(tmp_path)}
    assert recs["learned-s1"]["outcome"] == "failed" and recs["learned-s3"]["outcome"] == "passed"
    src = recs["learned-s3"]["source"]
    assert (src["repo"], src["memory_sha256"], src["sandbox_attempt_id"]) == ("some-repo", BASE_SHA256, "s3")
    assert "url_quote" in recs["learned-s3"]["error_signature"]
    # the model's reason is kept but is not what retrieval sees
    assert recs["learned-s3"]["model_reason"] and "Werkzeug 3" not in recs["learned-s3"]["cause"]
    # a second write of the same run adds nothing
    assert write_back(r, directory=tmp_path)[0] == []


@pytest.mark.parametrize("case,evaluation,why", [
    ({"id": "gen-011", "split": "test"}, False, "held-out test fault"),
    (None, True, "evaluation run"),
    ({"id": "gen-004", "split": "memory"}, True, "evaluation run"),
])
def test_never_written_back(tmp_path, case, evaluation, why):
    written, reason = write_back(result([attempt(1, "passed", "s1")]), case=case, evaluation=evaluation,
                                 directory=tmp_path)
    assert written == [] and why in reason and not list(tmp_path.iterdir())


def test_learned_set_adds_only_passed_fixes_and_changes_the_hash(tmp_path):
    write_back(result([attempt(1, "failed at health", "s1"), attempt(2, "passed", "s2")]), directory=tmp_path)
    base, learned = memory_docs("base", tmp_path), memory_docs("learned", tmp_path)
    assert [d.id for d in learned if d.id not in {b.id for b in base}] == ["learned-s2"]
    assert snapshot_sha256(learned) != snapshot_sha256(base)
    assert snapshot_sha256(learned) == snapshot_sha256(list(reversed(learned)))       # order doesn't matter
    assert memory_docs("base", tmp_path) == base                                      # base never includes them


def test_an_evaluation_refuses_a_changed_memory(tmp_path):
    check_snapshot(memory_docs("base", tmp_path), BASE_SHA256)
    write_back(result([attempt(1, "passed", "s1")]), directory=tmp_path)
    with pytest.raises(SnapshotMismatch, match="refusing to run"):
        check_snapshot(memory_docs("learned", tmp_path), BASE_SHA256)


def test_learned_documents_are_retrievable(tmp_path):
    from builder_agent.memory import FixMemory
    from test_memory import FakeEmbedder
    write_back(result([attempt(1, "passed", "s1")]), directory=tmp_path)
    memory = FixMemory.build(FakeEmbedder(), docs=memory_docs("learned", tmp_path))
    hits = memory.advanced("health", TAIL, k=len(memory.docs))
    assert "learned-s1" in [h.doc.id for h in hits]
    assert json.loads((tmp_path / "learned-s1.json").read_text(encoding="utf-8"))["fix_text"] == "pin_package werkzeug==2.1.0"
