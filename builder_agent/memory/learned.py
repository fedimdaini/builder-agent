"""Fix memory that learns from real runs, with frozen snapshots for evaluations.

After a fix-loop run, every fix the sandbox checked becomes one JSON document in memory/learned/
(builder-agent, never the target repo): the failed stage, the error signature it addressed, the fix,
its outcome (passed, failed, reverted) and where it came from (repo, date, model, prompt, sandbox
attempt). The model's stated reason is kept for reading but never indexed or shown: reasons can be
copied from retrieved cases (docs/RESULTS.md sections 4-6), the sandbox outcome can't.

Memory sets:
- "base": fault-001 plus the generated faults with split "memory". The memory of every experiment in
  docs/RESULTS.md, unchanged; its snapshot hash is BASE_SHA256.
- "learned": base plus the learned documents whose fix passed.

Never written back: a run started from a split=test fault (held out for evaluation), and any run in
evaluation mode (a frozen snapshot was asked for). Evaluations name the snapshot by its sha256 and
refuse to start if the memory differs.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..faults import error_signature
from . import MemoryDoc, doc_from_case, memory_cases

LEARNED_DIR = Path(os.environ.get("BUILDER_MEMORY_DIR") or Path(__file__).resolve().parents[2] / "memory" / "learned")
MEMORY_SETS = ("base", "learned")
# sha256 of the "base" set when write-back was added (2026-10-10): the memory of every RESULTS.md experiment
BASE_SHA256 = "4aa9a3ad11e5515b27a384826d23a17f1cc328cd43ad9b5d848870d0c0dda708"


class SnapshotMismatch(RuntimeError):
    pass


def learned_documents(directory: Path | None = None) -> list[dict]:
    d = Path(directory or LEARNED_DIR)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def doc_from_learned(rec: dict) -> MemoryDoc:
    return MemoryDoc(id=rec["id"], family=None, variant=f"{rec['source']['repo']} (learned)", stage=rec["stage"],
                     error=rec["error_signature"], cause=rec["cause"], fix=rec["fix"])


def memory_docs(memory_set: str = "base", directory: Path | None = None) -> list[MemoryDoc]:
    if memory_set not in MEMORY_SETS:
        raise ValueError(f"unknown memory set {memory_set!r}: {', '.join(MEMORY_SETS)}")
    docs = [doc_from_case(c) for c in memory_cases()]
    if memory_set == "learned":
        docs += [doc_from_learned(r) for r in learned_documents(directory) if r["outcome"] == "passed"]
    return docs


def snapshot_sha256(docs: list[MemoryDoc]) -> str:
    """Order-independent hash of what retrieval can return: id, stage, error, cause, fix of every document."""
    canon = sorted((d.model_dump() for d in docs), key=lambda d: d["id"])
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_snapshot(docs: list[MemoryDoc], expected_sha256: str) -> None:
    got = snapshot_sha256(docs)
    if got != expected_sha256:
        raise SnapshotMismatch(f"memory snapshot is {got[:12]}, the evaluation asks for {expected_sha256[:12]}: "
                               "the memory changed since the snapshot was taken; refusing to run")


def _outcome(outcome: str) -> str:
    if outcome == "passed":
        return "passed"
    if outcome.startswith("reverted"):
        return "reverted"
    return "failed"


def skip_reason(case: dict | None, evaluation: bool) -> str | None:
    if evaluation:
        return "evaluation run (frozen memory snapshot)"
    if case is not None and case.get("split") == "test":
        return f"{case['id']} is a held-out test fault"
    return None


def write_back(result, case: dict | None = None, evaluation: bool = False, memory_set: str = "base",
               memory_sha256: str | None = None, directory: Path | None = None) -> tuple[list[Path], str | None]:
    """Write one document per sandbox-checked fix of `result` (a fix.FixLoopResult).
    Returns (files written, why nothing was written)."""
    why = skip_reason(case, evaluation)
    if why:
        return [], why
    d = Path(directory or LEARNED_DIR)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for a in result.attempts:
        if not (a.action and a.sandbox_attempt_id):
            continue                                  # no fix applied, so nothing was checked
        rec_id = f"learned-{a.sandbox_attempt_id}"
        path = d / f"{rec_id}.json"
        if path.exists():
            continue
        signature = error_signature(a.error_tail)
        outcome = _outcome(a.outcome)
        rec = {
            "id": rec_id,
            "stage": a.stage,
            "error_signature": signature,
            "fix": a.action,
            "fix_text": a.action_text,
            "outcome": outcome,
            "outcome_detail": a.outcome,
            # a fact, not the model's words: what failed and what the sandbox showed after the fix
            "cause": f"stage {a.stage} failed with this error; {a.action_text} was applied and the sandbox "
                     f"run after it {'passed' if outcome == 'passed' else a.outcome}",
            "model_reason": a.reason,                 # kept for reading; never indexed or shown in prompts
            "source": {"repo": result.repo, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "model": result.model, "model_digest": result.model_digest,
                       "prompt_version": result.prompt_version, "retriever": result.retriever,
                       "memory_set": memory_set, "memory_sha256": memory_sha256,
                       "sandbox_attempt_id": a.sandbox_attempt_id, "fix_n": a.n,
                       "injected_case": case["id"] if case else None},
        }
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written, None
