"""Fix memory for RAG: past failures with their verified fixes, retrieved for the diagnosis prompt.

    from builder_agent.memory import FixMemory
    memory = FixMemory.build(OllamaEmbedder())
    hits = memory.advanced(stage, error_tail)          # or memory.naive(error_tail)
    text = render_retrieved(hits)                      # -> {{ retrieved_text }} in prompts/diagnose_v3.md

Documents: the generated cases with split "memory" plus fault-001 (tests/faults/). A case with
split "test" is never indexed (MemoryCaseError). Each document holds the failed stage, the error
signature, the cause and the verified expected fix. The index lives in memory (Chroma's ephemeral
client, embeddings computed here through Ollama); nothing is written to disk.

Two retrievers, top 3:
- naive: the full error tail as the query, vectors only;
- advanced: the error signature as the query, BM25 + vectors merged with reciprocal rank fusion,
  plus a boost (not a filter) for documents from the same stage.
"""

from __future__ import annotations

import json
import uuid
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from ..faults import error_signature
from ..fix.models import Diagnosis, describe

FAULTS_DIR = Path(__file__).resolve().parents[2] / "tests" / "faults"
TOP_K = 3
RRF_K = 60                              # reciprocal rank fusion: score = sum of 1 / (RRF_K + rank)
STAGE_BOOST = 0.5 / (RRF_K + 1)         # same stage: worth half a first place in one ranking


class MemoryCaseError(ValueError):
    pass


class MemoryDoc(BaseModel):
    id: str                 # fault-001, gen-002, ...
    family: str | None      # generated cases only
    variant: str
    stage: str
    error: str              # the error signature
    cause: str
    fix: dict               # the verified expected fix (a menu action)

    @property
    def fix_text(self) -> str:
        return describe(Diagnosis.model_validate({"fix": self.fix, "reason": "x"}).fix)

    def text(self) -> str:
        return f"stage: {self.stage}\nerror: {self.error}\ncause: {self.cause}\nfix: {self.fix_text}"


class Hit(BaseModel):
    doc: MemoryDoc
    score: float
    ranks: dict[str, int | None]        # rank in each list (1 = best), None if not ranked


def doc_from_case(case: dict) -> MemoryDoc:
    if case.get("split") == "test":
        raise MemoryCaseError(f"{case['id']} ({case.get('variant')}) is a held-out test case; "
                              "test cases are never indexed")
    if "expected_fix" not in case:
        raise MemoryCaseError(f"{case['id']} has no expected_fix")
    return MemoryDoc(id=case["id"], family=case.get("family"), variant=case.get("variant", case["name"]),
                     stage=case["stage"], error=case["error_signature"], cause=case["cause"]["summary"],
                     fix=case["expected_fix"])


def memory_cases(faults_dir: str | Path = FAULTS_DIR) -> list[dict]:
    """fault-001 plus the generated cases with split "memory"."""
    faults_dir = Path(faults_dir)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    hand = [c for c in map(load, sorted(faults_dir.glob("*/case.json"))) if c["id"] == "fault-001"]
    generated = [c for c in map(load, sorted((faults_dir / "generated").glob("*/case.json")))
                 if c.get("split") == "memory"]
    return hand + generated


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", text.lower())


class FixMemory:
    def __init__(self, docs: list[MemoryDoc], embedder):
        self.docs = {d.id: d for d in docs}
        self.embedder = embedder
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        name = f"fix_memory_{uuid.uuid4().hex}"       # unique: the ephemeral client is shared per process
        self.collection = client.get_or_create_collection(name, embedding_function=None,
                                                          metadata={"hnsw:space": "cosine"})
        self.collection.add(ids=[d.id for d in docs], documents=[d.text() for d in docs],
                            embeddings=embedder.embed([d.text() for d in docs], kind="document"),
                            metadatas=[{"stage": d.stage, "variant": d.variant} for d in docs])
        self._order = [d.id for d in docs]
        self.bm25 = BM25Okapi([tokens(d.text()) for d in docs])

    @classmethod
    def build(cls, embedder, cases: list[dict] | None = None, docs: list[MemoryDoc] | None = None) -> FixMemory:
        """docs: ready documents (memory/learned.py memory_docs); else cases, else the base memory cases."""
        if docs is not None:
            return cls(docs, embedder)
        return cls([doc_from_case(c) for c in (memory_cases() if cases is None else cases)], embedder)

    def _vector_ranking(self, query: str) -> list[tuple[str, float]]:
        r = self.collection.query(query_embeddings=self.embedder.embed([query], kind="query"),
                                  n_results=len(self.docs))
        return [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]     # cosine similarity

    def naive(self, error_tail: list[str] | str, k: int = TOP_K) -> list[Hit]:
        query = error_tail if isinstance(error_tail, str) else "\n".join(error_tail)
        ranked = self._vector_ranking(query)
        return [Hit(doc=self.docs[i], score=round(s, 4), ranks={"vector": n})
                for n, (i, s) in enumerate(ranked[:k], 1)]

    def advanced(self, stage: str, error_tail: list[str] | str, k: int = TOP_K) -> list[Hit]:
        signature = error_signature(error_tail if isinstance(error_tail, list) else error_tail.splitlines())
        vector = [i for i, _ in self._vector_ranking(signature)]
        bm = self.bm25.get_scores(tokens(signature))
        bm25 = [i for _, i in sorted(zip(bm, self._order), key=lambda t: -t[0])]
        hits = []
        for i, doc in self.docs.items():
            ranks = {"vector": vector.index(i) + 1, "bm25": bm25.index(i) + 1}
            score = sum(1 / (RRF_K + r) for r in ranks.values()) + (STAGE_BOOST if doc.stage == stage else 0)
            hits.append(Hit(doc=doc, score=round(score, 5), ranks=ranks))
        return sorted(hits, key=lambda h: -h.score)[:k]


def render_retrieved(hits: list[Hit]) -> str:
    """The text for {{ retrieved_text }}: each case with its stage, error line, cause and fix."""
    if not hits:
        return "none"
    return "\n".join(f"{n}. stage: {h.doc.stage}\n   error: {h.doc.error}\n   cause: {h.doc.cause}\n"
                     f"   verified fix: {h.doc.fix_text}" for n, h in enumerate(hits, 1))


def retriever(memory: FixMemory, kind: str):
    """A function (stage, error_tail) -> hits for the fix loop."""
    if kind == "naive":
        return lambda stage, tail: memory.naive(tail)
    if kind == "advanced":
        return lambda stage, tail: memory.advanced(stage, tail)
    raise ValueError(f"unknown retriever {kind!r}: naive or advanced")
