"""python -m builder_agent.memory [--embed-model nomic-embed-text] [--json]

Retrieval only (no LLM, no sandbox): for each held-out test case (split "test"), the top 3 of
both retrievers, queried with the case's recorded error tail and stage, and whether a case of the
same family is among them."""

import argparse
import json
import sys

from ..llm.ollama import OllamaEmbedder
from . import FAULTS_DIR, FixMemory


def evaluate(memory: FixMemory) -> list[dict]:
    rows = []
    for p in sorted((FAULTS_DIR / "generated").glob("*/case.json")):
        case = json.loads(p.read_text(encoding="utf-8"))
        if case["split"] != "test":
            continue
        for kind, hits in (("naive", memory.naive(case["error_tail"])),
                           ("advanced", memory.advanced(case["stage"], case["error_tail"]))):
            top = [{"id": h.doc.id, "variant": h.doc.variant, "family": h.doc.family or "version_mismatch",
                    "stage": h.doc.stage, "score": h.score, "ranks": h.ranks} for h in hits]
            rows.append({"case": case["id"], "variant": case["variant"], "family": case["family"],
                         "stage": case["stage"], "retriever": kind, "top3": top,
                         "same_family": any(t["family"] == case["family"] for t in top),
                         "same_family_rank": next((n for n, t in enumerate(top, 1)
                                                   if t["family"] == case["family"]), None)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.memory", description=__doc__)
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    embedder = OllamaEmbedder(args.embed_model)
    rows = evaluate(FixMemory.build(embedder))
    if args.json:
        print(json.dumps({"embed_model": args.embed_model, "rows": rows}, indent=2))
        return
    print("| Test case | Family | Retriever | Top 3 (id variant) | Same family in top 3 |")
    print("|---|---|---|---|---|")
    for r in rows:
        top = "; ".join(f"{t['id']} {t['variant']}" for t in r["top3"])
        hit = f"yes (rank {r['same_family_rank']})" if r["same_family"] else "no"
        print(f"| {r['case']} {r['variant']} | {r['family']} | {r['retriever']} | {top} | {hit} |")
    for kind in ("naive", "advanced"):
        n = [r for r in rows if r["retriever"] == kind]
        print(f"{kind}: same family in top 3 for {sum(r['same_family'] for r in n)}/{len(n)}")


if __name__ == "__main__":
    main()
