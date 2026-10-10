"""python -m builder_agent.fix <repo> --slots S --prompt diagnose_v1 --model M [--expected E] [--mlflow-client mlflow]
    [--fault tests/faults/<name>/case.json]   # score the first fix against the case's expected_fix
    [--case tests/faults/generated/<variant>/case.json]   # start from a generated fault, score against it
    [--out experiments/fixes/<run>]           # copy this run's logs and summary there
    [--retriever naive|advanced]              # RAG over the fix memory (prompts with {{ retrieved_text }})
    [--memory base|learned]                   # base: the memory of every RESULTS.md run (default)
    [--memory-snapshot SHA256]                # evaluation: refuse unless the memory has this hash; no write-back
    [--no-write-back]                         # don't add this run's checked fixes to memory/learned/"""

import argparse
import json
import sys
from pathlib import Path

from ..llm import OllamaClient
from ..render.configs import Overrides
from ..render.slots import load_answer
from . import DEFAULT_FIX_LOG, export_run, run_fix_loop, score_fix
from ..sandbox import DEFAULT_LOG


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.fix", description=__doc__)
    p.add_argument("repo")
    p.add_argument("--slots", required=True)
    p.add_argument("--prompt", required=True, help="prompt version, e.g. diagnose_v1")
    p.add_argument("--model", default="qwen2.5-coder:7b",
                   help="Ollama model name, e.g. qwen-builder-sft (the QLoRA model from notebooks/qlora_sft.ipynb)")
    p.add_argument("--expected", help="expected prediction JSON for the smoke test")
    p.add_argument("--mlflow-client", help='fault injection, e.g. "mlflow" (unpinned) to reproduce fault-001')
    p.add_argument("--fault", help="fault case.json with an expected_fix to score against")
    p.add_argument("--case", help="generated fault case.json: inject its fault, score against its expected_fix")
    p.add_argument("--out", help="folder for this run's logs and summary")
    p.add_argument("--retriever", choices=["naive", "advanced"], help="RAG retriever over the fix memory")
    p.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model for --retriever")
    p.add_argument("--memory", choices=["base", "learned"], default="base",
                   help="base: fault-001 + generated memory faults (every RESULTS.md run); learned: + passed fixes "
                        "written back from real runs")
    p.add_argument("--memory-snapshot", metavar="SHA256",
                   help="evaluation mode: refuse to run unless the memory has this snapshot hash; never writes back")
    p.add_argument("--no-write-back", action="store_true", help="don't write this run's checked fixes to memory")
    args = p.parse_args()

    slots, _ = load_answer(args.slots)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    mlflow_client, injection, case = args.mlflow_client, None, None
    if args.fault or args.case:
        case = json.loads(Path(args.fault or args.case).read_text(encoding="utf-8"))
    if args.case:
        fault = dict(case["injection"])
        mlflow_client = fault.pop("mlflow_client", mlflow_client)
        injection = Overrides(**fault)
    from ..memory.learned import check_snapshot, memory_docs, snapshot_sha256, write_back
    docs = memory_docs(args.memory)
    memory_sha = snapshot_sha256(docs)
    if args.memory_snapshot:
        check_snapshot(docs, args.memory_snapshot)    # raises SnapshotMismatch: the run doesn't start
    retrieve = None
    if args.retriever:
        from ..llm.ollama import OllamaEmbedder
        from ..memory import FixMemory, retriever
        retrieve = retriever(FixMemory.build(OllamaEmbedder(args.embed_model), docs=docs), args.retriever)
    result = run_fix_loop(args.repo, slots, args.prompt, OllamaClient(args.model), expected=expected,
                          mlflow_client=mlflow_client, injection=injection, retriever=retrieve,
                          retriever_name=args.retriever)
    if args.retriever:
        result.memory_set, result.memory_sha256 = args.memory, memory_sha
    sys.stdout.reconfigure(encoding="utf-8")
    text = result.text()
    score = score_fix(result, case["expected_fix"]) | {"case": case["id"]} if case else None
    if case:
        text += f"\nscore vs {case['id']} ({case.get('variant', case['name'])}): " + json.dumps(score)
    if args.retriever:
        text += f"\nmemory: {args.memory} {memory_sha[:12]} ({len(docs)} documents)"
    if args.no_write_back:
        text += "\nwrite-back: off (--no-write-back)"
    else:
        written, why = write_back(result, case, evaluation=bool(args.memory_snapshot), memory_set=args.memory,
                                  memory_sha256=memory_sha)
        text += f"\nwrite-back: none, {why}" if why else f"\nwrite-back: {len(written)} document(s) to memory/learned/"
    print(text)
    if args.out:
        print("logs ->", export_run(result, args.out, text, DEFAULT_FIX_LOG, DEFAULT_LOG, score))
    sys.exit(0 if result.final_ok else 1)


if __name__ == "__main__":
    main()
