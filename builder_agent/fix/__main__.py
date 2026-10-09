"""python -m builder_agent.fix <repo> --slots S --prompt diagnose_v1 --model M [--expected E] [--mlflow-client mlflow]
    [--fault tests/faults/<name>/case.json]   # score the first fix against the case's expected_fix
    [--case tests/faults/generated/<variant>/case.json]   # start from a generated fault, score against it
    [--out experiments/fixes/<run>]           # copy this run's logs and summary there
    [--retriever naive|advanced]              # RAG over the fix memory (prompts with {{ retrieved_text }})"""

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
    p.add_argument("--model", required=True)
    p.add_argument("--expected", help="expected prediction JSON for the smoke test")
    p.add_argument("--mlflow-client", help='fault injection, e.g. "mlflow" (unpinned) to reproduce fault-001')
    p.add_argument("--fault", help="fault case.json with an expected_fix to score against")
    p.add_argument("--case", help="generated fault case.json: inject its fault, score against its expected_fix")
    p.add_argument("--out", help="folder for this run's logs and summary")
    p.add_argument("--retriever", choices=["naive", "advanced"], help="RAG retriever over the fix memory")
    p.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model for --retriever")
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
    retrieve = None
    if args.retriever:
        from ..llm.ollama import OllamaEmbedder
        from ..memory import FixMemory, retriever
        retrieve = retriever(FixMemory.build(OllamaEmbedder(args.embed_model)), args.retriever)
    result = run_fix_loop(args.repo, slots, args.prompt, OllamaClient(args.model), expected=expected,
                          mlflow_client=mlflow_client, injection=injection, retriever=retrieve,
                          retriever_name=args.retriever)
    sys.stdout.reconfigure(encoding="utf-8")
    text = result.text()
    score = score_fix(result, case["expected_fix"]) | {"case": case["id"]} if case else None
    if case:
        text += f"\nscore vs {case['id']} ({case.get('variant', case['name'])}): " + json.dumps(score)
    print(text)
    if args.out:
        print("logs ->", export_run(result, args.out, text, DEFAULT_FIX_LOG, DEFAULT_LOG, score))
    sys.exit(0 if result.final_ok else 1)


if __name__ == "__main__":
    main()
