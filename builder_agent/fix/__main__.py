"""python -m builder_agent.fix <repo> --slots S --prompt diagnose_v1 --model M [--expected E] [--mlflow-client mlflow]
    [--fault tests/faults/<name>/case.json]   # score the first fix against the case's expected_fix"""

import argparse
import json
import sys
from pathlib import Path

from ..llm import OllamaClient
from ..render.slots import load_answer
from . import run_fix_loop, score_fix


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.fix", description=__doc__)
    p.add_argument("repo")
    p.add_argument("--slots", required=True)
    p.add_argument("--prompt", required=True, help="prompt version, e.g. diagnose_v1")
    p.add_argument("--model", required=True)
    p.add_argument("--expected", help="expected prediction JSON for the smoke test")
    p.add_argument("--mlflow-client", help='fault injection, e.g. "mlflow" (unpinned) to reproduce fault-001')
    p.add_argument("--fault", help="fault case.json with an expected_fix to score against")
    args = p.parse_args()

    slots, _ = load_answer(args.slots)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    result = run_fix_loop(args.repo, slots, args.prompt, OllamaClient(args.model), expected=expected,
                          mlflow_client=args.mlflow_client)
    sys.stdout.reconfigure(encoding="utf-8")
    print(result.text())
    if args.fault:
        case = json.loads(Path(args.fault).read_text(encoding="utf-8"))
        print("score vs", case["id"], score_fix(result, case["expected_fix"]))
    sys.exit(0 if result.final_ok else 1)


if __name__ == "__main__":
    main()
