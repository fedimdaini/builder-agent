"""python -m builder_agent.sandbox <repo> --slots slots.json [--expected expected.json] [--keep]"""

import argparse
import json
import sys
from pathlib import Path

from ..render.slots import load_answer
from . import DEFAULT_CONTRACTS, DEFAULT_LOG, run_sandbox


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.sandbox", description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--slots", required=True, help="slot answers JSON")
    parser.add_argument("--expected", help="expected prediction JSON for the smoke test")
    parser.add_argument("--contracts", default=DEFAULT_CONTRACTS)
    parser.add_argument("--log", default=DEFAULT_LOG, help="attempts log (JSON lines)")
    parser.add_argument("--keep", action="store_true", help="keep containers and the temp copy for debugging")
    parser.add_argument("--mlflow-client", help='override the MLflow client install spec, e.g. "mlflow" '
                                                "(unpinned) to reproduce fault-001")
    args = parser.parse_args()

    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    slots, _ = load_answer(args.slots)          # gold files may carry "_alternatives"
    result = run_sandbox(args.repo, slots=slots,
                         expected=load(args.expected) if args.expected else None,
                         contracts_path=args.contracts, log_path=args.log, keep=args.keep,
                         mlflow_client=args.mlflow_client)
    sys.stdout.reconfigure(encoding="utf-8")
    print(result.summary())
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
