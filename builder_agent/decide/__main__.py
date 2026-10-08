"""python -m builder_agent.decide <repo> [--contracts contracts.yaml] [--json]"""

import argparse
import sys
from pathlib import Path

from ..scan import scan_repo
from . import load_contracts, plan_build

DEFAULT_CONTRACTS = Path(__file__).resolve().parents[2] / "contracts.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.decide", description=__doc__)
    parser.add_argument("repo", help="path to the repository to plan for")
    parser.add_argument("--contracts", default=DEFAULT_CONTRACTS, help="path to contracts.yaml")
    parser.add_argument("--json", action="store_true", help="print the full BuildPlan as JSON")
    args = parser.parse_args()

    plan = plan_build(scan_repo(args.repo), load_contracts(args.contracts))
    sys.stdout.reconfigure(encoding="utf-8")
    print(plan.model_dump_json(indent=2) if args.json else plan.summary())


if __name__ == "__main__":
    main()
