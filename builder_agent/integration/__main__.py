"""python -m builder_agent.integration [--contracts contracts.yaml] [--strict]"""

import argparse
import sys

from ..decide import load_contracts
from ..sandbox import DEFAULT_CONTRACTS
from . import report, unconfirmed


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.integration",
                                description="List the contracts.yaml integration fields not confirmed yet.")
    p.add_argument("--contracts", default=str(DEFAULT_CONTRACTS))
    p.add_argument("--strict", action="store_true", help="exit 1 while any field is unconfirmed")
    args = p.parse_args()
    c = load_contracts(args.contracts)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report(c))
    sys.exit(1 if args.strict and unconfirmed(c) else 0)


if __name__ == "__main__":
    main()
