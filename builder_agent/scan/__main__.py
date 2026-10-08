"""python -m builder_agent.scan <repo> [--json]"""

import argparse
import sys

from . import scan_repo


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.scan", description=__doc__)
    parser.add_argument("repo", help="path to the repository to scan")
    parser.add_argument("--json", action="store_true", help="print the full RepoContext as JSON")
    args = parser.parse_args()

    ctx = scan_repo(args.repo)
    sys.stdout.reconfigure(encoding="utf-8")
    print(ctx.model_dump_json(indent=2) if args.json else ctx.summary())


if __name__ == "__main__":
    main()
