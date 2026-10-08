"""python -m builder_agent.verify <repo> [--ref HEAD | --working-tree] [--out DIR]"""

import argparse
import sys
from pathlib import Path

from . import verify_repo

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.verify", description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--ref", default="HEAD", help="git commit to verify (default HEAD, not the working tree)")
    parser.add_argument("--working-tree", action="store_true", help="verify the working tree instead of a commit")
    parser.add_argument("--out", help="output folder (default experiments/verify/<repo>/ in builder-agent)")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out) if args.out else ROOT / "experiments" / "verify" / repo.name
    report = verify_repo(repo, ref=None if args.working_tree else args.ref, out_dir=out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{report.repo} ({report.source}): {report.errors} error(s), {report.warnings} warning(s)")
    for i, f in enumerate(report.findings, 1):
        print(f"  {i:>2}. {f.severity:<7} {f.file}:{f.line or '-'}  {f.rule} ({f.fault_id})")
    print(f"patch: {report.patch_check}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
