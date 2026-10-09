"""python -m builder_agent.faults <repo> --slots slots.json [--expected expected.json] --variant NAME
python -m builder_agent.faults --list"""

import argparse
import json
import sys
from pathlib import Path

from ..render.slots import load_answer
from ..sandbox import DEFAULT_CONTRACTS, DEFAULT_LOG
from . import CATALOG, DEFAULT_OUT, FAMILIES, generate_case


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m builder_agent.faults", description=__doc__)
    parser.add_argument("repo", nargs="?")
    parser.add_argument("--slots", help="slot answers JSON")
    parser.add_argument("--expected", help="expected prediction JSON for the smoke test")
    parser.add_argument("--variant", choices=sorted(CATALOG))
    parser.add_argument("--list", action="store_true", help="list the catalog")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--contracts", default=DEFAULT_CONTRACTS)
    parser.add_argument("--log", default=DEFAULT_LOG, help="sandbox attempts log (JSON lines)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.list:
        for v in CATALOG.values():
            print(f"{v.name:<24} {FAMILIES[v.family]:<38} expected: {json.dumps(v.expected_fix)}")
        return
    if not (args.repo and args.slots and args.variant):
        parser.error("repo, --slots and --variant are required (or --list)")

    slots, _ = load_answer(args.slots)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    command = " ".join(["python -m builder_agent.faults", args.repo, "--slots", args.slots]
                       + (["--expected", args.expected] if args.expected else []) + ["--variant", args.variant])
    r = generate_case(CATALOG[args.variant], args.repo, slots, expected, out=args.out,
                      contracts_path=args.contracts, sandbox_log=args.log, reproduce=command)
    print(r.fault_attempt.summary())
    if r.fix_attempt:
        print(f"--- with the expected fix: {'ALL STAGES OK' if r.fix_attempt.ok else 'FAILED at ' + str(r.fix_attempt.failed_stage)}"
              f" (attempt {r.fix_attempt.attempt_id})")
    print(f"{r.variant}: {r.reason}" + (f" -> {r.saved}" if r.saved else ""))
    if r.case:
        print(f"  stage {r.case['stage']} | signature: {r.case['error_signature']}")
    sys.exit(0 if r.saved else 1)


if __name__ == "__main__":
    main()
