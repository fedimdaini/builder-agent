"""Leave-one-out evaluation runner (docs/EVAL_PROTOCOL.md). Resumable; stops when Docker or Ollama is down.

    python experiments/eval_loo/run.py --pass main --configs v3-advanced            # one config per session
    python experiments/eval_loo/run.py --pass t0                                    # the temperature-0 pass
    python experiments/eval_loo/run.py --pass main --cases gen-002 gen-013 --seeds 101 --out experiments/eval_loo/_smoke
    python experiments/eval_loo/run.py --pass main --list                           # what is done, what is left

One run = one fix loop on one case with one config and one seed, LLM-only (--no-python-rule), with the
case's leave-one-out memory (builder_agent.memory.loo_docs), no write-back. Each run goes to
<out>/<pass>/<config>/<case>/seed-<n>/ (fix_attempts.jsonl, sandbox_attempts.jsonl, result.json). result.json
is written last (via a temp file), so a run without it is incomplete: it is deleted and done again.

A run is only kept if it measured the model: Docker and Ollama answered before and after it, no LLM call
failed, no Docker error is in its output, and the first sandbox run failed at the case's recorded stage
(the fault reproduced). Otherwise the runner deletes the run and stops with the reason (exit 2).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from builder_agent.fix import run_fix_loop  # noqa: E402
from builder_agent.llm import OllamaClient  # noqa: E402
from builder_agent.llm.ollama import DEFAULT_HOST, DEFAULT_OPTIONS  # noqa: E402
from builder_agent.memory import all_cases, loo_docs  # noqa: E402
from builder_agent.memory.learned import snapshot_sha256  # noqa: E402
from builder_agent.render.configs import Overrides  # noqa: E402
from builder_agent.render.slots import load_answer  # noqa: E402

REPO = ROOT.parent / "taxi-trip-regression"
SLOTS = ROOT / "tests" / "gold" / "taxi_slots.json"
EXPECTED = ROOT / "tests" / "gold" / "taxi_expected.json"
OUT = ROOT / "experiments" / "eval_loo"
MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
CONFIGS = {"v1": ("diagnose_v1", None), "v3-advanced": ("diagnose_v3", "advanced"),
           "v5-advanced": ("diagnose_v5", "advanced")}
PASSES = {"main": (0.7, [101, 102, 103, 104, 105]), "t0": (0.0, [42])}
ORDER_SEED = 2026
INFRA_PATTERNS = ("failed to connect to the docker API", "Cannot connect to the Docker daemon", "error during connect",
                  "docker daemon is not running", "dockerDesktopLinuxEngine")


class Stop(Exception):
    """Docker or Ollama is down, or a run didn't measure the model: stop, don't record it."""


def health() -> str | None:
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                           timeout=60)
        if r.returncode != 0:
            return f"Docker is not answering: {(r.stderr or r.stdout).strip()[:200]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"Docker is not answering: {e}"
    try:
        with urllib.request.urlopen(f"{DEFAULT_HOST}/api/tags", timeout=10) as r:
            json.load(r)
    except OSError as e:
        return f"Ollama is not answering at {DEFAULT_HOST}: {e}"
    return None


def fault(case: dict) -> tuple[Overrides | None, str | None]:
    """The injection: fault-001 is an unpinned MLflow client; a generated case carries its own."""
    if "injection" not in case:
        return None, "mlflow"                       # fault-001 (tests/faults/mlflow_client_server_mismatch)
    inj = dict(case["injection"])
    client = inj.pop("mlflow_client", None)
    return Overrides(**inj), client


def run_dir(out: Path, pass_: str, config: str, case_id: str, seed: int) -> Path:
    return out / pass_ / config / case_id / f"seed-{seed}"


def is_done(d: Path) -> bool:
    return (d / "result.json").exists()


def plan(pass_: str, configs: list[str], case_ids: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    """Every (config, case, seed), in a shuffled but fixed order."""
    todo = [(cfg, c, s) for cfg in configs for c in case_ids for s in seeds]
    random.Random(f"{ORDER_SEED}-{pass_}").shuffle(todo)
    return todo


def invalid(result, case: dict) -> str | None:
    """Why a finished run didn't measure the model, or None."""
    if any(c.error for a in result.attempts for c in a.calls):
        return "an LLM call failed (Ollama)"
    tails = list(result.final_error_tail) + [line for a in result.attempts for line in a.error_tail]
    if any(p.lower() in line.lower() for line in tails for p in INFRA_PATTERNS):
        return "a Docker error is in the sandbox output"
    if result.initial != f"failed at {case['stage']}":
        return f"the fault didn't reproduce: first sandbox run {result.initial!r}, the case says failed at {case['stage']}"
    return None


def run_one(d: Path, pass_: str, config: str, case: dict, seed: int, retrievers: dict, model: str,
            run_fix_loop=run_fix_loop, runner=None) -> dict:
    prompt, retriever_name = CONFIGS[config]
    temperature, _ = PASSES[pass_]
    if d.exists():
        shutil.rmtree(d)                            # an incomplete run: start it again
    d.mkdir(parents=True)
    docs = loo_docs(case["id"])
    retrieve = None
    if retriever_name:
        if case["id"] not in retrievers:
            from builder_agent.llm.ollama import OllamaEmbedder
            from builder_agent.memory import FixMemory, retriever
            retrievers[case["id"]] = retriever(FixMemory.build(OllamaEmbedder(EMBED_MODEL), docs=docs), retriever_name)
        retrieve = retrievers[case["id"]]
    injection, mlflow_client = fault(case)
    slots, _ = load_answer(SLOTS)
    client = OllamaClient(model, options=DEFAULT_OPTIONS | {"temperature": temperature, "seed": seed})
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kw = {"runner": runner} if runner else {}
    result = run_fix_loop(REPO, slots, prompt, client, expected=json.loads(EXPECTED.read_text(encoding="utf-8")),
                          mlflow_client=mlflow_client, injection=injection, retriever=retrieve,
                          retriever_name=retriever_name, python_rule=False,
                          sandbox_log=d / "sandbox_attempts.jsonl", fix_log=d / "fix_attempts.jsonl", **kw)
    if why := invalid(result, case):
        shutil.rmtree(d)
        raise Stop(f"{config} {case['id']} seed {seed}: {why}; the run was deleted")
    passed = next((a.n for a in result.attempts if a.outcome == "passed"), None)
    record = {
        "pass": pass_, "config": config, "case": case["id"], "variant": case.get("variant", case.get("name")),
        "family": case.get("family", "version_mismatch"), "seed": seed, "temperature": temperature,
        "model": result.model, "model_digest": result.model_digest,
        "prompt_version": result.prompt_version, "prompt_sha256": result.prompt_sha256,
        "retriever": retriever_name, "memory": "loo" if retriever_name else None,
        "memory_ids": sorted(doc.id for doc in docs) if retriever_name else [],
        "memory_sha256": snapshot_sha256(docs) if retriever_name else None, "python_rule": False,
        "initial": result.initial, "final_ok": result.final_ok, "stopped_because": result.stopped_because,
        "first_fix_passed": passed == 1, "fixes_to_pass": passed,
        "attempts": [a.model_dump(exclude={"calls"}) | {"n_calls": len(a.calls),
                                                        "rejected": [c.reasons for c in a.calls if not c.valid]}
                     for a in result.attempts],
        "started": started, "seconds": round(time.monotonic() - t0, 1),
    }
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(d / "result.json")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pass", dest="pass_", choices=list(PASSES), required=True)
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    p.add_argument("--cases", nargs="+", help="case ids (default: all 14)")
    p.add_argument("--seeds", nargs="+", type=int, help="default: the pass's seeds")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--list", action="store_true", help="show what is done and what is left, run nothing")
    args = p.parse_args()

    cases = {c["id"]: c for c in all_cases()}
    case_ids = args.cases or list(cases)
    if unknown := set(case_ids) - set(cases):
        p.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    seeds = args.seeds or PASSES[args.pass_][1]
    out = Path(args.out)
    todo = plan(args.pass_, args.configs, case_ids, seeds)
    left = [t for t in todo if not is_done(run_dir(out, args.pass_, *t))]
    print(f"{args.pass_}: {len(todo) - len(left)} of {len(todo)} runs done, {len(left)} left", flush=True)
    if args.list:
        return
    retrievers: dict = {}
    for i, (config, case_id, seed) in enumerate(left, 1):
        if problem := health():
            print(f"STOP before {config} {case_id} seed {seed}: {problem}", flush=True)
            sys.exit(2)
        d = run_dir(out, args.pass_, config, case_id, seed)
        try:
            rec = run_one(d, args.pass_, config, cases[case_id], seed, retrievers, args.model)
        except Stop as e:
            print(f"STOP: {e}", flush=True)
            sys.exit(2)
        if problem := health():                     # down during the run: don't trust it
            shutil.rmtree(d, ignore_errors=True)
            print(f"STOP after {config} {case_id} seed {seed}: {problem}; the run was deleted", flush=True)
            sys.exit(2)
        print(f"[{i}/{len(left)}] {config} {case_id} seed {seed}: first fix "
              f"{'passed' if rec['first_fix_passed'] else 'no'}, fixed {'yes' if rec['final_ok'] else 'no'}, "
              f"{len(rec['attempts'])} fix(es), {rec['seconds']:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
