"""Leave-one-out evaluation runner (docs/EVAL_PROTOCOL.md). Resumable; stops when Docker or Ollama is down.

    python experiments/eval_loo/run.py --pass main --configs v3-advanced            # one config per session
    python experiments/eval_loo/run.py --pass t0                                    # the temperature-0 pass
    python experiments/eval_loo/run.py --pass main --cases gen-002 gen-013 --seeds 101 --out experiments/eval_loo/_smoke
    python experiments/eval_loo/run.py --pass main --list                           # what is done, what is left

One run = one fix loop on one case with one config and one seed, LLM-only (--no-python-rule), with the
case's leave-one-out memory (builder_agent.memory.loo_docs), no write-back. Each run goes to
<out>/<pass>/<config>/<case>/seed-<n>/ (fix_attempts.jsonl, sandbox_attempts.jsonl, result.json). result.json
is written last (via a temp file), so a run without it is incomplete: it is deleted and done again.

Speed (the protocol is unchanged):
- runs are grouped by case: all configs and seeds of one case run one after another, so the Docker builds of
  that case's fault and fixes are still in the build cache (cases in a fixed shuffled order, and the runs
  within a case too);
- the first sandbox run of a case (the injected fault, before any fix) is the same in every config, seed and
  pass, so it runs once and is shared (docs/EVAL_PROTOCOL.md section 5 item 4). It is stored in
  <out>/_initial/<case>.json with a fingerprint of its inputs (the Builder's code, contracts.yaml, the gold
  files, the taxi repo's HEAD), reused only while that fingerprint holds and never from uncommitted code,
  and only if it failed at the case's recorded stage. Each run's sandbox_attempts.jsonl starts with a copy.

A stopped or interrupted run is never deleted: it is moved to <out>/_stopped/<pass>/<config>/<case>/seed-<n>-<time>/
with a reason.txt (EVAL_PROTOCOL.md, "Changes during the run (2026-10-14)").

A Docker error in a run's output while Docker is healthy right after it: the run is moved to _stopped and done once
more with the same seed. If the error comes back, the output is the run's own and it is kept as a failed attempt
(result.json records the line); if Docker isn't healthy, the runner stops.

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
from builder_agent.sandbox import SandboxResult, run_sandbox  # noqa: E402
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
    """Every (config, case, seed), grouped by case: cases in a fixed shuffled order, and within a case its
    configs and seeds in a fixed shuffled order (the build cache stays warm for the case)."""
    rng = random.Random(f"{ORDER_SEED}-{pass_}")
    order = list(case_ids)
    rng.shuffle(order)
    todo = []
    for c in order:
        runs = [(cfg, c, s) for cfg in configs for s in seeds]
        rng.shuffle(runs)
        todo += runs
    return todo


def fingerprint() -> dict:
    """What the first sandbox run depends on: if any of it changes, the stored first runs are not reused."""
    def git(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    dirty = git(ROOT, "status", "--porcelain", "--", "builder_agent", "contracts.yaml", "tests/gold")
    return {"builder_agent": git(ROOT, "rev-parse", "HEAD:builder_agent"),
            "contracts": git(ROOT, "rev-parse", "HEAD:contracts.yaml"),
            "gold": git(ROOT, "rev-parse", "HEAD:tests/gold"),
            "taxi_head": git(REPO, "rev-parse", "HEAD"),
            "uncommitted": bool(dirty)}


def initial_run(out: Path, case: dict, sandbox=None) -> tuple[SandboxResult, bool]:
    """The case's first sandbox run: stored and reused while the fingerprint holds. Returns (result, reused)."""
    path = out / "_initial" / f"{case['id']}.json"
    fp = fingerprint()
    if path.exists() and not fp["uncommitted"]:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["fingerprint"] == fp:
            return SandboxResult.model_validate(stored["result"]), True
    injection, mlflow_client = fault(case)
    slots, _ = load_answer(SLOTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = (sandbox or run_sandbox)(REPO, slots, expected=json.loads(EXPECTED.read_text(encoding="utf-8")),
                                 log_path=path.with_suffix(".sandbox.jsonl"), mlflow_client=mlflow_client,
                                 overrides=injection)
    failed = next((s for s in r.stages if s.status == "failed"), None)
    if r.failed_stage != case["stage"] or failed is None:
        raise Stop(f"{case['id']}: the fault didn't reproduce: first sandbox run failed at {r.failed_stage}, "
                   f"the case says {case['stage']}")
    signature_seen = case["error_signature"].strip() in "\n".join(failed.output_tail)
    if not fp["uncommitted"]:                     # never store a run made from uncommitted code
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"fingerprint": fp, "case": case["id"], "signature_seen": signature_seen,
                                   "result": r.model_dump()}, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
        tmp.replace(path)
    return r, False


def docker_line(result) -> str | None:
    """The first output line that looks like a Docker engine error, or None."""
    tails = list(result.final_error_tail) + [line for a in result.attempts for line in a.error_tail]
    return next((line for line in tails if any(p.lower() in line.lower() for p in INFRA_PATTERNS)), None)


def invalid(result, case: dict) -> str | None:
    """Why a finished run didn't measure the model, or None (a Docker error line is handled by run_one)."""
    if any(c.error for a in result.attempts for c in a.calls):
        return "an LLM call failed (Ollama)"
    if line := docker_line(result):
        return f"a Docker error is in the sandbox output: {line.strip()[:200]}"
    if result.initial != f"failed at {case['stage']}":
        return f"the fault didn't reproduce: first sandbox run {result.initial!r}, the case says failed at {case['stage']}"
    return None


def move_to_stopped(d: Path, reason: str) -> Path:
    """Keep a stopped run as evidence: <out>/_stopped/<pass>/<config>/<case>/<seed>-<time>/, with reason.txt."""
    out, rel = d.parents[3], d.relative_to(d.parents[3])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out / "_stopped" / rel.parent / f"{rel.name}-{stamp}"
    n = 1
    while dest.exists():                            # two moves in the same second: never nest one in the other
        n += 1
        dest = out / "_stopped" / rel.parent / f"{rel.name}-{stamp}-{n}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    (dest / "reason.txt").write_text(reason + "\n", encoding="utf-8", newline="\n")
    return dest


def run_one(d: Path, pass_: str, config: str, case: dict, seed: int, retrievers: dict, model: str,
            run_fix_loop=run_fix_loop, runner=None, initial: SandboxResult | None = None, health=health,
            reruns: list | None = None) -> dict:
    """reruns: the earlier tries of this run that hit a Docker error with Docker healthy (moved to _stopped)."""
    prompt, retriever_name = CONFIGS[config]
    temperature, _ = PASSES[pass_]
    reruns = list(reruns or [])
    if d.exists():
        move_to_stopped(d, "incomplete: no result.json (the runner was interrupted); started again")
    d.mkdir(parents=True)
    if initial is not None:                         # the shared first run heads this run's own sandbox log
        (d / "sandbox_attempts.jsonl").write_text(initial.model_dump_json() + "\n", encoding="utf-8", newline="\n")
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
                          retriever_name=retriever_name, python_rule=False, initial=initial,
                          sandbox_log=d / "sandbox_attempts.jsonl", fix_log=d / "fix_attempts.jsonl", **kw)
    kept_docker_line = None
    if why := invalid(result, case):
        line = docker_line(result)
        if line and not any(c.error for a in result.attempts for c in a.calls) and health() is None:
            if not reruns:                          # Docker is fine: do the run once more, same seed
                kept = move_to_stopped(d, f"{why}; Docker healthy right after: run again once")
                return run_one(d, pass_, config, case, seed, retrievers, model, run_fix_loop, runner, initial,
                               health, reruns=[{"moved_to": str(kept), "line": line.strip()[:200]}])
            kept_docker_line = line.strip()[:200]   # the same again: the run's own output, kept as a failed attempt
            if result.initial != f"failed at {case['stage']}":
                kept_docker_line = None             # not the Docker case after all: stop below
        if kept_docker_line is None:
            kept = move_to_stopped(d, why)
            raise Stop(f"{config} {case['id']} seed {seed}: {why}; the run was moved to {kept}")
    passed = next((a.n for a in result.attempts if a.outcome == "passed"), None)
    record = {
        "pass": pass_, "config": config, "case": case["id"], "variant": case.get("variant", case.get("name")),
        "family": case.get("family", "version_mismatch"), "seed": seed, "temperature": temperature,
        "model": result.model, "model_digest": result.model_digest,
        "prompt_version": result.prompt_version, "prompt_sha256": result.prompt_sha256,
        "retriever": retriever_name, "memory": "loo" if retriever_name else None,
        "memory_ids": sorted(doc.id for doc in docs) if retriever_name else [],
        "memory_sha256": snapshot_sha256(docs) if retriever_name else None, "python_rule": False,
        "initial": result.initial, "initial_attempt_id": result.initial_attempt_id,
        "initial_shared": initial is not None, "final_ok": result.final_ok, "stopped_because": result.stopped_because,
        "first_fix_passed": passed == 1, "fixes_to_pass": passed,
        "attempts": [a.model_dump(exclude={"calls"}) | {"n_calls": len(a.calls),
                                                        "rejected": [c.reasons for c in a.calls if not c.valid]}
                     for a in result.attempts],
        "started": started, "seconds": round(time.monotonic() - t0, 1),
        "reruns": reruns, "docker_line_kept": kept_docker_line,
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
    firsts: dict = {}
    for i, (config, case_id, seed) in enumerate(left, 1):
        if problem := health():
            print(f"STOP before {config} {case_id} seed {seed}: {problem}", flush=True)
            sys.exit(2)
        d = run_dir(out, args.pass_, config, case_id, seed)
        try:
            if case_id not in firsts:
                t0 = time.monotonic()
                first, reused = initial_run(out, cases[case_id])
                firsts[case_id] = first
                print(f"{case_id}: first sandbox run {'reused' if reused else 'done'} (failed at "
                      f"{first.failed_stage}, {time.monotonic() - t0:.0f} s)", flush=True)
            rec = run_one(d, args.pass_, config, cases[case_id], seed, retrievers, args.model,
                          initial=firsts[case_id])
        except Stop as e:
            print(f"STOP: {e}", flush=True)
            sys.exit(2)
        if problem := health():                     # down during the run: don't trust it
            kept = move_to_stopped(d, f"Docker or Ollama down right after the run: {problem}")
            print(f"STOP after {config} {case_id} seed {seed}: {problem}; the run was moved to {kept}", flush=True)
            sys.exit(2)
        if rec["reruns"]:
            print(f"    (a Docker error with Docker healthy: run again once; "
                  f"{'the same error again, kept as a failed attempt' if rec['docker_line_kept'] else 'clean the second time'})",
                  flush=True)
        print(f"[{i}/{len(left)}] {config} {case_id} seed {seed}: first fix "
              f"{'passed' if rec['first_fix_passed'] else 'no'}, fixed {'yes' if rec['final_ok'] else 'no'}, "
              f"{len(rec['attempts'])} fix(es), {rec['seconds']:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
