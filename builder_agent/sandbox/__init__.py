"""Sandbox runner: render the pipeline into a copy of the repo and run it in Docker. No LLM.

    from builder_agent.sandbox import run_sandbox
    result = run_sandbox("../taxi-trip-regression", slots=..., expected=...)
    print(result.summary())

The real repo is never written to: it is copied to a temp folder (without .git
and without its data folders), and the data folders are bind-mounted read-only.
Stages run in order and stop at the first failure:

    render -> build -> mlflow -> data -> train -> evaluate -> serve -> health -> predict

Every attempt is appended to an attempts log (JSON lines).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ..decide import load_contracts, plan_build
from ..render import Expected, render_adapters, validate_slots
from ..render.configs import _top_dirs, config_context, render_configs
from ..scan import scan_repo

STAGES = ["render", "build", "mlflow", "data", "train", "evaluate", "serve", "health", "predict"]
TAIL_LINES = 50
OVERRIDE_FILE = "compose.sandbox.yml"
DEFAULT_CONTRACTS = Path(__file__).resolve().parents[2] / "contracts.yaml"
DEFAULT_LOG = Path(__file__).resolve().parents[2] / "logs" / "sandbox_attempts.jsonl"
TIMEOUTS = {"build": 1800, "mlflow": 180, "data": 600, "train": 1800, "evaluate": 900,
            "serve": 120, "health": 180, "predict": 120, "teardown": 180}
COPY_SKIP = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}

# (command, cwd, timeout) -> (exit code, combined stdout+stderr)
Runner = Callable[[list[str], Path, float], tuple[int, str]]


class StageResult(BaseModel):
    name: str
    status: Literal["ok", "failed", "skipped"]
    duration_s: float = 0.0
    command: str | None = None
    exit_code: int | None = None
    output_tail: list[str] = Field(default_factory=list)   # last 50 lines on failure, last few on success


class SandboxResult(BaseModel):
    attempt_id: str
    repo: str
    started_at: str
    ok: bool
    failed_stage: str | None
    total_s: float
    workdir: str
    compose_project: str
    stages: list[StageResult]
    teardown: str | None = None

    def summary(self) -> str:
        out = [f"SANDBOX {self.repo} | attempt {self.attempt_id} | "
               + ("ALL STAGES OK" if self.ok else f"FAILED at {self.failed_stage}") + f" | {self.total_s:.0f} s"]
        for s in self.stages:
            out.append(f"  {s.name:<9} {s.status:<7} {s.duration_s:7.1f} s")
        failed = next((s for s in self.stages if s.status == "failed"), None)
        if failed:
            out.append(f"--- {failed.name}: `{failed.command}` exit {failed.exit_code}, last lines:")
            out += [f"  | {line}" for line in failed.output_tail]
        return "\n".join(out)


def subprocess_runner(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired as e:
        partial = e.output.decode("utf-8", "replace") if isinstance(e.output, bytes) else (e.output or "")
        return 124, partial + f"\n[sandbox] timed out after {timeout:.0f} s"


def _tail(text: str, n: int) -> list[str]:
    return [line for line in text.splitlines() if line.strip()][-n:]


# --- preparation --------------------------------------------------------------

def copy_repo(src: Path, dst: Path, skip_top: set[str]) -> None:
    """Copy the repo without .git, environments, caches and the top-level data folders."""
    def ignore(directory: str, names: list[str]) -> set[str]:
        top = Path(directory).resolve() == src.resolve()
        return {n for n in names if n in COPY_SKIP or (top and n in skip_top)}
    shutil.copytree(src, dst, ignore=ignore)


def write_override(workdir: Path, compose_file: str, model_service: str, real_repo: Path) -> Path:
    """compose.sandbox.yml: no host ports, data bind-mounted read-only from the real repo."""
    base = yaml.safe_load((workdir / compose_file).read_text(encoding="utf-8"))
    services = base["services"]
    volumes = []
    for v in services[model_service].get("volumes", []):
        src, target, *mode = v.split(":")
        if mode == ["ro"] and src.startswith("./"):
            volumes.append({"type": "bind", "source": (real_repo / src[2:]).resolve().as_posix(),
                            "target": target, "read_only": True})
        else:
            volumes.append(v)
    override = {"services": {name: {"ports": _Tagged("!reset", [])} for name in services}}
    override["services"][model_service]["volumes"] = _Tagged("!override", volumes)
    path = workdir / OVERRIDE_FILE
    path.write_text("# Sandbox only (not a Builder output): no host ports, real data mounted read-only.\n"
                    + yaml.dump(override, Dumper=_Dumper, sort_keys=False), encoding="utf-8")
    return path


class _Tagged:
    """A YAML value with a compose merge tag (!reset / !override)."""
    def __init__(self, tag: str, value):
        self.tag, self.value = tag, value


class _Dumper(yaml.SafeDumper):
    pass


def _represent_tagged(dumper: yaml.SafeDumper, data: _Tagged):
    return dumper.represent_sequence(data.tag, data.value, flow_style=not data.value)


_Dumper.add_representer(_Tagged, _represent_tagged)


# --- the run ------------------------------------------------------------------

def run_sandbox(repo: str | Path, slots: dict, expected: dict | None = None,
                contracts_path: str | Path = DEFAULT_CONTRACTS, log_path: str | Path | None = DEFAULT_LOG,
                keep: bool = False, runner: Runner = subprocess_runner,
                mlflow_client: str | None = None) -> SandboxResult:
    """mlflow_client overrides the MLflow client install spec (fault injection, e.g. "mlflow")."""
    repo = Path(repo).resolve()
    started = time.monotonic()
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    ctx = scan_repo(repo)
    c = load_contracts(contracts_path)
    plan = plan_build(ctx, c)
    workdir = Path(tempfile.mkdtemp(prefix="builder-sbx-")) / ctx.name
    project = f"builder-sbx-{ctx.name}".lower()
    stages: list[StageResult] = []
    names: dict = {}

    def record(name: str, t0: float, code: int, output: str, command: str | None) -> bool:
        ok = code == 0
        stages.append(StageResult(name=name, status="ok" if ok else "failed",
                                  duration_s=round(time.monotonic() - t0, 1), command=command,
                                  exit_code=code, output_tail=_tail(output, 5 if ok else TAIL_LINES)))
        return ok

    def compose(*args: str) -> list[str]:
        return ["docker", "compose", "-p", project, "-f", names["compose_file"], "-f", OVERRIDE_FILE, *args]

    def run(name: str, cmd: list[str], logs_on_failure: str | None = None) -> bool:
        t0 = time.monotonic()
        code, output = runner(cmd, workdir, TIMEOUTS[name])
        if code != 0 and logs_on_failure:
            _, logs = runner(compose("logs", "--no-color", "--tail", str(TAIL_LINES), logs_on_failure),
                             workdir, 60)
            output += f"\n[sandbox] logs of {logs_on_failure}:\n{logs}"
        return record(name, t0, code, output, shlex.join(cmd))

    # 1. render: copy the repo, render adapters + configs, write the sandbox override
    t0 = time.monotonic()
    try:
        v = validate_slots(ctx, slots)
        if not v.ok:
            raise ValueError("invalid slot answers: " + "; ".join(v.reasons))
        tctx = config_context(ctx, c, plan, v.slots, mlflow_client)
        names = {"compose_file": tctx["compose_file"], "model": tctx["model_service"],
                 "mlflow": tctx["mlflow_service"], "mlflow_port": tctx["mlflow_port"],
                 "port": tctx["port"], "adapters": c.paths["adapters_dir"].strip("/"),
                 "sample": c.sample_mode.variable}
        copy_repo(repo, workdir, set(_top_dirs(ctx, c)[0]))
        a = render_adapters(ctx, c, v.slots, workdir, expected=Expected.model_validate(expected) if expected else None)
        cfg = render_configs(ctx, c, plan, v.slots, workdir, mlflow_client)
        write_override(workdir, names["compose_file"], names["model"], repo)
        lint = a.lint + cfg.lint
        ok = record("render", t0, 1 if lint else 0,
                    "\n".join(lint) or f"rendered {len(a.files) + len(cfg.files)} files into {workdir}", None)
    except Exception as e:  # noqa: BLE001 - any render problem is a failed stage, not a crash
        ok = record("render", t0, 1, f"{type(e).__name__}: {e}", None)

    mlflow_ready = ["exec", "-T", names.get("mlflow", ""), "python", "-c",
                    f"import urllib.request; urllib.request.urlopen('http://localhost:{names.get('mlflow_port')}/health')"]
    smoke = ["exec", "-T", names.get("model", ""), "python", f"{names.get('adapters')}/smoke_test.py",
             "--url", f"http://localhost:{names.get('port')}"]
    steps: list[tuple[str, Callable[[], bool]]] = [
        ("build", lambda: run("build", compose("--progress", "plain", "build"))),
        ("mlflow", lambda: _start_and_wait(run, runner, compose, workdir, names["mlflow"], mlflow_ready, record)),
        *[(target, (lambda target=target: run(target, compose(
            "run", "--rm", "-T", "--no-deps", names["model"], "make", target, f"{names['sample']}=1"))))
          for target in ("data", "train", "evaluate")],
        ("serve", lambda: run("serve", compose("up", "-d", "--no-deps", names["model"]))),
        ("health", lambda: run("health", compose(*smoke, "--check", "health", "--wait", "90"),
                               logs_on_failure=names["model"])),
        ("predict", lambda: run("predict", compose(*smoke, "--check", "predict"), logs_on_failure=names["model"])),
    ]
    for name, step in steps:
        if not ok:
            stages.append(StageResult(name=name, status="skipped"))
            continue
        ok = step()

    if keep:
        teardown = f"kept: containers still up, files in {workdir}"
    else:
        teardown = "ok"
        if names and (workdir / OVERRIDE_FILE).exists():
            code, out = runner(compose("down", "-v", "--remove-orphans"), workdir, TIMEOUTS["teardown"])
            teardown = "ok" if code == 0 else f"failed: {' | '.join(_tail(out, 3))}"
        shutil.rmtree(workdir.parent, ignore_errors=True)

    failed = next((s.name for s in stages if s.status == "failed"), None)
    result = SandboxResult(
        attempt_id=attempt_id, repo=ctx.name, started_at=datetime.now(timezone.utc).isoformat(),
        ok=failed is None, failed_stage=failed, total_s=round(time.monotonic() - started, 1),
        workdir=str(workdir), compose_project=project, stages=stages, teardown=teardown,
    )
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
    return result


def _start_and_wait(run, runner: Runner, compose, workdir: Path, service: str, ready_cmd: list[str],
                    record) -> bool:
    """`up -d <service>`, then poll its health endpoint until it answers or the stage times out."""
    t0 = time.monotonic()
    up = compose("up", "-d", "--no-deps", service)
    code, output = runner(up, workdir, TIMEOUTS["mlflow"])
    if code == 0:
        deadline = t0 + TIMEOUTS["mlflow"]
        while True:
            code, out = runner(compose(*ready_cmd), workdir, 30)
            if code == 0 or time.monotonic() > deadline:
                output += "\n" + out
                break
            time.sleep(3)
    if code != 0:
        _, logs = runner(compose("logs", "--no-color", "--tail", str(TAIL_LINES), service), workdir, 60)
        output += f"\n[sandbox] logs of {service}:\n{logs}"
    return record("mlflow", t0, code, output, shlex.join(up))
