"""Render layer for the config files: Dockerfile, .dockerignore, Makefile, compose base file.

Inputs are the BuildPlan's decided ([ok]) items and, for the targets the rules
couldn't settle, the pipeline/ adapters rendered from validated slot answers.
Nothing is built or run here; lint_configs() checks the files statically.
"""

from __future__ import annotations

import posixpath
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel

from ..decide.contracts import Contracts
from ..decide.models import BuildPlan
from ..scan.deps import declared_names, normalize
from ..scan.models import RepoContext
from ..repo_writer import RepoWriter
from . import MARKER, RenderError, RenderResult, _env, manifest_path, write_new_files
from .slots import FLAVOR_DIST, SlotAnswers, validate_slots

MLFLOW_IMAGE = "ghcr.io/mlflow/mlflow:v2.17.2"   # tag checked to exist; Builder-internal default
WORKDIR = "/app"
ADAPTER_SCRIPTS = {"data": "data.py", "train": "train.py", "evaluate": "evaluate.py"}
ADAPTER_PACKAGES = ["mlflow", "pandas", "numpy", "flask", "gunicorn"]   # what the adapters import
TEST_STUB = ['@echo "make test: no smoke test: the adapters were not rendered" >&2', "@exit 1"]


def mlflow_client_spec(image: str) -> str:
    """mlflow==X.Y.Z from the MLflow server image tag, so client and server can't drift apart.

    See tests/faults/mlflow_client_server_mismatch: an unpinned 3.x client can't log models
    to a 2.17.2 server.
    """
    tag = image.rsplit("/", 1)[-1].partition(":")[2]
    m = re.fullmatch(r"v?(\d+\.\d+\.\d+)", tag)
    if not m:
        raise RenderError(f"can't derive the MLflow client version from image {image!r}: use a vX.Y.Z tag")
    return f"mlflow=={m.group(1)}"


class MakeRule(BaseModel):
    name: str
    description: str
    commands: list[str]
    note: str | None = None


def _deps_files(plan: BuildPlan) -> list[str] | None:
    """Files to COPY before installing, or None when the install needs the whole source."""
    tool, source = plan.install.tool, plan.install.source
    if tool == "pipenv":
        return ["Pipfile", "Pipfile.lock"] if source == "Pipfile.lock" else ["Pipfile"]
    if tool == "poetry":
        return ["pyproject.toml", "poetry.lock"]
    if tool == "pip" and source and source.endswith((".txt", ".in")):
        return [source]
    return None


def _top_dirs(ctx: RepoContext, c: Contracts) -> tuple[list[str], list[str], list[str]]:
    """(data dirs, model dirs, output dirs) at the repo top level, never holding code."""
    code_dirs = {PurePosixPath(e.path).parts[0] for e in ctx.entry_points if "/" in e.path}

    def top(path: str | None) -> str | None:
        return PurePosixPath(path).parts[0] if path else None

    data = {top(d.path) for d in ctx.data_dirs} | {top(c.paths.get("raw_data")), top(c.paths.get("processed_data"))}
    models = {top(d.path) for d in ctx.model_dirs} | {top(c.paths.get("models"))}
    outputs = {top(posixpath.dirname(c.paths[k])) for k in ("eval_report", "predictions_log")
               if posixpath.dirname(c.paths.get(k, ""))}
    clean = lambda dirs: sorted(d for d in dirs if d and d not in code_dirs)  # noqa: E731
    return clean(data), clean(models), clean(outputs - data - models)


def _make_rules(plan: BuildPlan, c: Contracts, adapters: str | None, serve_cmd: str,
                install: list[str]) -> list[MakeRule]:
    rules, missing = [], []
    for name, desc in c.make_targets.items():
        planned = plan.target(name) if any(t.name == name for t in plan.make_targets) else None
        note = None
        if name == "install":
            cmds = install
        elif name == "serve":
            cmds = [serve_cmd]   # resolved in config_context (repo app or serve.py adapter)
            if planned and planned.status != "decided":
                note = f"adapter, because {planned.reason}"
        elif planned and planned.status == "decided" and planned.command:
            cmds = [planned.command]
        elif name in ADAPTER_SCRIPTS and adapters:
            cmds = [f"$(PYTHON) {adapters}/{ADAPTER_SCRIPTS[name]}"]
            note = f"adapter, because {planned.reason}" if planned else None
        elif name == "test" and adapters:
            cmds = [f"$(PYTHON) {adapters}/smoke_test.py"]
            note = "in-process (Flask test client); the sandbox also runs it over HTTP with --url"
        elif name == "test":
            cmds, note = TEST_STUB, "fails on purpose: no smoke test without the adapters"
        else:
            missing.append(f"{name} ({planned.reason if planned else 'no rule'})")
            continue
        rules.append(MakeRule(name=name, description=desc, commands=cmds, note=note))
    if missing:
        raise RenderError("no command for make target(s): " + "; ".join(missing))
    return rules


def config_context(ctx: RepoContext, c: Contracts, plan: BuildPlan, slots: SlotAnswers | None,
                   mlflow_client: str | None = None) -> dict:
    """mlflow_client overrides the derived client pin (e.g. "mlflow" to reproduce fault-001)."""
    if plan.python.status != "decided":
        raise RenderError(f"Python version not decided: {plan.python.reason}")
    if plan.install.status != "decided":
        raise RenderError(f"install not decided: {plan.install.reason}")

    adapters = c.paths["adapters_dir"].strip("/") if slots else None
    port = c.serving.port
    if plan.serving.status == "decided" and plan.serving.mode == "repo":
        serve_cmd = plan.serving.command          # the repo's own app meets the contract
    elif plan.serving.mode == "generate" and not adapters:
        raise RenderError("serving.mode is generate, but the serve.py adapter needs slot answers")
    elif adapters:
        serve_cmd = f"gunicorn --bind 0.0.0.0:{port} {adapters.replace('/', '.')}.serve:app"
    else:
        raise RenderError(f"serving not decided and no adapters: {plan.serving.reason}")

    # packages the adapters need on top of the repo's own dependencies
    extras = list(plan.install.extra_packages)
    if slots:
        declared = declared_names(ctx.dependency_files)
        for pkg in ADAPTER_PACKAGES + [FLAVOR_DIST[slots.model_flavor]]:
            if normalize(pkg) not in declared and pkg not in extras:
                extras.append(pkg)
    # the MLflow client always matches the server image (fault-001)
    if slots or any(normalize(e) == "mlflow" for e in extras):
        spec = mlflow_client_spec(MLFLOW_IMAGE) if mlflow_client is None else mlflow_client
        extras = [spec] + [e for e in extras if normalize(e) != "mlflow"]
    install = plan.install.commands + ([f"pip install {' '.join(extras)}"] if extras else [])

    mlflow = urlparse(c.paths["mlflow_uri"])
    if not mlflow.hostname or mlflow.hostname in {"localhost", "127.0.0.1"}:
        raise RenderError(f"paths.mlflow_uri {c.paths['mlflow_uri']!r} must name the compose service, "
                          "e.g. http://mlflow:5000")
    compose = (c.model_extra or {}).get("compose", {})
    router = (c.model_extra or {}).get("router", {})
    data_dirs, model_dirs, output_dirs = _top_dirs(ctx, c)
    models_dir = PurePosixPath(c.paths["models"]).parts[0]
    logs_dir = posixpath.dirname(c.paths["predictions_log"])
    health = c.serving.endpoints["health"].path

    return {
        "marker": MARKER,
        "tab": "\t",
        "repo": ctx.name,
        "python_version": plan.python.version,
        "deps_files": _deps_files(plan),
        "install_source": plan.install.source,
        "install_commands": install,
        "workdir": WORKDIR,
        "serve_cmd": shlex.split(serve_cmd),
        "port": port,
        "mlflow_uri": c.paths["mlflow_uri"],
        "mlflow_service": mlflow.hostname,
        "mlflow_port": mlflow.port or 5000,
        "mlflow_image": MLFLOW_IMAGE,
        "model_service": router.get("services", {}).get("current", "serving-current"),
        "model_uri_env": c.serving.model_uri_env,
        "compose_file": compose.get("base_file", "compose.base.yml"),
        "deploy_file": compose.get("deploy_file", "compose.deploy.yml"),
        "data_dirs": data_dirs,
        "model_dirs": model_dirs,
        "models_dir": models_dir,
        "output_dirs": output_dirs,
        "mounts": [f"./{d}:{WORKDIR}/{d}:ro" for d in data_dirs]
                  + [f"./{d}:{WORKDIR}/{d}" for d in dict.fromkeys([models_dir, logs_dir])],
        "healthcheck": ["CMD", "python", "-c",
                        f"import urllib.request; urllib.request.urlopen('http://localhost:{port}{health}', timeout=3)"],
        "sample_var": c.sample_mode.variable,
        "fraction": c.sample_mode.fraction,
        "targets": _make_rules(plan, c, adapters, serve_cmd, install),
    }


def render_configs(ctx: RepoContext, contracts: Contracts, plan: BuildPlan,
                   slots: dict | SlotAnswers | None, out_root: str | Path,
                   mlflow_client: str | None = None, writer: RepoWriter | None = None) -> RenderResult:
    """Render Dockerfile, .dockerignore, Makefile and the compose base file into out_root."""
    answers = None
    if slots is not None:
        v = validate_slots(ctx, slots)
        if not v.ok:
            raise RenderError("invalid slot answers: " + "; ".join(v.reasons))
        answers = v.slots
    tctx = config_context(ctx, contracts, plan, answers, mlflow_client)
    env = _env()
    rendered = {
        "Dockerfile": env.get_template("Dockerfile.j2").render(**tctx),
        ".dockerignore": env.get_template("dockerignore.j2").render(**tctx),
        "Makefile": env.get_template("Makefile.j2").render(**tctx),
        tctx["compose_file"]: env.get_template("compose.base.yml.j2").render(**tctx),
    }
    out_root = Path(out_root)
    write_new_files(writer or RepoWriter(out_root), manifest_path(contracts), rendered)
    return RenderResult(files=list(rendered), lint=lint_configs(out_root, tctx))


# --- static checks -----------------------------------------------------------

def _instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) with line continuations joined and comments dropped."""
    out, buf = [], ""
    for line in dockerfile.splitlines():
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        buf += line.rstrip("\\").strip() + " " if line.rstrip().endswith("\\") else line.strip()
        if not line.rstrip().endswith("\\"):
            word, _, args = buf.partition(" ")
            out.append((word.upper(), args.strip()))
            buf = ""
    return out


def lint_configs(root: Path, tctx: dict) -> list[str]:
    problems: list[str] = []
    bad = problems.append

    # Dockerfile: dependencies before code, no data copied, serves on the contract port
    ins = _instructions((root / "Dockerfile").read_text(encoding="utf-8"))
    kinds = [k for k, _ in ins]
    if not kinds or kinds[0] != "FROM":
        bad("Dockerfile: first instruction must be FROM")
    copies = [(i, a) for i, (k, a) in enumerate(ins) if k in {"COPY", "ADD"}]
    code_copy = next((i for i, a in copies if a.split()[0] == "."), None)
    if tctx["deps_files"]:
        deps_copy = next((i for i, a in copies if a.split()[:-1] == tctx["deps_files"]), None)
        install_run = next((i for i, (k, a) in enumerate(ins) if k == "RUN" and "install" in a
                            and deps_copy is not None and i > deps_copy), None)
        if deps_copy is None or install_run is None or code_copy is None or not deps_copy < install_run < code_copy:
            bad("Dockerfile: dependency files must be copied and installed before `COPY . .` (layer caching)")
    for _, a in copies:
        for src in a.split()[:-1]:
            if any(src.rstrip("/") == d or src.startswith(d + "/") for d in tctx["data_dirs"]):
                bad(f"Dockerfile: copies data ({src}); data must be mounted, not baked in")
    if ("EXPOSE", str(tctx["port"])) not in ins:
        bad(f"Dockerfile: EXPOSE {tctx['port']} missing (contracts.yaml serving.port)")
    if "CMD" not in kinds:
        bad("Dockerfile: no CMD")

    # .dockerignore: every data dir excluded
    ignore = {line.strip() for line in (root / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    for d in tctx["data_dirs"]:
        if f"{d}/" not in ignore and d not in ignore:
            bad(f".dockerignore: data dir {d}/ is not excluded")

    # Makefile: tabs before recipes, every contract target, SAMPLE exported
    make = (root / "Makefile").read_text(encoding="utf-8").splitlines()
    rules = {line.split(":")[0] for line in make if line and not line[0].isspace()
             and not line.startswith(("#", ".")) and line.rstrip().endswith(":")}
    for t in tctx["targets"]:
        if t.name not in rules:
            bad(f"Makefile: no rule for contract target {t.name}")
    phony = next((line for line in make if line.startswith(".PHONY:")), "")
    for t in tctx["targets"]:
        if t.name not in phony.split():
            bad(f"Makefile: {t.name} missing from .PHONY")
    in_rule = False
    for n, line in enumerate(make, 1):
        if line and not line[0].isspace() and line.rstrip().endswith(":") and not line.startswith("#"):
            in_rule = True
        elif not line.strip():
            in_rule = False
        elif in_rule and not line.startswith("\t") and not line.lstrip().startswith("#"):
            bad(f"Makefile line {n}: recipe line must start with a tab")
    if f"export {tctx['sample_var']}" not in make:
        bad(f"Makefile: {tctx['sample_var']} is not exported to the adapters")

    # compose: valid YAML with both services, data mounted read-only
    compose_path = root / tctx["compose_file"]
    try:
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = compose.get("services", {})
        for s in (tctx["mlflow_service"], tctx["model_service"]):
            if s not in services:
                bad(f"{tctx['compose_file']}: service {s} missing")
        vols = services.get(tctx["model_service"], {}).get("volumes", [])
        for d in tctx["data_dirs"]:
            if not any(v.startswith(f"./{d}:") and v.endswith(":ro") for v in vols):
                bad(f"{tctx['compose_file']}: data dir {d} is not mounted read-only")
    except yaml.YAMLError as e:
        bad(f"{tctx['compose_file']}: invalid YAML: {e}")
    problems += compose_config_check(compose_path)
    return problems


def compose_config_check(compose_path: Path) -> list[str]:
    """`docker compose config -q`: parses and validates the file only (no build, no containers)."""
    if not shutil.which("docker"):
        return []
    r = subprocess.run(["docker", "compose", "-f", str(compose_path), "config", "-q"],
                       capture_output=True, text=True, timeout=60)
    return [] if r.returncode == 0 else [f"docker compose config: {r.stderr.strip()}"]
