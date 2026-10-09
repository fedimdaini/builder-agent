"""CI workflow (.github/workflows/ci.yml) for create mode: rendered from scanner facts, no LLM.

Every step comes from a rule; whatever the rules can't decide is listed as needs_llm (in the
result and in a comment at the top of the workflow). Validated with a YAML parse and actionlint.
"""

from __future__ import annotations

import ast
import posixpath
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..decide.contracts import Contracts
from ..decide.models import BuildPlan, NeedsLLM
from ..repo_writer import RepoWriter
from ..scan.deps import declared_names, normalize
from ..scan.models import RepoContext
from ..verify.rules import compose_files, dockerfiles, services
from ..verify.source import RepoSource
from . import MARKER, RenderError, _env
from .actionlint import run_actionlint

CI_PATH = ".github/workflows/ci.yml"
RUNNER = "ubuntu-24.04"                                  # pinned, not ubuntu-latest (fault-002)
ACTIONS = {"checkout": "actions/checkout@v4.2.2", "setup_python": "actions/setup-python@v5.6.0"}
# a test that imports one of these (directly or through a local module) needs live services
SERVICE_CLIENTS = {"requests", "httpx", "aiohttp", "boto3", "botocore", "s3fs", "psycopg2", "sqlalchemy",
                   "redis", "pymongo", "mlflow", "urllib3"}
COMPOSE_VAR_RE = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)(:?[-?+][^}]*)?\}|(?<![\$\w])\$([A-Za-z_][A-Za-z0-9_]*)")


class Build(BaseModel):
    dockerfile: str
    context: str
    tag: str
    output: str                    # step output that says whether this image needs rebuilding
    paths: list[str]               # a change under any of these (prefixes) triggers the build


class CIPlan(BaseModel):
    python_version: str | None = None
    install: list[str] = Field(default_factory=list)
    unit_tests: list[str] = Field(default_factory=list)
    test_install: list[str] = Field(default_factory=list)
    smoke: list[str] = Field(default_factory=list)          # commands, when the smoke test can run
    compose_files: list[str] = Field(default_factory=list)
    builds: list[Build] = Field(default_factory=list)
    precommit: bool = False
    needs_llm: list[NeedsLLM] = Field(default_factory=list)


class CIResult(BaseModel):
    files: list[str]
    plan: CIPlan
    yaml_ok: bool
    actionlint: list[str]
    actionlint_version: str


# --- facts ---------------------------------------------------------------------------

def _imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def classify_tests(src: RepoSource, tests: list[str]) -> tuple[list[str], dict[str, str]]:
    """(unit tests, {service-dependent test: why}). One level of local imports is followed."""
    files = set(src.files())
    unit, service = [], {}
    for t in tests:
        direct = _imports(src.read_text(t))
        why = sorted(direct & SERVICE_CLIENTS)
        if why:
            service[t] = f"imports {', '.join(why)}"
            continue
        folder = posixpath.dirname(t)
        for mod in sorted(direct):
            local = next((p for p in (posixpath.join(folder, f"{mod}.py"), f"{mod}.py",
                                      posixpath.join(folder, mod, "__init__.py")) if p in files), None)
            hit = sorted(_imports(src.read_text(local)) & SERVICE_CLIENTS) if local else []
            if hit:
                service[t] = f"imports {mod} ({local}), which imports {', '.join(hit)}"
                break
        else:
            unit.append(t)
    return unit, service


def compose_vars_without_value(src: RepoSource, compose: list[str]) -> list[str]:
    committed_env = {p for p in src.files() if posixpath.basename(p) == ".env"}
    if committed_env:
        return []
    missing = set()
    for f in compose:
        for line in src.read_text(f).splitlines():
            if line.lstrip().startswith("#"):
                continue
            for m in COMPOSE_VAR_RE.finditer(line):
                name, default = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), None)
                if not default:
                    missing.add(name)
    return sorted(missing)


# --- the plan ---------------------------------------------------------------------------------

def plan_ci(ctx: RepoContext, c: Contracts, plan: BuildPlan, created: dict[str, str] | None = None) -> CIPlan:
    """created: files this Builder run adds (path -> text), e.g. Dockerfile, compose.base.yml, smoke test."""
    created = created or {}
    src = RepoSource(ctx.root)
    ci = CIPlan()
    need = ci.needs_llm.append
    declared = declared_names(ctx.dependency_files)

    if plan.python.status == "decided":
        ci.python_version = plan.python.version
    else:
        need(NeedsLLM(item="ci:python", reason=f"Python version not decided: {plan.python.reason}",
                      context=[f"{h.value} ({h.kind}: {h.source})" for h in ctx.python_version_hints][:6]))

    # pre-commit: system hooks need their tool installed from the repo's own dependencies
    dev_needed = False
    if src.exists(".pre-commit-config.yaml"):
        ci.precommit = True
        cfg = yaml.safe_load(src.read_text(".pre-commit-config.yaml")) or {}
        for repo in cfg.get("repos", []):
            for hook in repo.get("hooks", []):
                if hook.get("language") == "system":
                    tool = str(hook.get("entry", hook["id"])).split()[0]
                    dev = {normalize(r.name) for d in ctx.dependency_files for r in d.packages if r.dev}
                    if normalize(tool) in dev:
                        dev_needed = True
                    elif normalize(tool) not in declared:
                        need(NeedsLLM(item="ci:pre-commit", reason=f"system hook {hook['id']} runs {tool}, "
                                      "which no dependency file declares", context=[f"hook: {hook}"]))

    if plan.install.status == "decided":
        cmds = list(plan.install.commands)
        if dev_needed and plan.install.tool == "pipenv":
            cmds = [cmd.replace("pipenv install ", "pipenv install --dev ", 1) if cmd.startswith("pipenv install")
                    else cmd for cmd in cmds]
        ci.install = cmds
    else:
        need(NeedsLLM(item="ci:install", reason=plan.install.reason))

    # tests
    if ctx.test_files:
        unit, service = classify_tests(src, ctx.test_files)
        ci.unit_tests = unit
        if unit:
            folders = sorted({posixpath.dirname(t) for t in unit} - {""})
            ci.test_install = [f"pip install -r {d.path}" for d in ctx.dependency_files
                               if d.kind == "requirements" and posixpath.dirname(d.path) in folders]
            if "pytest" not in declared:
                ci.test_install.append("pip install pytest")
        if service:
            need(NeedsLLM(item="ci:service-tests", reason="these tests need running services (run them against "
                          "compose services, mock the clients, or leave them out of CI)",
                          context=[f"{t}: {why}" for t, why in service.items()]))
    else:
        smoke = f"{c.paths['adapters_dir'].strip('/')}/smoke_test.py"
        if smoke in created or src.exists(smoke):
            data = sorted(ctx.lfs.tracked_files) + [p.path for p in ctx.lfs.pointer_files]
            processed = c.paths.get("processed_data", "")
            lfs_data = [f for f in data if f.startswith(processed + "/")]
            if lfs_data:
                need(NeedsLLM(item="ci:smoke-data", reason="no test files, and the generated smoke test needs "
                              f"{processed}/ for make data/train, which is in Git LFS: fetch it in CI (LFS "
                              "bandwidth), use a small committed sample, or skip the smoke test",
                              context=[f"LFS: {f}" for f in lfs_data][:6]))
            else:
                ci.smoke = [f"make data train evaluate {c.sample_mode.variable}=1", "make test"]
        else:
            need(NeedsLLM(item="ci:tests", reason="no test files and no generated smoke test: nothing to run"))

    # compose files and Dockerfiles: the repo's and this run's
    ci.compose_files = sorted(set(compose_files(src)) | {f for f in created if f.endswith((".yml", ".yaml"))
                                                          and posixpath.basename(f).startswith(("compose", "docker-compose"))})
    missing = compose_vars_without_value(src, [f for f in ci.compose_files if src.exists(f)])
    if missing:
        need(NeedsLLM(item="ci:compose-env", reason=f"{len(missing)} variable(s) used by the compose files have no "
                      "default and no committed env file: `config` passes with blanks; CI needs values",
                      context=[", ".join(missing)]))
    svcs = [s for f in ci.compose_files if src.exists(f) for s in services(src, f)]
    for df in sorted(set(dockerfiles(src)) | {f for f in created if posixpath.basename(f) == "Dockerfile"}):
        ctx_dir = next((s.build_context() for s in svcs if s.build_dockerfile() == df), None)
        ctx_dir = ctx_dir or posixpath.dirname(df) or "."
        slug = re.sub(r"[^a-z0-9]+", "-", df.lower().replace("dockerfile", "")).strip("-") or "app"
        # the build context is the whole repo for ".": then any change triggers it
        paths = [df] + ([] if ctx_dir == "." else [ctx_dir.rstrip("/") + "/"])
        ci.builds.append(Build(dockerfile=df, context=ctx_dir, tag=f"ci/{slug}",
                               output="build_" + slug.replace("-", "_"), paths=paths if ctx_dir != "." else [""]))
    return ci


# --- render ------------------------------------------------------------------------------------

def render_ci(ctx: RepoContext, contracts: Contracts, plan: BuildPlan, out_root: str | Path,
              writer: RepoWriter | None = None, created: dict[str, str] | None = None) -> CIResult:
    ci = plan_ci(ctx, contracts, plan, created)
    text = render_ci_text(ctx, contracts, ci)
    yaml_ok, problems, version = validate_workflow(text)
    if not yaml_ok or problems:
        raise RenderError("the rendered workflow is not valid, nothing written: " + "; ".join(problems))
    writer = writer or RepoWriter(out_root)
    try:
        writer.write_new({CI_PATH: text})
    except Exception as e:  # noqa: BLE001 - WriteRefused and friends
        raise RenderError(str(e)) from e
    return CIResult(files=[CI_PATH], plan=ci, yaml_ok=yaml_ok, actionlint=problems, actionlint_version=version)


# GitHub expressions, kept out of Jinja
GH = {name: "${{ github.%s }}" % path for name, path in {
    "ref": "ref", "workspace": "workspace", "event_name": "event_name", "base_ref": "base_ref",
    "before": "event.before"}.items()}


def render_ci_text(ctx: RepoContext, contracts: Contracts, ci: CIPlan) -> str:
    return _env().get_template("ci.yml.j2").render(marker=MARKER, repo=ctx.name, ci=ci, runner=RUNNER, gh=GH,
                                                   actions=ACTIONS, sample_var=contracts.sample_mode.variable)


def validate_workflow(text: str) -> tuple[bool, list[str], str]:
    """(YAML parses, actionlint problems, actionlint version). Run before anything is written."""
    try:
        doc = yaml.safe_load(text)
        yaml_ok = isinstance(doc, dict) and bool(doc.get("jobs"))
    except yaml.YAMLError:
        yaml_ok = False
    problems, version = run_actionlint(text, CI_PATH)
    if not yaml_ok and not problems:
        problems = ["the workflow has no jobs"]
    return yaml_ok, problems, version
