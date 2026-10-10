"""Rules that turn RepoContext + contracts.yaml into a BuildPlan. No LLM.

Each rule either decides (with the reason) or returns a NeedsLLM item that says
why it couldn't and which facts matter.
"""

from __future__ import annotations

import posixpath
import re

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from ..scan.deps import declared_names, normalize
from ..scan.models import EntryPoint, RepoContext, WebApp
from .contracts import Contracts, Endpoint
from .models import InstallPlan, MakeTarget, NeedsLLM, PythonChoice, ServingPlan, TaskChoice

# Exact-version hint sources, most trusted first. Builder-internal, not a team decision.
EXACT_PRIORITY = ["pipfile", "python-version-file", "runtime.txt", "pipfile-lock",
                  "conda", "dockerfile", "ci", "pyc-cache"]
CONSTRAINT_KINDS = {"requires-python", "python_requires", "poetry"}
MAJOR_MINOR_RE = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")

PIPENV_VERSION = "2023.12.1"   # known good in docs/reference-run-taxi.md
# Used only when the repo declares no Python version, and reported as an assumption. Builder-internal:
# a version the pinned MLflow client (render/configs.py) supports and the common ML wheels exist for.
DEFAULT_PYTHON = "3.11"

# Which scanner roles can implement each script target.
SCRIPT_ROLES = {"data": ("data", "features"), "train": ("train",), "evaluate": ("evaluate",)}


# --- python version --------------------------------------------------------

def _major_minor(value: str) -> str | None:
    m = MAJOR_MINOR_RE.match(value.strip())
    return f"{m.group(1)}.{m.group(2)}" if m else None


def _specifier(value: str) -> SpecifierSet | None:
    """PEP 440 specifier, also accepting Poetry's ^X.Y and ~X.Y and a bare X.Y."""
    parts = []
    for piece in (p.strip() for p in value.split(",") if p.strip()):
        if piece[0] in "^~" and not piece.startswith("~=") and (mm := _major_minor(piece[1:])):
            major, minor = map(int, mm.split("."))
            upper = f"{major + 1}" if piece[0] == "^" else f"{major}.{minor + 1}"
            parts += [f">={mm}", f"<{upper}"]
        elif mm := _major_minor(piece):
            parts.append(f"=={mm}.*")
        else:
            parts.append(piece)
    try:
        return SpecifierSet(",".join(parts))
    except InvalidSpecifier:
        return None


def _allows(spec: SpecifierSet, major_minor: str) -> bool:
    # any patch release of that minor version is fine
    return any(Version(f"{major_minor}.{p}") in spec for p in (0, 99))


def _lower_bound(spec: SpecifierSet) -> str | None:
    bounds = [s.version.removesuffix(".*") for s in spec if s.operator in (">=", "~=", "==")]
    mms = sorted({mm for b in bounds if (mm := _major_minor(b))}, key=Version)
    return mms[-1] if mms else None


def choose_python(ctx: RepoContext, c: Contracts) -> tuple[PythonChoice, NeedsLLM | None]:
    evidence = [f"{h.value} ({h.kind}: {h.source})" for h in ctx.python_version_hints]
    constraints = [(h, _specifier(h.value)) for h in ctx.python_version_hints if h.kind in CONSTRAINT_KINDS]
    exact = [(h, mm) for h in ctx.python_version_hints
             if h.kind not in CONSTRAINT_KINDS and (mm := _major_minor(h.value))]

    def fail(reason: str) -> tuple[PythonChoice, NeedsLLM]:
        return (PythonChoice(status="needs_llm", reason=reason, evidence=evidence),
                NeedsLLM(item="python", reason=reason, context=evidence))

    def assume(why: str) -> tuple[PythonChoice, None]:
        return PythonChoice(status="decided", version=DEFAULT_PYTHON, assumed=True, evidence=evidence,
                            reason=f"assumed: {why}; Builder default {DEFAULT_PYTHON} "
                                   "(set project.python_version in contracts.yaml to change it)"), None

    forced = c.project.python_version
    if forced != "auto":
        version, reason = forced, "forced by contracts.yaml project.python_version"
    elif exact:
        versions = {mm for _, mm in exact}
        if len(versions) == 1:
            version = versions.pop()
            kinds = sorted({h.kind for h, _ in exact}, key=EXACT_PRIORITY.index)
            reason = f"all exact hints agree ({', '.join(kinds)})"
        else:
            top = min(exact, key=lambda x: EXACT_PRIORITY.index(x[0].kind))[0].kind
            top_versions = {mm for h, mm in exact if h.kind == top}
            if len(top_versions) > 1:
                return fail(f"hints disagree and the most trusted source ({top}) lists "
                            f"several versions: {', '.join(sorted(top_versions, key=Version))}")
            version = top_versions.pop()
            others = sorted(versions - {version}, key=Version)
            reason = f"hints disagree; {top} wins by priority (also seen: {', '.join(others)})"
    elif constraints:
        bounds = [b for _, s in constraints if s and (b := _lower_bound(s))]
        if not bounds:
            if all(s and _allows(s, DEFAULT_PYTHON) for _, s in constraints):
                return assume("only version constraints without a lower bound ("
                              + ", ".join(h.value for h, _ in constraints) + f"), which allow {DEFAULT_PYTHON}")
            return fail("only version constraints without a lower bound: "
                        + ", ".join(h.value for h, _ in constraints))
        version = max(bounds, key=Version)
        reason = "no exact hint; lowest version allowed by the constraints"
    else:
        return assume("no Python version declared in the repo")

    for h, spec in constraints:
        if spec is None:
            return fail(f"can't parse constraint {h.value!r} from {h.source}")
        if not _allows(spec, version):
            return fail(f"{version} ({reason}) violates {h.value} from {h.source}")
    return PythonChoice(status="decided", version=version, reason=reason, evidence=evidence), None


# --- task ------------------------------------------------------------------

def task_evidence(ctx: RepoContext) -> list[tuple[str, str]]:
    """(task, evidence line) for every scan signal: model classes, metrics, objectives, target values."""
    out = [(s.task, f"{s.kind} {s.value} -> {s.task} ({', '.join(s.files[:2])}"
                    f"{', ...' if len(s.files) > 2 else ''})") for s in ctx.task_signals]
    v = ctx.target_values
    if v:
        kind = ("non-numeric" if not v.numeric else "integer" if v.integer else "decimal")
        line = (f"target values: {v.column} in {v.file}: {v.distinct} distinct {kind} values in {v.n} rows "
                f"(e.g. {', '.join(v.examples)})")
        out.append((v.task, line + (f" -> {v.task}" if v.task else " -> unclear")))
    return out


def choose_task(ctx: RepoContext) -> tuple[TaskChoice, NeedsLLM | None]:
    """Decided when every signal that points somewhere points the same way; needs_llm when there are
    none or they disagree."""
    evidence = task_evidence(ctx)
    tasks = {t for t, _ in evidence if t}
    lines = [line for _, line in evidence]
    if len(tasks) == 1:
        task = tasks.pop()
        return TaskChoice(status="decided", task=task, evidence=lines,
                          reason=f"all {sum(1 for t, _ in evidence if t)} task signal(s) agree"), None
    reason = ("no task signal (no model class, metric, objective or readable target values)" if not tasks
              else "task signals disagree")
    return (TaskChoice(status="needs_llm", reason=reason, evidence=lines),
            NeedsLLM(item="task", reason=reason, context=lines))


# --- serving ---------------------------------------------------------------

def _has_route(app: WebApp, ep: Endpoint) -> bool:
    return any(ep.method.upper() in r.methods and r.path == ep.path for r in app.routes)


def _imports_in(ctx: RepoContext, path: str) -> set[str]:
    return {t.distribution for t in ctx.third_party_imports if path in t.files}


def plan_serving(ctx: RepoContext, c: Contracts) -> tuple[ServingPlan, NeedsLLM | None]:
    port = c.serving.port
    predict = c.serving.endpoints.get("predict")
    apps = ctx.web_apps
    app_lines = [f"{a.framework} {a.target} routes: "
                 + ", ".join(f"{'/'.join(r.methods)} {r.path}" for r in a.routes) for a in apps]

    app_paths = {a.path for a in apps}
    app_modules = [_entry_line(e) for e in ctx.entry_points if e.path in app_paths]

    def fail(reason: str, **kw) -> tuple[ServingPlan, NeedsLLM]:
        context = app_lines + app_modules + kw.get("contract_gaps", []) + target_facts(ctx)
        return (ServingPlan(status="needs_llm", reason=reason, port=port, **kw),
                NeedsLLM(item="serving", reason=reason, context=context))

    mode = c.serving.mode
    if mode == "generate" or (mode == "auto" and not apps):
        why = "forced by contracts.yaml" if mode == "generate" else "repo has no Flask/FastAPI app"
        target = f"{c.paths['adapters_dir'].strip('/').replace('/', '.')}.serve:app"
        # the generated service is the Flask serve.py adapter template
        return ServingPlan(
            status="decided", mode="generate", framework="flask", app_target=target,
            port=port, command=f"gunicorn --bind 0.0.0.0:{port} {target}",
            reason=f"generate a service: {why}",
        ), None

    if not apps:
        return fail("contracts.yaml forces serving.mode=repo but the repo has no Flask/FastAPI app")
    matching = [a for a in apps if predict and _has_route(a, predict)]
    want = f"{predict.method} {predict.path}" if predict else "the predict endpoint"
    if len(matching) > 1:
        return fail(f"{len(matching)} apps serve {want}; can't tell which is the model service")
    if not matching:
        return fail(f"repo has web app(s) but none serves {want}")

    app = matching[0]
    if app.framework == "flask":
        command = f"gunicorn --bind 0.0.0.0:{port} {app.target}"
    else:
        command = f"uvicorn {app.target} --host 0.0.0.0 --port {port}"

    gaps = [f"no {ep.method} {ep.path} route (contract serving.endpoints.{name})"
            for name, ep in c.serving.endpoints.items() if not _has_route(app, ep)]
    if "mlflow" not in _imports_in(ctx, app.path):
        local = [r.literal for r in ctx.path_references if app.path in r.files]
        where = f"; loads {', '.join(local)} instead" if local else ""
        gaps.append(f"does not import mlflow, so it can't load the model from "
                    f"${c.serving.model_uri_env}{where}")

    extra = c.serving.model_extra or {}
    unverified = []
    if field := extra.get("response", {}).get("prediction_field"):
        unverified.append(f"response has field '{field}'")
    if status := extra.get("errors", {}).get("status"):
        unverified.append(f"errors return {status}, never 200")
    if extra.get("request", {}).get("format"):
        unverified.append("features are read by name, not position")

    plan = dict(mode="repo", framework=app.framework, app_path=app.path, app_target=app.target,
                repo_port=app.port, command=command, contract_gaps=gaps, unverified=unverified)
    if gaps:
        return fail(f"repo app {app.target} found, but it misses {len(gaps)} contract requirement(s)", **plan)
    return ServingPlan(status="decided", port=port, reason=f"repo app {app.target} serves {want}", **plan), None


# --- install ---------------------------------------------------------------

def plan_install(ctx: RepoContext, c: Contracts, serving: ServingPlan) -> tuple[InstallPlan, NeedsLLM | None]:
    root_deps = {d.kind: d for d in ctx.dependency_files if "/" not in d.path}
    declared = declared_names(ctx.dependency_files)
    req_files = [d for d in ctx.dependency_files
                 if d.kind == "requirements" and "/" not in d.path and d.packages
                 and not all(r.dev for r in d.packages)]

    extras: list[str] = []
    wanted = []
    if serving.mode == "generate":
        wanted += ["flask", "gunicorn"]
    elif serving.framework == "flask":
        wanted.append("gunicorn")
    elif serving.framework == "fastapi":
        wanted.append("uvicorn")
    if "train" in c.make_targets:
        wanted.append("mlflow")   # contract: train logs to MLflow
    tests = set(ctx.test_files)
    wanted += [t.distribution for t in ctx.undeclared_imports
               if not t.notebooks_only and not set(t.files) <= tests]
    for pkg in wanted:
        if normalize(pkg) not in declared and pkg not in extras:
            extras.append(pkg)

    def ok(tool, source, commands, reason):
        return InstallPlan(status="decided", tool=tool, source=source, commands=commands,
                           extra_packages=extras, reason=reason), None

    if "pipfile_lock" in root_deps and "pipfile" in root_deps:
        return ok("pipenv", "Pipfile.lock",
                  [f"pip install pipenv=={PIPENV_VERSION}", "pipenv install --deploy --system"],
                  "Pipfile.lock pins every package")
    if "poetry_lock" in root_deps and "pyproject" in root_deps:
        return ok("poetry", "poetry.lock",
                  ["pip install poetry", "poetry config virtualenvs.create false",
                   "poetry install --no-root --only main"], "poetry.lock pins every package")
    if req_files:
        f = next((d for d in req_files if d.path == "requirements.txt"), req_files[0])
        return ok("pip", f.path, [f"pip install -r {f.path}"], f"{f.path} lists {len(f.packages)} packages")
    if "pipfile" in root_deps:
        return ok("pipenv", "Pipfile",
                  [f"pip install pipenv=={PIPENV_VERSION}", "pipenv install --system --skip-lock"],
                  "Pipfile without a lock file")
    for kind in ("pyproject", "setup_py", "setup_cfg"):
        d = root_deps.get(kind)
        if d and d.packages:
            return ok("pip", d.path, ["pip install ."], f"{d.path} declares {len(d.packages)} packages")

    other = [f"{d.path} ({d.kind})" for d in ctx.dependency_files]
    reason = ("no rule for these dependency files: " + ", ".join(other)) if other \
        else "no dependency file found"
    return (InstallPlan(status="needs_llm", extra_packages=extras, reason=reason),
            NeedsLLM(item="install", reason=reason,
                     context=[f"third-party imports: {', '.join(t.distribution for t in ctx.third_party_imports)}"]))


# --- make targets ----------------------------------------------------------

def _paths_written(ctx: RepoContext, path: str) -> set[str]:
    """Repo-relative forms of the path literals in one file (as-is and relative to the file)."""
    out = set()
    for r in ctx.path_references:
        if path in r.files:
            out.add(posixpath.normpath(r.literal))
            out.add(posixpath.normpath(posixpath.join(posixpath.dirname(path), r.literal)))
    return out


def _requirement(name: str, e: EntryPoint, ctx: RepoContext, c: Contracts) -> str | None:
    """What the contract needs from this script that the code doesn't show, or None."""
    if name == "train":
        return None if "mlflow" in _imports_in(ctx, e.path) else "it does not import mlflow (contract: log the model to MLflow)"
    target = {"data": c.paths.get("processed_data"), "evaluate": c.paths.get("eval_report")}[name]
    if target and not any(p == target or p.startswith(target + "/") for p in _paths_written(ctx, e.path)):
        key = "processed_data" if name == "data" else "eval_report"
        return f"it never mentions {target} (contract paths.{key})"
    return None


def target_facts(ctx: RepoContext, n: int = 3) -> list[str]:
    """Target candidates and transforms, one line each, for needs_llm contexts."""
    lines = []
    if ctx.target_candidates:
        lines.append("target candidates: " + ", ".join(
            f"{c.column} (score {c.score}{'' if c.in_data else ', not in data headers'})"
            for c in ctx.target_candidates[:n]))
    for t in ctx.target_transforms:
        inverse = ", ".join(t.inverse_files) or "NOWHERE"
        lines.append(f"target transform: {t.forward} on {', '.join(t.applied_to)} "
                     f"in {', '.join(t.forward_files)}; inverse {t.inverse} in {inverse}")
    return lines


def _columns_line(ctx: RepoContext, under: str | None) -> list[str]:
    files = [d for d in ctx.data_columns if under and d.path.startswith(under + "/")]
    if not files:
        return []
    return [f"columns of {', '.join(d.path for d in files)}: {', '.join(files[0].columns)}"]


def _entry_line(e: EntryPoint) -> str:
    main = "__main__" if e.has_main_guard else "no __main__"
    if e.signatures:
        defs = f", defs: {'; '.join(s.render() for s in e.signatures)}"
    else:
        defs = f", defs: {', '.join(e.functions)}" if e.functions else ""
    return f"{e.path} [{e.role}] {main}{defs}"


def _script_target(name: str, desc: str, ctx: RepoContext, c: Contracts) -> tuple[MakeTarget, NeedsLLM | None]:
    roles = SCRIPT_ROLES[name]
    candidates = [e for e in ctx.entry_points if e.role in roles]
    runnable = [e for e in candidates if e.has_main_guard]
    context = [_entry_line(e) for e in candidates]

    def fail(reason: str, command: str | None = None) -> tuple[MakeTarget, NeedsLLM]:
        extra = []
        if name == "data":
            extra = [f"data dir {d.path}/ subdirs: {', '.join(d.subdirs)}" for d in ctx.data_dirs]
        elif name in {"train", "evaluate"}:
            extra = [f"model file {m.path}" for m in ctx.model_files] + target_facts(ctx)
        return (MakeTarget(name=name, description=desc, status="needs_llm", command=command, reason=reason),
                NeedsLLM(item=f"target:{name}", reason=reason, context=context + extra))

    if not candidates:
        return fail(f"no {'/'.join(roles)} script found")
    if len(runnable) > 1:
        return fail(f"{len(runnable)} runnable scripts: {', '.join(e.path for e in runnable)}")
    if not runnable:
        names = ", ".join(e.path for e in candidates)
        return fail(f"{names} {'has' if len(candidates) == 1 else 'have'} no __main__ block, "
                    "so there is nothing to run")
    e = runnable[0]
    command = f"python {e.path}"
    if missing := _requirement(name, e, ctx, c):
        return fail(f"{e.path} runs, but {missing}", command)
    return MakeTarget(name=name, description=desc, status="decided", command=command,
                      reason=f"{e.path} has a __main__ block and meets the contract"), None


def plan_targets(ctx: RepoContext, c: Contracts, install: InstallPlan,
                 serving: ServingPlan) -> tuple[list[MakeTarget], list[NeedsLLM]]:
    targets: list[MakeTarget] = []
    needs: list[NeedsLLM] = []
    for name, desc in c.make_targets.items():
        item: NeedsLLM | None = None
        if name in SCRIPT_ROLES:
            t, item = _script_target(name, desc, ctx, c)
        elif name == "install":
            cmds = install.commands + ([f"pip install {' '.join(install.extra_packages)}"]
                                       if install.extra_packages else [])
            t = MakeTarget(name=name, description=desc, status=install.status,
                           command=" && ".join(cmds) or None, reason=install.reason)
        elif name == "serve":
            t = MakeTarget(name=name, description=desc, status=serving.status,
                           command=serving.command, reason=serving.reason)
        elif name == "test":
            reason = "no sample /predict request: the scanner doesn't find one yet"
            t = MakeTarget(name=name, description=desc, status="needs_llm", reason=reason)
            processed = c.paths.get("processed_data")
            item = NeedsLLM(item="target:test", reason=reason, context=[
                f"path in code: {r.resolved}" for r in ctx.path_references
                if processed and r.resolved and r.resolved.startswith(processed + "/")][:4]
                + _columns_line(ctx, processed) + target_facts(ctx))
        elif name == "all":
            # order comes from the contract's own wording, e.g. "install, data, train, ... in order"
            words = re.findall(r"[A-Za-z_]+", desc)
            deps = [w for w in words if w in c.make_targets and w != "all"]
            t = MakeTarget(name=name, description=desc, status="decided", depends_on=deps,
                           command=f"$(MAKE) {' '.join(deps)}", reason="order from contracts.yaml")
        else:
            reason = f"no rule for make target '{name}' (new in contracts.yaml?)"
            t = MakeTarget(name=name, description=desc, status="needs_llm", reason=reason)
            item = NeedsLLM(item=f"target:{name}", reason=reason, context=[desc])
        targets.append(t)
        if item:
            needs.append(item)
    return targets, needs
