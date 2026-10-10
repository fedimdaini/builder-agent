"""AST analysis of Python files and notebook code cells. Code is parsed, never executed."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import scripts, targets, task
from .layout import DATA_EXTS, MODEL_EXTS
from .models import EntryRole, FunctionSig, PortHint, Route, TrainScript, WebApp

MAX_FUNCTIONS = 10
CLI_LIBS = ("typer", "click", "fire", "argparse")
HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
APP_CLASSES = {"Flask": "flask", "FastAPI": "fastapi"}
ROUTER_CLASSES = {"Blueprint", "APIRouter"}
PATH_LITERAL_RE = re.compile(r"^[\w.\-/\\]+\.([A-Za-z0-9]+)$")
AMBIGUOUS_EXTS = {".json", ".bin", ".txt", ".model", ".h5"}

ROLE_PATTERNS: list[tuple[EntryRole, re.Pattern]] = [
    ("serve", re.compile(r"serve|server|^app$|^api$|wsgi|asgi")),
    ("predict", re.compile(r"predict|inference|infer|scor")),
    ("evaluate", re.compile(r"eval")),
    ("train", re.compile(r"train")),
    ("data", re.compile(r"dataset|preprocess|prepare|ingest|download|etl|load_data|make_data")),
    ("features", re.compile(r"feature")),
    ("main", re.compile(r"^(main|run|cli|__main__)$")),
]


@dataclass
class ModuleFacts:
    path: str
    imports: set[str] = field(default_factory=set)       # absolute dotted names
    has_main_guard: bool = False
    cli: str | None = None
    functions: list[str] = field(default_factory=list)
    signatures: list[FunctionSig] = field(default_factory=list)   # all top-level defs
    target: targets.TargetFacts = field(default_factory=targets.TargetFacts)
    web_apps: list[WebApp] = field(default_factory=list)
    ports: list[PortHint] = field(default_factory=list)
    path_literals: set[str] = field(default_factory=set)
    task_signals: list = field(default_factory=list)     # scan/task.py TaskSignal, regression vs classification
    script: object = None                                # scan/models.py TrainScript, if the module is one


def role_for(path: str) -> EntryRole | None:
    stem = PurePosixPath(path).stem.lower()
    for role, pat in ROLE_PATTERNS:
        if pat.search(stem):
            return role
    return None


def module_target(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# --- helpers ---------------------------------------------------------------

def _is_main_guard(node: ast.AST) -> bool:
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    sides = [node.test.left, *node.test.comparators]
    has_name = any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
    has_main = any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return has_name and has_main


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _port_value(node: ast.AST) -> tuple[int | None, str | None]:
    """Port from a literal, int(...), os.environ.get("PORT", 8000) or os.getenv(...)."""
    if isinstance(node, ast.Constant) and str(node.value).isdigit():
        return int(node.value), None
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name == "int" and node.args:
            return _port_value(node.args[0])
        if name in {"get", "getenv"} and node.args and isinstance(node.args[0], ast.Constant):
            env = str(node.args[0].value)
            default = node.args[1] if len(node.args) > 1 else None
            port = _port_value(default)[0] if default is not None else None
            return port, env
    return None, None


def _str_items(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _route_from_decorator(dec: ast.AST, holders: set[str]) -> Route | None:
    if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name) and dec.func.value.id in holders):
        return None
    attr = dec.func.attr
    path = dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else None
    if path is None:
        path = next((k.value.value for k in dec.keywords
                     if k.arg == "path" and isinstance(k.value, ast.Constant)), None)
    if path is None:
        return None
    if attr in HTTP_METHODS:
        return Route(methods=[attr.upper()], path=str(path))
    if attr in {"route", "api_route"}:
        methods = next((_str_items(k.value) for k in dec.keywords if k.arg == "methods"), ["GET"])
        return Route(methods=[m.upper() for m in methods], path=str(path))
    return None


def _is_path_literal(s: str) -> bool:
    if len(s) > 200 or "://" in s:
        return False
    m = PATH_LITERAL_RE.match(s)
    if not m:
        return False
    ext = "." + m.group(1).lower()
    if ext in DATA_EXTS or ext in MODEL_EXTS:
        return True
    return ext in AMBIGUOUS_EXTS and ("/" in s or "\\" in s)


# --- analysis --------------------------------------------------------------

def analyze_tree(tree: ast.Module, path: str, *, is_notebook: bool = False) -> ModuleFacts:
    facts = ModuleFacts(path=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            facts.imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            facts.imports.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _is_path_literal(node.value):
            facts.path_literals.add(node.value.replace("\\", "/"))

    top = {i.split(".")[0] for i in facts.imports}
    facts.target = targets.collect(tree)
    facts.task_signals = task.code_signals(tree, path)
    if is_notebook:
        return facts

    facts.has_main_guard = any(_is_main_guard(n) for n in tree.body)
    facts.signatures = targets.signatures(tree)
    facts.functions = [s.name for s in facts.signatures][:MAX_FUNCTIONS]
    facts.cli = next((lib for lib in CLI_LIBS if lib in top), None)
    if facts.cli is None and any(isinstance(n, ast.Attribute) and n.attr == "argv"
                                 and isinstance(n.value, ast.Name) and n.value.id == "sys"
                                 for n in ast.walk(tree)):
        facts.cli = "sys.argv"
    if (facts.has_main_guard or facts.cli) and scripts.trains(tree):
        facts.script = TrainScript(path=path, has_main_guard=facts.has_main_guard, cli=facts.cli,
                                   cli_args=scripts.cli_args(tree), argv=scripts.argv_positions(tree),
                                   outputs=scripts.model_outputs(tree), imports_mlflow="mlflow" in top)

    # app = Flask(__name__) / app = FastAPI(); router = APIRouter()
    apps: dict[str, str] = {}
    holders: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            cls = _call_name(node.value.func)
            assigned = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in assigned if isinstance(t, ast.Name)]
            if cls in APP_CLASSES and APP_CLASSES[cls] in top:
                apps.update({n: APP_CLASSES[cls] for n in names})
                holders.update(names)
            elif cls in ROUTER_CLASSES:
                holders.update(names)

    routes: list[Route] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            routes += [r for d in node.decorator_list if (r := _route_from_decorator(d, holders))]

    # app.run(port=...) / uvicorn.run(app, port=...)
    port_by_app: dict[str, tuple[int | None, str | None]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run"):
            continue
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        if owner in apps:
            kind, pos = "app.run", 1          # Flask.run(host, port, ...)
        elif owner == "uvicorn":
            kind, pos = "uvicorn.run", 2      # uvicorn.run(app, host, port, ...)
        else:
            continue
        port_node = next((k.value for k in node.keywords if k.arg == "port"), None)
        if port_node is None and len(node.args) > pos:
            port_node = node.args[pos]
        if port_node is None:
            continue
        port, env = _port_value(port_node)
        if port is not None:
            facts.ports.append(PortHint(port=port, source=path, kind=kind))
        target_app = owner if owner in apps else next(iter(apps), None)
        if target_app:
            port_by_app[target_app] = (port, env)

    for var, fw in apps.items():
        port, env = port_by_app.get(var, (None, None))
        facts.web_apps.append(WebApp(
            path=path, framework=fw, app_var=var, target=f"{module_target(path)}:{var}",
            routes=routes, port=port, port_env_var=env,
        ))
    return facts


def analyze_python(source: str, path: str) -> ModuleFacts:
    """Raises SyntaxError (e.g. Python 2 code)."""
    return analyze_tree(ast.parse(source), path)


def _clean_cell(source: str) -> str:
    lines = []
    for line in source.splitlines():
        s = line.lstrip()
        if s.startswith(("%", "!")) or s.endswith("?"):
            lines.append("")  # IPython magic / shell / help: keep line numbers
        else:
            lines.append(line)
    return "\n".join(lines)


def analyze_notebook(raw: str, path: str) -> tuple[ModuleFacts, int, bool]:
    """Returns (facts, number of code cells, all cells parsed). Raises ValueError on bad JSON."""
    nb = json.loads(raw)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    body: list[ast.stmt] = []
    ok = True
    for c in cells:
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        try:
            body += ast.parse(_clean_cell(src)).body
        except SyntaxError:
            ok = False
    facts = analyze_tree(ast.Module(body=body, type_ignores=[]), path, is_notebook=True)
    return facts, len(cells), ok
