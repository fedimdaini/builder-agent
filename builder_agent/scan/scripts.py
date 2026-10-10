"""Facts about training: which functions train a model, and training scripts run from the command line.

For the script-training mode (decide/rules.py choose_train_mode): a script is a candidate when it has a
main guard or a CLI and trains a model somewhere in the module. Facts recorded for the slot prompt and
the validator: its argparse options, the sys.argv positions it reads, and where its model ends up
(joblib/pickle dumps, save_model calls, mlflow log_model calls). Code is parsed, never run.
Nothing here is shown in the scan summary, so the existing prompts don't change.
"""

from __future__ import annotations

import ast

from .models import CliArg, ModelOutput

TRAIN_MODULES = {"xgb", "xgboost", "lgb", "lightgbm"}


def trains(node: ast.AST) -> bool:
    """A call to .fit(...) or to xgb/lgb.train(...) anywhere in the node."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "fit":
                return True
            if n.func.attr == "train" and isinstance(n.func.value, ast.Name) and n.func.value.id in TRAIN_MODULES:
                return True
    return False


def returns_value(fn: ast.AST) -> bool:
    """The function itself (not a nested def) has a `return <value>`: it can hand a model back."""
    stack = list(getattr(fn, "body", []))
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Return) and n.value is not None:
            return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return False


def _const(node: ast.AST | None):
    return node.value if isinstance(node, ast.Constant) else None


def cli_args(tree: ast.Module) -> list[CliArg]:
    """argparse add_argument(...) calls: flags, required, default, type, help."""
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"):
            continue
        flags = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        kw = {k.arg: k.value for k in n.keywords if k.arg}
        positional = not flags[0].startswith("-")
        required = positional or _const(kw.get("required")) is True
        if "nargs" in kw and _const(kw["nargs"]) in ("?", "*"):
            required = False
        default = ast.unparse(kw["default"]) if "default" in kw else None
        typ = ast.unparse(kw["type"]) if "type" in kw else None
        out.append(CliArg(flags=flags, required=required and default is None, default=default, type=typ,
                          help=(_const(kw.get("help")) or None) if isinstance(_const(kw.get("help")), str) else None,
                          action=_const(kw.get("action")) if isinstance(_const(kw.get("action")), str) else None))
    return out


def argv_positions(tree: ast.Module) -> list[int]:
    """Indexes i of sys.argv[i] the module reads."""
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "argv"
                and isinstance(n.value.value, ast.Name) and n.value.value.id == "sys"
                and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, int)):
            out.add(n.slice.value)
    return sorted(i for i in out if i > 0)


def model_outputs(tree: ast.Module) -> list[ModelOutput]:
    """Where a model may be written: joblib/pickle dump, .save_model(path), mlflow.<flavor>.log_model."""
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        f, attr = n.func, n.func.attr
        owner = f.value.id if isinstance(f.value, ast.Name) else ast.unparse(f.value)
        if attr == "dump" and owner in ("joblib", "pickle") and len(n.args) >= 2:
            out.append(ModelOutput(kind=owner, target=ast.unparse(n.args[1]), line=n.lineno))
        elif attr == "save_model" and n.args and not owner.startswith("mlflow"):
            out.append(ModelOutput(kind="save_model", target=ast.unparse(n.args[0]), line=n.lineno))
        elif attr == "log_model" and owner.startswith("mlflow."):
            path = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords
                                                          if k.arg in ("artifact_path", "name")), None)
            out.append(ModelOutput(kind=owner, target=ast.unparse(path) if path is not None else "",
                                   line=n.lineno))
    return out
