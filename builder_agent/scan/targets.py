"""Slot candidates for the LLM: target columns, target transforms, function signatures.

Facts only, ranked by evidence. Per file we record raw evidence; the aggregation
across files (call sites -> parameters, transform pairs) happens in aggregate().
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .models import FunctionSig, TargetCandidate, TargetTransform

# Variable/parameter names that usually hold the target: target, target_col, label, y, Y_train, y_val ...
TARGET_NAME_RE = re.compile(r"(^|_)(targets?|labels?)(_|$)|^y(_|$)", re.I)
# Parameters that take a target *column name*: target, target_col, label_col, y_column ...
# Stricter than TARGET_NAME_RE: plotting helpers take y="col" / label="text" for other reasons.
TARGET_PARAM_RE = re.compile(r"target|^(label|y)_?col(umn)?s?$", re.I)
# Model APIs, detected per function from the calls it makes: (module aliases, attribute) or class name
API_CALLS = {
    ("xgb", "train"): "xgboost-native", ("xgboost", "train"): "xgboost-native",
    ("xgb", "Booster"): "xgboost-native", ("xgboost", "Booster"): "xgboost-native",
    ("xgb", "DMatrix"): "xgboost-native", ("xgboost", "DMatrix"): "xgboost-native",
    ("lgb", "train"): "lightgbm-native", ("lightgbm", "train"): "lightgbm-native",
}
API_CLASSES = {
    "XGBRegressor": "xgboost-sklearn", "XGBClassifier": "xgboost-sklearn", "XGBRanker": "xgboost-sklearn",
    "LGBMRegressor": "lightgbm-sklearn", "LGBMClassifier": "lightgbm-sklearn",
    "CatBoostRegressor": "catboost", "CatBoostClassifier": "catboost",
}
# forward transform -> inverse
TRANSFORM_PAIRS = {"log1p": "expm1", "log": "exp"}
TRANSFORM_MODULES = {"np", "numpy", "math", "torch", "tf"}
# evidence kind -> weight (drop() is weak: many non-target columns are dropped too)
WEIGHTS = {"label": 3, "fit": 3, "y-assign": 3, "target-name": 3, "target-param": 3, "drop": 1}
RESOLVE_DEPTH = 3
MAX_DEFAULT_LEN = 20


@dataclass
class TargetFacts:
    evidence: list[tuple[str, str]] = field(default_factory=list)         # (column, kind)
    calls: list[tuple[str, list[str | None], dict[str, str]]] = field(default_factory=list)
    transforms: list[tuple[str, set[str], set[str]]] = field(default_factory=list)  # (fn, columns, names)


# --- per file --------------------------------------------------------------

def signatures(tree: ast.Module) -> list[FunctionSig]:
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params: list[str] = []
        positional = a.posonlyargs + a.args
        defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
        for arg, default in zip(positional, defaults):
            params.append(arg.arg + (f"={_short(default)}" if default is not None else ""))
        if a.vararg:
            params.append(f"*{a.vararg.arg}")
        elif a.kwonlyargs:
            params.append("*")
        for arg, default in zip(a.kwonlyargs, a.kw_defaults):
            params.append(arg.arg + (f"={_short(default)}" if default is not None else ""))
        if a.kwarg:
            params.append(f"**{a.kwarg.arg}")
        doc = ast.get_docstring(node)
        api, evidence = model_api(node)
        out.append(FunctionSig(name=node.name, params=params,
                               doc=doc.strip().splitlines()[0][:80] if doc else None,
                               model_api=api, api_evidence=evidence))
    return out


def model_api(fn: ast.AST) -> tuple[str | None, list[str]]:
    """Which model API a function uses, from the calls in its body (first match wins)."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in API_CALLS:
            found.append((API_CALLS[(f.value.id, f.attr)], f"{f.value.id}.{f.attr}"))
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in API_CLASSES:
            found.append((API_CLASSES[name], name))
    if not found:
        return None, []
    api = found[0][0]
    return api, sorted({e for a, e in found if a == api})


def _short(node: ast.AST) -> str:
    s = ast.unparse(node)
    return s if len(s) <= MAX_DEFAULT_LEN else s[: MAX_DEFAULT_LEN - 3] + "..."


def _str_consts(node: ast.AST | None) -> list[str]:
    """'a' or ['a', 'b'] -> list of strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _subscript_cols(expr: ast.AST, assigns: dict[str, list[ast.AST]]) -> set[str]:
    """df["col"], and df[TARGET] where TARGET = "col"."""
    cols = set()
    for n in ast.walk(expr):
        if not isinstance(n, ast.Subscript):
            continue
        if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            cols.add(n.slice.value)
        elif isinstance(n.slice, ast.Name):
            cols.update(v.value for v in assigns.get(n.slice.id, [])
                        if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return cols


def _assignments(tree: ast.AST) -> dict[str, list[ast.AST]]:
    out: dict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id].append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            out[node.target.id].append(node.value)
    return out


def _resolve(expr: ast.AST, assigns: dict[str, list[ast.AST]], depth: int = RESOLVE_DEPTH,
             seen: frozenset[str] = frozenset()) -> tuple[set[str], set[str]]:
    """Columns (df['col']) and variable names an expression depends on, following assignments."""
    cols, names = _subscript_cols(expr, assigns), set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            names.add(n.id)
            if depth and n.id not in seen:
                for value in assigns.get(n.id, []):
                    c, nm = _resolve(value, assigns, depth - 1, seen | {n.id})
                    cols |= c
                    names |= nm
    return cols, names


def _transform_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute) and func.attr in {*TRANSFORM_PAIRS, *TRANSFORM_PAIRS.values()}:
        if isinstance(func.value, ast.Name) and func.value.id in TRANSFORM_MODULES:
            return func.attr
    if isinstance(func, ast.Name) and func.id in {"log1p", "expm1"}:  # from numpy import log1p
        return func.id
    return None


def collect(tree: ast.Module) -> TargetFacts:
    facts = TargetFacts()
    assigns = _assignments(tree)
    add = facts.evidence.append

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "drop":
                cols = _str_consts(node.args[0]) if node.args else []
                cols += [c for k in node.keywords if k.arg == "columns" for c in _str_consts(k.value)]
                for c in cols:
                    add((c, "drop"))
            for k in node.keywords:
                # label=<variable>; a literal label='Train' is a plot label, not a target
                if k.arg == "label" and not isinstance(k.value, ast.Constant):
                    for c in _resolve(k.value, assigns)[0]:
                        add((c, "label"))
            if name == "fit" and len(node.args) >= 2:
                for c in _resolve(node.args[1], assigns)[0]:
                    add((c, "fit"))
            if fn := _transform_name(func):
                if node.args:
                    cols, names = _resolve(node.args[0], assigns)
                    facts.transforms.append((fn, cols, names))
            if name:
                pos = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
                       for a in node.args]
                kws = {k.arg: k.value.value for k in node.keywords
                       if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)}
                if any(pos) or kws:
                    facts.calls.append((name, pos, kws))

        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(TARGET_NAME_RE.search(n) for n in names):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    add((node.value.value, "target-name"))      # TARGET = "price"
                else:
                    for c in _subscript_cols(node.value, assigns):
                        add((c, "y-assign"))                     # y = df["price"]

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            positional = a.posonlyargs + a.args
            pairs = list(zip(positional[len(positional) - len(a.defaults):], a.defaults))
            pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if TARGET_PARAM_RE.search(arg.arg):
                    for c in _str_consts(default):
                        add((c, "target-param"))                 # def train(target_col="price")
    return facts


# --- across files ----------------------------------------------------------

def aggregate(per_file: list[tuple[str, TargetFacts]], sigs: dict[str, FunctionSig],
              data_columns: set[str]) -> tuple[list[TargetCandidate], list[TargetTransform]]:
    """per_file: (path, facts) for .py files and notebooks. sigs: local function name -> signature."""
    evidence: dict[str, list[tuple[str, str]]] = defaultdict(list)   # column -> [(kind, path)]
    for path, facts in per_file:
        for col, kind in facts.evidence:
            evidence[col].append((kind, path))
        # train_xgboost(path, 'trip_duration') where the parameter is called target_col
        for callee, pos, kws in facts.calls:
            sig = sigs.get(callee)
            if not sig:
                continue
            params = [p.split("=")[0] for p in sig.params if not p.startswith("*")]
            bound = {params[i]: v for i, v in enumerate(pos) if i < len(params) and v is not None}
            bound |= kws
            for param, value in bound.items():
                if TARGET_PARAM_RE.search(param):
                    evidence[value].append(("target-param", path))

    candidates = []
    for col, ev in evidence.items():
        if not col.strip():
            continue  # '' and ' ' are never column names
        kinds = sorted({k for k, _ in ev})
        if kinds == ["drop"] and data_columns and col not in data_columns:
            continue  # dropped column that isn't even in the data headers: not a target
        candidates.append(TargetCandidate(
            column=col,
            score=sum(WEIGHTS[k] for k, _ in ev),
            in_data=col in data_columns,
            evidence=sorted({f"{k}: {p}" for k, p in ev}),
        ))
    candidates.sort(key=lambda c: (not c.in_data, -c.score, c.column))

    top = {c.column for c in candidates if c.score >= WEIGHTS["label"]}
    transforms = []
    for fwd, inv in TRANSFORM_PAIRS.items():
        fwd_files, inv_files, on = set(), set(), set()
        for path, facts in per_file:
            for fn, cols, names in facts.transforms:
                if fn == fwd:
                    # the column if it resolves, else the target-like variable it goes through
                    hit = (cols & top) or {f"via {n}" for n in names if TARGET_NAME_RE.search(n)}
                    if hit:
                        fwd_files.add(path)
                        on |= hit
                elif fn == inv:
                    inv_files.add(path)
        if fwd_files:
            transforms.append(TargetTransform(
                forward=fwd, inverse=inv, applied_to=sorted(on),
                forward_files=sorted(fwd_files), inverse_files=sorted(inv_files),
            ))
    return candidates, transforms
