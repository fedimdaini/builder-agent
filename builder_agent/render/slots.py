"""Slot answers for the adapter templates, the candidates they must come from, and the validator.

The LLM (later) answers SlotAnswers as strict JSON. Every answer is checked
against slot_candidates(ctx), which is built from scanner facts only. The
validator never raises: it returns reasons, so they can go back to the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..scan.code import module_target
from ..scan.layout import DATA_EXTS
from ..scan.models import FunctionSig, RepoContext

# Builder-supported menus (what the templates know how to render)
FLAVOR_DIST = {"xgboost": "xgboost", "sklearn": "scikit-learn", "lightgbm": "lightgbm", "catboost": "catboost"}
FLAVOR_INPUTS = {
    "xgboost": ["dmatrix", "numpy", "dataframe"],   # dmatrix: Booster API; numpy/dataframe: sklearn API
    "sklearn": ["numpy", "dataframe"],
    "lightgbm": ["numpy", "dataframe"],
    "catboost": ["numpy", "dataframe"],
}
# arg_map value tokens -> what the train adapter passes
# When the scanner sees which model API the train function uses, only these inputs fit its predict()
API_FLAVOR = {"xgboost-native": "xgboost", "xgboost-sklearn": "xgboost", "lightgbm-native": "lightgbm",
              "lightgbm-sklearn": "lightgbm", "catboost": "catboost"}
API_INPUTS = {
    "xgboost-native": ["dmatrix"],                 # Booster.predict() takes a DMatrix only
    "xgboost-sklearn": ["numpy", "dataframe"],
    "lightgbm-native": ["numpy", "dataframe"],
    "lightgbm-sklearn": ["numpy", "dataframe"],
    "catboost": ["numpy", "dataframe"],
}
ARG_TOKENS = {
    "$train_path": "path to the training CSV (a sampled temp copy when SAMPLE=1)",
    "$train_df": "training DataFrame, target included",
    "$X": "feature columns, by name, in training order",
    "$y": "target column (forward transform applied unless the function does it)",
    "$target_column": "target column name",
}


class _Strict(BaseModel):
    # strict: an LLM answering "yes" for a boolean or "3" for a number is rejected, not coerced
    model_config = ConfigDict(extra="forbid", strict=True)


class TrainSlots(_Strict):
    train_function: str                                  # "pkg.module:function"
    train_file: str                                      # data file, repo-relative
    arg_map: dict[str, str | int | float | bool]         # parameter -> token or literal


class EvaluateSlots(_Strict):
    eval_file: str
    group_column: str                                    # per_group metrics in eval.json


class DataSlots(_Strict):
    data_step: str                                       # "existing" or "pkg.module:function" (no args)


class SlotAnswers(_Strict):
    target_column: str
    target_transform: str                                # "none" or a forward transform the scanner found
    transform_inside_train_fn: bool                      # the repo's train function applies it itself
    model_flavor: Literal["xgboost", "sklearn", "lightgbm", "catboost"]
    model_input: Literal["dmatrix", "numpy", "dataframe"]
    train: TrainSlots
    evaluate: EvaluateSlots
    data: DataSlots


class SlotValidation(BaseModel):
    ok: bool
    reasons: list[str]
    slots: SlotAnswers | None = None


# --- candidates ------------------------------------------------------------

def _functions(ctx: RepoContext) -> dict[str, tuple[str, FunctionSig]]:
    """'pkg.module:function' -> (file, signature) for every entry-point function."""
    return {f"{module_target(e.path)}:{s.name}": (e.path, s) for e in ctx.entry_points for s in e.signatures}


def _params(sig: FunctionSig) -> tuple[list[str], list[str]]:
    """(all parameter names, required ones) — *args/**kwargs and the bare * are skipped."""
    names, required = [], []
    for p in sig.params:
        if p.startswith("*"):
            continue
        name = p.split("=")[0]
        names.append(name)
        if "=" not in p:
            required.append(name)
    return names, required


def slot_candidates(ctx: RepoContext) -> dict[str, Any]:
    """Everything an answer may be chosen from. Also what the LLM prompt will show."""
    ml = {f.name for f in ctx.frameworks if f.category == "ml"}
    flavors = [fl for fl, dist in FLAVOR_DIST.items() if dist in ml]
    funcs = _functions(ctx)
    data_roles = {e.path for e in ctx.entry_points if e.role in ("data", "features")}
    return {
        "target_column": [c.column for c in ctx.target_candidates],
        "target_transform": ["none"] + [t.forward for t in ctx.target_transforms],
        "model_flavor": flavors,
        # inputs valid for at least one detected flavor; the validator checks the pair
        "model_input": _model_inputs(ctx, flavors),
        "train_function": {name: sig.render() for name, (_, sig) in funcs.items()},
        "data_files": {d.path: d.columns for d in ctx.data_columns},
        "group_column": _group_columns(ctx),
        "arg_tokens": ARG_TOKENS,
        # a no-argument function in a data/feature script that builds the processed data
        "data_step": ["existing"] + [n for n, (path, s) in funcs.items()
                                     if path in data_roles and not _params(s)[1]],
    }


def _model_inputs(ctx: RepoContext, flavors: list[str]) -> list[str]:
    """Inputs that fit the APIs of the train functions; all inputs of the flavors if no API is known."""
    apis = [s.model_api for e in ctx.entry_points if e.role == "train" for s in e.signatures if s.model_api]
    if apis:
        return list(dict.fromkeys(i for api in apis for i in API_INPUTS[api]))
    return list(dict.fromkeys(i for fl in flavors for i in FLAVOR_INPUTS[fl]))


def _group_columns(ctx: RepoContext) -> list[str]:
    """Columns present in every data file that has the top target column, targets excluded."""
    target = next((c.column for c in ctx.target_candidates if c.in_data), None)
    headers = [d.columns for d in ctx.data_columns if target and target in d.columns]
    if not headers:
        return []
    targets = {c.column for c in ctx.target_candidates if c.score >= 3}
    common = set.intersection(*(set(h) for h in headers))
    return [c for c in headers[0] if c in common and c not in targets]


# --- gold files --------------------------------------------------------------

ALTERNATIVES_KEY = "_alternatives"


def load_answer(path: str | Path) -> tuple[dict, dict[str, list]]:
    """A slot answer file. Gold files may add {"_alternatives": {"train.train_file": [...]}}:
    other values accepted for a slot (lenient scoring). Returns (answer, alternatives)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    alternatives = data.pop(ALTERNATIVES_KEY, {})
    return data, alternatives


# --- validation ------------------------------------------------------------

MAX_EVAL_OVERLAP = 0.01   # share of eval rows that may also be training rows


def _applied(applied_to: list[str]) -> str:
    """["trip_duration", "via target_col"] -> "trip_duration (via target_col)"."""
    cols = [a for a in applied_to if not a.startswith("via ")]
    via = [a[4:] for a in applied_to if a.startswith("via ")]
    return (", ".join(cols) or "the target") + (f" (via {', '.join(via)})" if via else "")


def _shared_fraction(ctx: RepoContext, eval_file: str, train_file: str) -> float:
    """Estimated share of eval_file's rows that are also in train_file (0 if unknown)."""
    for o in ctx.data_overlaps:
        if (o.a, o.b) == (eval_file, train_file):
            return o.a_in_b
        if (o.a, o.b) == (train_file, eval_file):
            return o.b_in_a
    return 0.0


def _pydantic_reasons(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}" for err in e.errors()]


def validate_slots(ctx: RepoContext, answer: dict | SlotAnswers) -> SlotValidation:
    if isinstance(answer, SlotAnswers):
        s = answer
    else:
        try:
            s = SlotAnswers.model_validate(answer)
        except ValidationError as e:
            return SlotValidation(ok=False, reasons=_pydantic_reasons(e))

    cand = slot_candidates(ctx)
    files: dict[str, list[str]] = cand["data_files"]
    funcs = _functions(ctx)
    reasons: list[str] = []
    bad = reasons.append

    # target column
    if s.target_column not in cand["target_column"]:
        bad(f"target_column {s.target_column!r} is not a scanned target candidate "
            f"(candidates: {', '.join(cand['target_column'][:5]) or 'none'})")
    for slot, path in (("train.train_file", s.train.train_file), ("evaluate.eval_file", s.evaluate.eval_file)):
        if path not in files:
            bad(f"{slot} {path!r} is not a data file with a readable header "
                f"(choices: {', '.join(files) or 'none'})")
        elif s.target_column not in files[path]:
            bad(f"target_column {s.target_column!r} is not in the header of {path}")

    # features must exist in the eval file, by name
    if s.train.train_file in files and s.evaluate.eval_file in files:
        features = [c for c in files[s.train.train_file] if c != s.target_column]
        missing = [c for c in features if c not in files[s.evaluate.eval_file]]
        if missing:
            bad(f"{s.evaluate.eval_file} lacks training features: {', '.join(missing[:5])}")
        shared = _shared_fraction(ctx, s.evaluate.eval_file, s.train.train_file)
        if s.evaluate.eval_file == s.train.train_file or shared >= MAX_EVAL_OVERLAP:
            clean = [f for f, cols in files.items()
                     if f != s.train.train_file and s.target_column in cols
                     and all(c in cols for c in features)
                     and _shared_fraction(ctx, f, s.train.train_file) < MAX_EVAL_OVERLAP]
            how = ("is the training file" if s.evaluate.eval_file == s.train.train_file else
                   f"shares rows with train.train_file {s.train.train_file}: about {shared:.0%} of its rows "
                   "are in the training file (estimated from row-hash samples)")
            bad(f"evaluate.eval_file {s.evaluate.eval_file} {how}, so the metrics would be inflated. "
                f"Files with no shared rows: {', '.join(clean) or 'none'}")

    # transform
    transform = next((t for t in ctx.target_transforms if t.forward == s.target_transform), None)
    if s.target_transform not in cand["target_transform"]:
        bad(f"target_transform {s.target_transform!r} was not found in the code "
            f"(choices: {', '.join(cand['target_transform'])})")
    elif transform and s.target_column not in transform.applied_to \
            and not any(a.startswith("via ") for a in transform.applied_to):
        bad(f"{s.target_transform} is applied to {', '.join(transform.applied_to)}, not {s.target_column!r}")

    # model flavor / input
    if s.model_flavor not in cand["model_flavor"]:
        bad(f"model_flavor {s.model_flavor!r} needs {FLAVOR_DIST[s.model_flavor]}, which the repo doesn't use "
            f"(choices: {', '.join(cand['model_flavor']) or 'none'})")
    elif s.model_input not in FLAVOR_INPUTS[s.model_flavor]:
        bad(f"model_input {s.model_input!r} doesn't fit flavor {s.model_flavor} "
            f"(choices: {', '.join(FLAVOR_INPUTS[s.model_flavor])})")

    # train function and its arguments
    fn = funcs.get(s.train.train_function)
    if fn is None:
        bad(f"train.train_function {s.train.train_function!r} is not a repo function "
            f"(choices: {', '.join(cand['train_function']) or 'none'})")
    else:
        fn_file, sig = fn
        fn_name = s.train.train_function.split(":")[-1]
        names, required = _params(sig)
        arg_map = s.train.arg_map
        swapped = any(k in ARG_TOKENS or k.startswith("$") for k in arg_map) or (
            not any(k in names for k in arg_map) and any(v in names for v in arg_map.values()))
        if swapped:
            shape = ", ".join(f'"{p}": <token>' for p in (required or names))
            bad(f"train.arg_map is the wrong way round: its KEYS must be {fn_name}'s parameter names "
                f"({', '.join(names)}) and its VALUES tokens ({', '.join(ARG_TOKENS)}). You used "
                f"{', '.join(arg_map)} as keys. Expected shape: {{{shape}}}")
        unknown = [] if swapped else [k for k in arg_map if k not in names]
        if unknown:
            bad(f"train.arg_map: {fn_name} has no parameter {', '.join(unknown)}; "
                f"its parameters are {', '.join(names)}")
        missing = [] if swapped else [p for p in required if p not in arg_map]
        if missing:
            bad(f"train.arg_map: required parameter(s) not mapped: {', '.join(missing)}")
        targets = set(cand["target_column"]) | {s.target_column}
        for k, v in ({} if swapped else arg_map).items():
            if not isinstance(v, str):
                continue
            if v.startswith("$") and v not in ARG_TOKENS:
                bad(f"train.arg_map.{k}: unknown token {v!r} (choices: {', '.join(ARG_TOKENS)})")
            elif v in files or PurePosixPath(v).suffix.lower() in DATA_EXTS:
                # a literal path bypasses SAMPLE mode, which swaps in a sampled copy for $train_path
                bad(f"train.arg_map.{k}: {v!r} is a data file path; use \"$train_path\" (the file comes "
                    "from train.train_file, and SAMPLE=1 needs to swap in a sampled copy)")
            elif v in targets:
                bad(f"train.arg_map.{k}: {v!r} is the target column; use \"$target_column\"")
        tokens = {v for v in s.train.arg_map.values() if isinstance(v, str)}

        # the scanner saw which model API the function uses: flavor and input must fit it
        if sig.model_api:
            want_flavor, inputs = API_FLAVOR[sig.model_api], API_INPUTS[sig.model_api]
            evidence = ", ".join(sig.api_evidence)
            if s.model_flavor != want_flavor:
                bad(f"{s.train.train_function} uses {sig.model_api} ({evidence}); model_flavor must be {want_flavor!r}")
            elif s.model_input not in inputs:
                bad(f"{s.train.train_function} uses {sig.model_api} ({evidence}), whose predict() needs "
                    f"model_input {' or '.join(repr(i) for i in inputs)}, not {s.model_input!r}")

        seen = (f"the scanner saw {transform.forward} applied to {_applied(transform.applied_to)} "
                f"in {', '.join(transform.forward_files)}") if transform else ""
        if s.transform_inside_train_fn:
            if s.target_transform == "none":
                bad("transform_inside_train_fn is true but target_transform is 'none'")
            elif transform and fn_file not in transform.forward_files:
                bad(f"transform_inside_train_fn is true, but {seen}, not in {fn_file} where {fn_name} is: "
                    f"set it to false and map a parameter of {fn_name} to $y so the adapter applies it")
        elif s.target_transform != "none" and "$y" not in tokens:
            if transform and fn_file in transform.forward_files:
                bad(f"{s.target_transform} must be applied before training and {fn_name} gets no $y, but "
                    f"{seen}, the file of {fn_name}: if {fn_name} applies it itself, "
                    "set transform_inside_train_fn to true")
            else:
                bad(f"{s.target_transform} must be applied before training, but {fn_name} gets no $y to apply "
                    f"it to ({seen or 'the transform was not found in the code'}; it would train on the raw "
                    f"target): map a parameter of {fn_name} to $y, or set target_transform to 'none'")

    # data step
    if s.data.data_step != "existing" and s.data.data_step not in cand["data_step"]:
        bad(f"data.data_step {s.data.data_step!r} must be 'existing' or a repo function without "
            f"required parameters (choices: {', '.join(cand['data_step'])})")

    # group column
    if s.evaluate.eval_file in files:
        header = files[s.evaluate.eval_file]
        if s.evaluate.group_column not in header:
            bad(f"evaluate.group_column {s.evaluate.group_column!r} is not in the header of {s.evaluate.eval_file}")
        elif s.evaluate.group_column == s.target_column:
            bad("evaluate.group_column can't be the target column")

    return SlotValidation(ok=not reasons, reasons=reasons, slots=s if not reasons else None)
