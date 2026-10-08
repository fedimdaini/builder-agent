"""Slot answers for the adapter templates, the candidates they must come from, and the validator.

The LLM (later) answers SlotAnswers as strict JSON. Every answer is checked
against slot_candidates(ctx), which is built from scanner facts only. The
validator never raises: it returns reasons, so they can go back to the LLM.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..scan.code import module_target
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
    return {
        "target_column": [c.column for c in ctx.target_candidates],
        "target_transform": ["none"] + [t.forward for t in ctx.target_transforms],
        "model_flavor": flavors,
        "model_input": {fl: FLAVOR_INPUTS[fl] for fl in flavors},
        "train_function": {name: sig.render() for name, (_, sig) in funcs.items()},
        "data_files": {d.path: d.columns for d in ctx.data_columns},
        "arg_tokens": ARG_TOKENS,
        "data_step": ["existing"] + [n for n, (_, s) in funcs.items() if not _params(s)[1]],
    }


# --- validation ------------------------------------------------------------

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
        names, required = _params(sig)
        unknown = [k for k in s.train.arg_map if k not in names]
        if unknown:
            bad(f"train.arg_map: {sig.render()} has no parameter {', '.join(unknown)}")
        missing = [p for p in required if p not in s.train.arg_map]
        if missing:
            bad(f"train.arg_map: required parameter(s) not mapped: {', '.join(missing)}")
        for k, v in s.train.arg_map.items():
            if isinstance(v, str) and v.startswith("$") and v not in ARG_TOKENS:
                bad(f"train.arg_map.{k}: unknown token {v!r} (choices: {', '.join(ARG_TOKENS)})")
        tokens = {v for v in s.train.arg_map.values() if isinstance(v, str)}

        if s.transform_inside_train_fn:
            if s.target_transform == "none":
                bad("transform_inside_train_fn is true but target_transform is 'none'")
            elif transform and fn_file not in transform.forward_files:
                bad(f"transform_inside_train_fn is true but {s.target_transform} is not applied in {fn_file}")
        elif s.target_transform != "none" and "$y" not in tokens:
            bad(f"{s.target_transform} must be applied before training, but the function gets no $y "
                "to apply it to (it would train on the raw target)")

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
