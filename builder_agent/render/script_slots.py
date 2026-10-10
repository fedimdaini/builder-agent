"""Script-training mode: the slot answers for a repo whose model is trained by a script, and their validator.

The LLM reads the script (prompts/train_script_v1.md) and says how to run it: the command arguments,
and where the model ends up. The train adapter (templates/train_script.py.j2) runs the script inside an
MLflow run, then takes the model from that run or from the file the script wrote. Evaluation and
serving are the same as in function mode. The sandbox is the final check: the validator below only
catches answers that can't work.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ValidationError

from ..decide.rules import choose_task
from ..scan.layout import DATA_EXTS
from ..scan.models import RepoContext, TrainScript
from .slots import (FLAVOR_DIST, FLAVOR_INPUTS, MAX_EVAL_OVERLAP, TASKS, DataSlots, EvaluateSlots, SlotValidation,
                    _Strict, _pydantic_reasons, _shared_fraction, slot_candidates)

SCRIPT_TOKENS = {
    "$train_path": "path to the training CSV (a sampled temp copy when SAMPLE=1)",
    "$target_column": "target column name",
    "$model_dir": "an empty folder the script may write its model into (the adapter reads it from there)",
}


class ScriptRun(_Strict):
    script: str                                          # repo-relative path of the training script
    args: list[str]                                      # command-line arguments: tokens or literals
    train_file: str                                      # the data file the script trains on (features, SAMPLE)
    model_output: Literal["mlflow", "file"]              # the script logs the model to MLflow, or writes a file
    mlflow_artifact_path: str | None = None              # "mlflow": the artifact path it logs the model under
    model_file: str | None = None                        # "file": the path it writes, e.g. "$model_dir/model.pkl"


class ScriptSlotAnswers(_Strict):
    target_column: str
    target_transform: str                                # the script applies it itself; the adapter inverts it
    model_flavor: Literal["xgboost", "sklearn", "lightgbm", "catboost"]
    model_input: Literal["dmatrix", "numpy", "dataframe"]
    task: Literal["regression", "classification"]
    train_script: ScriptRun
    evaluate: EvaluateSlots
    data: DataSlots


def script_candidates(ctx: RepoContext) -> dict:
    cand = slot_candidates(ctx)
    return {"script": [t.path for t in ctx.train_scripts], "script_tokens": list(SCRIPT_TOKENS),
            "target_column": cand["target_column"], "target_transform": cand["target_transform"],
            "model_flavor": cand["model_flavor"], "model_input": cand["model_input"], "task": TASKS,
            "data_files": cand["data_files"], "group_column": cand["group_column"], "data_step": cand["data_step"]}


def validate_script_slots(ctx: RepoContext, answer: dict | ScriptSlotAnswers) -> SlotValidation:
    if isinstance(answer, ScriptSlotAnswers):
        s = answer
    else:
        try:
            s = ScriptSlotAnswers.model_validate(answer)
        except ValidationError as e:
            return SlotValidation(ok=False, reasons=_pydantic_reasons(e))
    cand = script_candidates(ctx)
    files: dict[str, list[str]] = cand["data_files"]
    reasons: list[str] = []
    bad = reasons.append
    run = s.train_script

    # the script and its arguments
    script: TrainScript | None = next((t for t in ctx.train_scripts if t.path == run.script), None)
    if script is None:
        bad(f"train_script.script {run.script!r} is not a training script the scanner found "
            f"(choices: {', '.join(cand['script']) or 'none'})")
    for a in run.args:
        if a.startswith("$") and a.split("/")[0] not in SCRIPT_TOKENS:      # "$model_dir/model.pkl" is fine
            bad(f"train_script.args: unknown token {a!r} (choices: {', '.join(SCRIPT_TOKENS)})")
        elif a in files or PurePosixPath(a).suffix.lower() in DATA_EXTS:
            bad(f"train_script.args: {a!r} is a data file path; use \"$train_path\" (SAMPLE=1 swaps in a "
                "sampled copy)")
        elif a in cand["target_column"]:
            bad(f"train_script.args: {a!r} is the target column; use \"$target_column\"")
    if script:
        for opt in script.cli_args:
            if opt.required and opt.flags[0].startswith("-") and not any(f in run.args for f in opt.flags):
                bad(f"train_script.args: {run.script} requires {' / '.join(opt.flags)}")
        positional = [o for o in script.cli_args if o.required and not o.flags[0].startswith("-")]
        flags = sum(1 for a in run.args if a.startswith("-"))
        if len(run.args) - 2 * flags < len(positional):
            bad(f"train_script.args: {run.script} takes {len(positional)} positional argument(s) "
                f"({', '.join(o.flags[0] for o in positional)})")

    # where the model ends up
    if run.model_output == "mlflow":
        if not run.mlflow_artifact_path:
            bad("train_script.mlflow_artifact_path is required with model_output 'mlflow'")
        if script and not script.imports_mlflow:
            bad(f"model_output 'mlflow', but {run.script} doesn't import mlflow")
        logged = [o.target.strip("'\"") for o in (script.outputs if script else []) if o.kind.startswith("mlflow.")]
        if script and run.mlflow_artifact_path and logged and run.mlflow_artifact_path not in logged:
            bad(f"train_script.mlflow_artifact_path {run.mlflow_artifact_path!r}: {run.script} logs the model "
                f"under {', '.join(repr(x) for x in logged)}")
    else:
        if not run.model_file:
            bad("train_script.model_file is required with model_output 'file'")
        elif "$model_dir" in run.model_file and "$model_dir" not in " ".join(run.args):
            bad("train_script.model_file is under $model_dir, but no argument passes $model_dir to the script")

    # data, target, model: the same rules as function mode
    for slot, path in (("train_script.train_file", run.train_file), ("evaluate.eval_file", s.evaluate.eval_file)):
        if path not in files:
            bad(f"{slot} {path!r} is not a data file with a readable header (choices: {', '.join(files) or 'none'})")
        elif s.target_column not in files[path]:
            bad(f"target_column {s.target_column!r} is not in the header of {path}")
    if s.target_column not in cand["target_column"]:
        bad(f"target_column {s.target_column!r} is not a scanned target candidate "
            f"(candidates: {', '.join(cand['target_column'][:5]) or 'none'})")
    if run.train_file in files and s.evaluate.eval_file in files:
        shared = _shared_fraction(ctx, s.evaluate.eval_file, run.train_file)
        if s.evaluate.eval_file == run.train_file or shared >= MAX_EVAL_OVERLAP:
            bad(f"evaluate.eval_file {s.evaluate.eval_file} is or overlaps the training file, so the metrics "
                "would be inflated")
    if s.target_transform not in cand["target_transform"]:
        bad(f"target_transform {s.target_transform!r} was not found in the code "
            f"(choices: {', '.join(cand['target_transform'])})")
    if s.model_flavor not in cand["model_flavor"]:
        bad(f"model_flavor {s.model_flavor!r} needs {FLAVOR_DIST[s.model_flavor]}, which the repo doesn't use "
            f"(choices: {', '.join(cand['model_flavor']) or 'none'})")
    elif s.model_input not in FLAVOR_INPUTS[s.model_flavor]:
        bad(f"model_input {s.model_input!r} doesn't fit flavor {s.model_flavor} "
            f"(choices: {', '.join(FLAVOR_INPUTS[s.model_flavor])})")
    if s.data.data_step != "existing" and s.data.data_step not in cand["data_step"]:
        bad(f"data.data_step {s.data.data_step!r} must be 'existing' or a repo function without required "
            f"parameters (choices: {', '.join(cand['data_step'])})")
    if s.evaluate.eval_file in files:
        if s.evaluate.group_column not in files[s.evaluate.eval_file]:
            bad(f"evaluate.group_column {s.evaluate.group_column!r} is not in the header of {s.evaluate.eval_file}")
        elif s.evaluate.group_column == s.target_column:
            bad("evaluate.group_column can't be the target column")
    choice, _ = choose_task(ctx)
    if choice.status == "decided" and s.task != choice.task:
        bad(f"task {s.task!r}: the scan signals decide {choice.task!r} ({'; '.join(choice.evidence[:3])})")
    if s.task == "classification" and s.target_transform != "none":
        bad(f"target_transform {s.target_transform!r} is for regression targets; with task 'classification' "
            "it must be 'none'")
    if s.task == "classification" and s.model_input == "dmatrix":
        bad("model_input 'dmatrix' (xgboost Booster API) isn't supported for classification")
    return SlotValidation(ok=not reasons, reasons=reasons, slots=None if reasons else s)
