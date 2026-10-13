"""Where a sandbox failure comes from: a slot answer, or the environment. A rule, no LLM.

    origin = classify(stage, output_tail, slot_files)

1. Environment (the fix loop's job): the build or mlflow stage failed, or any traceback ends in an
   environment exception (a missing module or shared library, a connection or MLflow server error).
2. Slot-caused: the stage is data, train, evaluate or serve-side, and a traceback's innermost frame is in a
   file the slots decide (the generated adapters, the repo function or script the slots point to), with an
   exception that a slot value explains: a column (KeyError), the function's arguments (TypeError), the
   script's arguments (ValueError converting an argument, argparse usage errors), a data file
   (FileNotFoundError), the train function (AttributeError / cannot import name), the model's location.
3. Anything else: environment (conservative: the fix loop handled every failure before this rule existed).

Tracebacks are parsed as blocks ("Traceback (most recent call last):", its File frames, then the exception
line), so a log line that merely contains "Error" (git's warning, MLflow's INFO lines) never counts.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

WORKDIR = "/app/"                         # the image's working dir (render/configs.py WORKDIR)
ENV_STAGES = {"render", "build", "mlflow"}
SLOT_STAGES = {"data", "train", "evaluate", "serve", "health", "predict"}

FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|KeyError)|KeyError)(?::\s?(.*))?$")

# exception lines that point at the environment, wherever they come from
ENV_EXC = re.compile(r"ModuleNotFoundError|No module named|cannot open shared object|ConnectionError|"
                     r"ConnectionRefused|MaxRetryError|NewConnectionError|RestException|API request to|"
                     r"Failed to establish|Name or service not known|pkg_resources|DistributionNotFound", re.I)

# slot groups: exception pattern -> group, and the slots each group covers (for the "changed outside" check)
SLOT_RULES = [
    ("script_args", re.compile(r"^ValueError: (could not convert|invalid literal)|^SystemExit: 2$")),
    ("columns", re.compile(r"^KeyError\b|not in index|^ValueError: .*column", re.I)),
    ("function_args", re.compile(r"^TypeError: .*(unexpected keyword argument|missing \d+ required|"
                                 r"takes \d+ positional|got multiple values)")),
    ("data_files", re.compile(r"^FileNotFoundError\b")),
    ("function", re.compile(r"^AttributeError: module .* has no attribute|^ImportError: cannot import name")),
    ("model_output", re.compile(r"^(OSError|MlflowException|mlflow\.exceptions\.MlflowException): .*"
                                r"(model|artifact|No such file)", re.I)),
]
ARGPARSE_RE = re.compile(r"error: (the following arguments are required|unrecognized arguments|argument )")
GROUP_SLOTS = {
    "script_args": ["train_script.args"],
    "columns": ["target_column", "target_transform", "evaluate.group_column"],
    "function_args": ["train.arg_map", "transform_inside_train_fn"],
    "data_files": ["train.train_file", "train_script.train_file", "evaluate.eval_file", "data.data_step"],
    "function": ["train.train_function", "train.arg_map"],
    "model_output": ["train_script.model_output", "train_script.mlflow_artifact_path", "train_script.model_file",
                     "model_flavor", "model_input"],
}


class Origin(BaseModel):
    kind: str                             # "slot" or "environment"
    group: str | None = None              # slot group (GROUP_SLOTS key)
    frame: str | None = None              # "scripts/train.py line 10"
    code: str | None = None               # the source line at that frame, if printed
    exception: str | None = None
    reason: str

    def describe(self) -> str:
        where = f"{self.frame}" + (f": `{self.code}`" if self.code else "")
        return where or "the pipeline"


def tracebacks(tail: list[str]) -> list[tuple[list[tuple[str, int, str | None]], str]]:
    """[(frames [(path, line, source line)], exception line)] for each traceback block in the output."""
    out, frames, inside = [], [], False
    for i, line in enumerate(tail):
        if line.strip() == "Traceback (most recent call last):":
            frames, inside = [], True
            continue
        if not inside:
            continue
        if m := FRAME_RE.match(line):
            nxt = tail[i + 1].strip() if i + 1 < len(tail) and not FRAME_RE.match(tail[i + 1]) else None
            frames.append((m.group(1), int(m.group(2)), nxt))
        elif line and not line[0].isspace():
            out.append((frames, line.strip()))
            inside = False
    return out


def _rel(path: str) -> str:
    return path[len(WORKDIR):] if path.startswith(WORKDIR) else path


def classify(stage: str, tail: list[str], slot_files: set[str]) -> Origin:
    """slot_files: repo-relative files the slot answers decide (adapters, the train module or script, ...)."""
    if stage in ENV_STAGES:
        return Origin(kind="environment", reason=f"the {stage} stage failed: an environment problem")
    blocks = tracebacks(tail)
    for _, exc in blocks:
        if ENV_EXC.search(exc):
            return Origin(kind="environment", exception=exc, reason="the error is a missing module, library or "
                                                                    "service")
    if stage not in SLOT_STAGES:
        return Origin(kind="environment", reason=f"stage {stage}: not one the slots decide")
    if any(ARGPARSE_RE.search(line) for line in tail):
        usage = next((line for line in tail if ARGPARSE_RE.search(line)), "")
        return Origin(kind="slot", group="script_args", exception=usage.strip(),
                      reason="the training script rejected its command-line arguments")
    for frames, exc in blocks:
        ours = [f for f in frames if _rel(f[0]) in slot_files]
        if not ours:
            continue
        path, line, code = ours[-1]                      # the innermost frame in a file the slots decide
        for group, pattern in SLOT_RULES:
            if pattern.search(exc):
                return Origin(kind="slot", group=group, frame=f"{_rel(path)} line {line}", code=code, exception=exc,
                              reason=f"{exc.split(':')[0]} raised in {_rel(path)}, a file the slot answers decide")
    return Origin(kind="environment", reason="no traceback points at a slot answer: left to the fix loop")


def slot_files(slots, adapters_dir: str) -> set[str]:
    """The repo-relative files whose behaviour the slot answers decide."""
    files = {f"{adapters_dir}/{n}" for n in ("train.py", "evaluate.py", "serve.py", "data.py", "smoke_test.py")}
    script = getattr(slots, "train_script", None)
    if script:
        files.add(script.script)
    else:
        files.add(slots.train.train_function.split(":")[0].replace(".", "/") + ".py")
    step = getattr(slots.data, "data_step", "existing")
    if step and step != "existing":
        files.add(step.split(":")[0].replace(".", "/") + ".py")
    return files


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "arg_map":
            out |= flatten(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def changed_slots(before: dict, after: dict) -> list[str]:
    a, b = flatten(before), flatten(after)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def outside_group(changed: list[str], group: str | None) -> list[str]:
    allowed = set(GROUP_SLOTS.get(group or "", []))
    return [c for c in changed if c not in allowed]
