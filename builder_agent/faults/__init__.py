"""Fault generator: inject a known fault into what the Builder generates, run the sandbox, save the case.

    from builder_agent.faults import CATALOG, generate_case
    case = generate_case(CATALOG["mlflow_client_3_1_4"], repo, slots, expected)

A fault is an Overrides (the same settings the fix menu changes, plus two injection hooks no menu
action can set), so it never touches the repo's own files. A case is saved only if the fault really
breaks the sandbox AND its expected fix (one menu action, applied on top of the fault) makes every
stage pass again. Cases go to tests/faults/generated/<name>/case.json, in the shape of tests/faults/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from ..decide import load_contracts, plan_build
from ..fix import _installed, _python_packages
from ..fix.models import Diagnosis, apply_fix, describe, validate_fix
from ..render import validate_slots
from ..render.configs import Overrides, config_context
from ..sandbox import DEFAULT_CONTRACTS, DEFAULT_LOG, SandboxResult, run_sandbox, subprocess_runner
from ..scan import scan_repo

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "faults" / "generated"

FAMILIES = {
    "version_mismatch": "client/server version mismatch",
    "missing_dependency": "missing Python dependency",
    "incompatible_pin": "incompatible pin",
    "python_version": "wrong Python version",
    "env_var": "missing or wrong environment variable",
    "system_library": "missing system library",
}


class Variant(BaseModel):
    name: str                         # unique; the case folder name
    family: str                       # a key of FAMILIES
    title: str
    injection: Overrides              # the fault, as Builder settings
    expected_fix: dict                # one menu action that undoes it
    cause: str
    static_rule: str | None = None    # set if a static check could catch it before any run
    static_evidence: list[str] = []


CATALOG: dict[str, Variant] = {v.name: v for v in [
    Variant(name="mlflow_client_3_1_4", family="version_mismatch",
            title="MLflow client pinned to 3.1.4 against a 2.17.2 tracking server",
            injection=Overrides(pins={"mlflow": "3.1.4"}),
            expected_fix={"action": "pin_package", "name": "mlflow", "version": "2.17.2"},
            cause="The image installs mlflow==3.1.4; the tracking server is ghcr.io/mlflow/mlflow:v2.17.2. "
                  "The 3.x client calls endpoints the 2.x server doesn't have.",
            static_rule="the MLflow client version differs from the server image's tag",
            static_evidence=["Dockerfile: pip install ... mlflow==3.1.4",
                             "compose.base.yml: image: ghcr.io/mlflow/mlflow:v2.17.2"]),
    # not pandas, flask or gunicorn: the Builder's `pip install mlflow==...` reinstalls them as MLflow's
    # own dependencies, so uninstalling them before it breaks nothing (tried for pandas, 2026-10-09)
    Variant(name="xgboost_uninstalled", family="missing_dependency",
            title="xgboost missing from the image",
            injection=Overrides(post_install_commands=["pip uninstall -y xgboost"]),
            expected_fix={"action": "add_dependency", "name": "xgboost"},
            cause="xgboost is not installed in the image, but the repo's training code and the adapters import it."),
    Variant(name="numpy_2_0_2", family="incompatible_pin",
            title="numpy 2.0.2 installed over pandas 2.0.3, which is built for numpy 1.x",
            injection=Overrides(post_install_commands=["pip install numpy==2.0.2"]),
            expected_fix={"action": "pin_package", "name": "numpy", "version": "1.23.5"},
            cause="pandas 2.0.3's compiled extensions use the numpy 1.x C API; numpy 2 changed it. "
                  "Pipfile.lock pins numpy 1.23.5.",
            static_rule="an installed version differs from the one Pipfile.lock pins",
            static_evidence=["Pipfile.lock: numpy ==1.23.5", "Dockerfile: pip install numpy==2.0.2"]),
    Variant(name="python_3_12", family="python_version",
            title="Image on Python 3.12, the repo asks for 3.9",
            injection=Overrides(python_version="3.12"),
            expected_fix={"action": "set_python_version", "version": "3.9"},
            cause="The Pipfile requires python_version 3.9 and Pipfile.lock pins packages for it; "
                  "the image is python:3.12-slim.",
            static_rule="the base image's Python version differs from the repo's",
            static_evidence=["Pipfile: python_version = \"3.9\"", "Dockerfile: FROM python:3.12-slim"]),
    Variant(name="mlflow_uri_wrong_host", family="env_var",
            title="MLFLOW_TRACKING_URI names a host that doesn't exist",
            injection=Overrides(env={"MLFLOW_TRACKING_URI": "http://mlflow-server:5000"}),
            expected_fix={"action": "set_env_var", "name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"},
            cause="The tracking URI points at mlflow-server; the compose service is called mlflow.",
            static_rule="the tracking URI's host is not a compose service",
            static_evidence=["compose.base.yml: MLFLOW_TRACKING_URI: \"http://mlflow-server:5000\"",
                             "compose.base.yml services: no mlflow-server"]),
    Variant(name="libffi8_removed", family="system_library",
            title="libffi8 (needed by Python's ctypes) missing from the base image",
            injection=Overrides(pre_apt_commands=["dpkg --remove --force-depends libffi8"]),
            expected_fix={"action": "add_system_package", "name": "libffi8"},
            cause="The image lacks libffi.so.8, so Python's ctypes module (_ctypes) can't load. "
                  "Simulates a slimmer base image."),
]}


# --- error signature ----------------------------------------------------------------------------

_BUILDKIT = re.compile(r"^#\d+ \d+(\.\d+)? ")          # "#12 34.5 " before each docker build line
_NOISE = re.compile(r"^(#\d+ |make(\[\d+\])?: \*\*\*|failed to solve|ERROR: failed to |------|\s*>|Dockerfile:\d+)")
_ERROR = re.compile(r"\b\w*(Error|Exception)\b:|\bERROR:|\berror:|No such file|cannot open shared object", re.I)


def error_signature(tail: list[str]) -> str:
    """The last line that names an error, without docker build prefixes or make's own summary."""
    lines = [_BUILDKIT.sub("", line).strip() for line in tail]
    lines = [line for line in lines if line and not _NOISE.match(line)]
    found = [line for line in lines if _ERROR.search(line)]
    return (found or lines or [""])[-1][:200]


# --- one case -------------------------------------------------------------------------------------

class CaseResult(BaseModel):
    variant: str
    saved: str | None                 # path of case.json, or None
    reason: str                       # why it was (not) saved
    fault_attempt: SandboxResult
    fix_attempt: SandboxResult | None = None
    case: dict | None = None


def next_id(out: Path) -> str:
    ids = [json.loads(p.read_text(encoding="utf-8"))["id"] for p in out.glob("*/case.json")]
    return f"gen-{len(ids) + 1:03d}"


def check_expected_fix(v: Variant, repo, slots: dict, contracts_path) -> list[str]:
    """Would the fix loop accept the expected fix in the faulty state? Reasons if not."""
    ctx = scan_repo(repo)
    c = load_contracts(contracts_path)
    tctx = config_context(ctx, c, plan_build(ctx, c), validate_slots(ctx, slots).slots, None, v.injection)
    check = validate_fix({"fix": v.expected_fix, "reason": "expected fix"}, v.injection,
                         _installed(ctx, tctx), python_packages=_python_packages(ctx))
    return check.reasons


def generate_case(v: Variant, repo: str | Path, slots: dict, expected: dict | None = None,
                  out: str | Path = DEFAULT_OUT, contracts_path: str | Path = DEFAULT_CONTRACTS,
                  runner=subprocess_runner, sandbox_log: str | Path | None = DEFAULT_LOG,
                  reproduce: str | None = None) -> CaseResult:
    out = Path(out)
    if (out / v.name / "case.json").exists():
        raise FileExistsError(f"{out / v.name / 'case.json'} exists; a variant is generated once")
    if problems := check_expected_fix(v, repo, slots, contracts_path):
        raise ValueError(f"{v.name}: the fix loop would reject the expected fix: {problems}")

    def sandbox(o: Overrides) -> SandboxResult:
        return run_sandbox(repo, slots, expected=expected, contracts_path=contracts_path, log_path=sandbox_log,
                           runner=runner, overrides=o)

    r = sandbox(v.injection)
    if r.ok:
        return CaseResult(variant=v.name, saved=None, reason="the fault did not break the sandbox", fault_attempt=r)
    failed = next(s for s in r.stages if s.status == "failed")
    fix = Diagnosis.model_validate({"fix": v.expected_fix, "reason": "expected fix"}).fix
    f = sandbox(apply_fix(v.injection, fix))
    if not f.ok:
        return CaseResult(variant=v.name, saved=None, fault_attempt=r, fix_attempt=f,
                          reason=f"the expected fix ({describe(fix)}) did not pass: failed at {f.failed_stage}")

    case = {
        "id": next_id(out), "name": v.name, "title": v.title,
        "family": v.family, "variant": v.name, "generated": True,
        "split": None,                # later: "memory" or "test", by variant
        "injection": v.injection.model_dump(exclude_defaults=True),
        "found_by": {"attempt_id": r.attempt_id, "repo": r.repo, "date": r.started_at[:10],
                     "log": "logs/sandbox_attempts.jsonl"},
        "stage": failed.name, "command": failed.command, "exit_code": failed.exit_code,
        "stages": {s.name: s.status for s in r.stages},
        "error_signature": error_signature(failed.output_tail),
        "error_tail": failed.output_tail,
        "cause": {"summary": v.cause},
        "fix": {"summary": describe(fix), "verified_by": {"attempt_id": f.attempt_id, "ok": f.ok}},
        "expected_fix": v.expected_fix,
        "static_check": {"catchable_before_run": v.static_rule is not None,
                         "rule": v.static_rule or "none: only a run shows it",
                         "evidence": v.static_evidence or [f"sandbox stage {failed.name}"]},
        "reproduce": {"how": "Run the generator for this variant; it injects the fault into the Builder's "
                             "generated files only, then runs the sandbox.",
                      "command": reproduce or f"python -m builder_agent.faults <repo> --variant {v.name}"},
    }
    path = out / v.name / "case.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return CaseResult(variant=v.name, saved=str(path), reason="saved", fault_attempt=r, fix_attempt=f, case=case)
