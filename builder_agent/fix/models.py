"""The fix menu: exactly one action per diagnosis, as strict JSON, plus a short reason."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..render.configs import Overrides
from ..scan.deps import normalize


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PinPackage(_Strict):
    action: Literal["pin_package"]
    name: str
    version: str


class AddSystemPackage(_Strict):
    action: Literal["add_system_package"]
    name: str


class SetPythonVersion(_Strict):
    action: Literal["set_python_version"]
    version: str


class SetEnvVar(_Strict):
    action: Literal["set_env_var"]
    name: str
    value: str


class AddDependency(_Strict):
    action: Literal["add_dependency"]
    name: str


class GiveUp(_Strict):
    action: Literal["give_up"]
    reason: str


Fix = Annotated[Union[PinPackage, AddSystemPackage, SetPythonVersion, SetEnvVar, AddDependency, GiveUp],
                Field(discriminator="action")]


class Diagnosis(_Strict):
    fix: Fix
    reason: str = Field(max_length=400)        # short: why this fix, from the error


class DiagnosisWithAnalysis(_Strict):
    """For prompts that ask for step-by-step reasoning (Chain-of-Thought): "analysis" is the FIRST
    property, so the model writes it before choosing the fix."""
    analysis: str
    fix: Fix
    reason: str = Field(max_length=400)


def answer_model(prompt_text: str) -> type[Diagnosis] | type[DiagnosisWithAnalysis]:
    """The schema a prompt asks for: with "analysis" only if the prompt names that field."""
    return DiagnosisWithAnalysis if '"analysis"' in prompt_text else Diagnosis


PIP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VERSION_RE = re.compile(r"^\d+(\.\d+){1,2}$")
APT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9+.-]+$")
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PROTECTED_ENV = {"PATH", "HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "PYTHONHOME"}


class FixValidation(BaseModel):
    ok: bool
    reasons: list[str]
    diagnosis: Diagnosis | DiagnosisWithAnalysis | None = None


def validate_fix(answer: dict, applied: Overrides, installed: set[str],
                 model: type[Diagnosis] | type[DiagnosisWithAnalysis] = Diagnosis,
                 python_packages: set[str] = frozenset()) -> FixValidation:
    """Check one diagnosis. applied: fixes already in place; installed: pip names already installed;
    python_packages: names known to be Python packages (installed, imported, or known to the scanner)."""
    try:
        d = model.model_validate(answer)
    except ValidationError as e:
        return FixValidation(ok=False, reasons=[f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}"
                                                for err in e.errors()])
    f, reasons = d.fix, []
    bad = reasons.append
    if isinstance(d, DiagnosisWithAnalysis) and not d.analysis.strip():
        bad("analysis is empty: write the steps the prompt asks for before the fix")
    if isinstance(f, PinPackage):
        if not PIP_NAME_RE.match(f.name):
            bad(f"pin_package: {f.name!r} is not a valid package name")
        if not VERSION_RE.match(f.version):
            bad(f"pin_package: version {f.version!r} must look like 2.17 or 2.17.2")
        if applied.pins.get(f.name) == f.version or applied.pins.get(normalize(f.name)) == f.version:
            bad(f"pin_package {f.name}=={f.version} was already applied and the stage still failed")
    elif isinstance(f, AddSystemPackage):
        if not APT_NAME_RE.match(f.name):
            bad(f"add_system_package: {f.name!r} is not a valid Debian package name")
        if f.name in applied.apt or f.name == "make":
            bad(f"add_system_package {f.name} is already installed")
        elif normalize(f.name) in installed | python_packages:
            # fault-001 / diagnose_v2: "add_system_package mlflow" broke the build (no such Debian package)
            bad(f"add_system_package: {f.name} is a Python package (installed with pip), not a Debian "
                "package; to change its version use pin_package, to add it use add_dependency")
    elif isinstance(f, SetPythonVersion):
        m = re.fullmatch(r"3\.(\d{1,2})", f.version)
        if not m or int(m.group(1)) < 8:
            bad(f"set_python_version: {f.version!r} must be 3.N with N >= 8, e.g. 3.11")
        elif applied.python_version == f.version:
            bad(f"set_python_version {f.version} was already applied and the stage still failed")
    elif isinstance(f, SetEnvVar):
        if not ENV_NAME_RE.match(f.name):
            bad(f"set_env_var: {f.name!r} must be UPPER_CASE letters, digits and _")
        elif f.name in PROTECTED_ENV:
            bad(f"set_env_var: {f.name} can't be changed")
        elif applied.env.get(f.name) == f.value:
            bad(f"set_env_var {f.name}={f.value} was already applied and the stage still failed")
    elif isinstance(f, AddDependency):
        if not PIP_NAME_RE.match(f.name):
            bad(f"add_dependency: {f.name!r} is not a valid package name")
        elif normalize(f.name) in installed:
            bad(f"add_dependency: {f.name} is already installed; to change its version use pin_package")
    elif isinstance(f, GiveUp) and not f.reason.strip():
        bad("give_up needs a reason")
    return FixValidation(ok=not reasons, reasons=reasons, diagnosis=d if not reasons else None)


def apply_fix(applied: Overrides, fix) -> Overrides:
    """A new Overrides with the fix added (the old one is left unchanged)."""
    o = applied.model_copy(deep=True)
    if isinstance(fix, PinPackage):
        o.pins[fix.name] = fix.version
    elif isinstance(fix, AddSystemPackage):
        o.apt.append(fix.name)
    elif isinstance(fix, SetPythonVersion):
        o.python_version = fix.version
    elif isinstance(fix, SetEnvVar):
        o.env[fix.name] = fix.value
    elif isinstance(fix, AddDependency):
        o.dependencies.append(fix.name)
    return o


def describe(fix) -> str:
    if isinstance(fix, PinPackage):
        return f"pin_package {fix.name}=={fix.version}"
    if isinstance(fix, AddSystemPackage):
        return f"add_system_package {fix.name}"
    if isinstance(fix, SetPythonVersion):
        return f"set_python_version {fix.version}"
    if isinstance(fix, SetEnvVar):
        return f"set_env_var {fix.name}={fix.value}"
    if isinstance(fix, AddDependency):
        return f"add_dependency {fix.name}"
    return f"give_up ({fix.reason})"
