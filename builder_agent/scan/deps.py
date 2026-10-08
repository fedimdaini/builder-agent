"""Dependency files and the Python version hints they carry. setup.py is parsed, never run."""

from __future__ import annotations

import ast
import configparser
import json
import re
import tomllib
from pathlib import Path, PurePosixPath

from .models import DependencyFile, Requirement, VersionHint

REQ_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")
EGG_RE = re.compile(r"#egg=([A-Za-z0-9._-]+)")
CONDA_LINE_RE = re.compile(r"^\s*-\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*([=<>!~].*)?$")


def normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(line: str, dev: bool = False) -> Requirement | None:
    line = line.split(" #", 1)[0].strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(("-r", "-c", "--")):
        return None
    if line.startswith("-e") or "://" in line:
        m = EGG_RE.search(line)
        return Requirement(name=m.group(1), spec=line, dev=dev) if m else None
    m = REQ_LINE_RE.match(line)
    if not m:
        return None
    spec = m.group(3).split(";", 1)[0].strip()
    return Requirement(name=m.group(1), spec=spec, dev=dev)


def _is_requirements(path: str) -> bool:
    p = PurePosixPath(path)
    if p.suffix not in {".txt", ".in"}:
        return False
    return p.name.startswith("requirements") or (len(p.parts) > 1 and p.parts[-2] == "requirements")


def _requirements(root: Path, rel: str) -> DependencyFile:
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    dev = "dev" in PurePosixPath(rel).name or "test" in PurePosixPath(rel).name
    pkgs = [r for line in text.splitlines() if (r := parse_requirement(line, dev))]
    return DependencyFile(path=rel, kind="requirements", packages=pkgs)


def _pipfile_spec(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("version") or value.get("git") or value.get("path") or "")
    return ""


def _pipfile(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile:
    data = tomllib.loads((root / rel).read_text(encoding="utf-8"))
    pkgs = [Requirement(name=n, spec=_pipfile_spec(v)) for n, v in data.get("packages", {}).items()]
    pkgs += [Requirement(name=n, spec=_pipfile_spec(v), dev=True)
             for n, v in data.get("dev-packages", {}).items()]
    req = data.get("requires", {})
    for key in ("python_full_version", "python_version"):
        if key in req:
            hints.append(VersionHint(value=str(req[key]), source=rel, kind="pipfile"))
    return DependencyFile(path=rel, kind="pipfile", packages=pkgs)


def _pipfile_lock(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile:
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    py = data.get("_meta", {}).get("requires", {}).get("python_version")
    if py:
        hints.append(VersionHint(value=str(py), source=rel, kind="pipfile-lock"))
    n = len(data.get("default", {}))
    n_dev = len(data.get("develop", {}))
    return DependencyFile(path=rel, kind="pipfile_lock", note=f"{n} pinned packages, {n_dev} dev")


def _pyproject(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile | None:
    data = tomllib.loads((root / rel).read_text(encoding="utf-8"))
    project = data.get("project", {})
    poetry = data.get("tool", {}).get("poetry", {})
    pkgs: list[Requirement] = [r for d in project.get("dependencies", []) if (r := parse_requirement(d))]
    for group, deps in project.get("optional-dependencies", {}).items():
        pkgs += [r for d in deps if (r := parse_requirement(d, dev=True))]
    for group, deps in data.get("dependency-groups", {}).items():
        pkgs += [r for d in deps if isinstance(d, str) and (r := parse_requirement(d, dev=True))]
    if rp := project.get("requires-python"):
        hints.append(VersionHint(value=rp, source=rel, kind="requires-python"))

    for section, dev in ((poetry.get("dependencies", {}), False),
                         (poetry.get("dev-dependencies", {}), True)):
        for name, v in section.items():
            if name.lower() == "python":
                hints.append(VersionHint(value=_pipfile_spec(v), source=rel, kind="poetry"))
            else:
                pkgs.append(Requirement(name=name, spec=_pipfile_spec(v), dev=dev))
    for group in poetry.get("group", {}).values():
        pkgs += [Requirement(name=n, spec=_pipfile_spec(v), dev=True)
                 for n, v in group.get("dependencies", {}).items()]

    if not pkgs and not project and not poetry:
        return DependencyFile(path=rel, kind="pyproject", note="no [project] or [tool.poetry] table")
    return DependencyFile(path=rel, kind="pyproject", packages=pkgs)


def _str_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None


def _setup_py(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile:
    tree = ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"))
    pkgs: list[Requirement] = []
    note = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "setup"):
            continue
        for kw in node.keywords:
            if kw.arg == "install_requires":
                items = _str_list(kw.value)
                if items is None:
                    note = "install_requires is computed at runtime (not a literal list)"
                else:
                    pkgs += [r for d in items if (r := parse_requirement(d))]
            elif kw.arg == "python_requires" and isinstance(kw.value, ast.Constant):
                hints.append(VersionHint(value=str(kw.value.value), source=rel, kind="python_requires"))
    if not pkgs and note is None:
        note = "no install_requires"
    return DependencyFile(path=rel, kind="setup_py", packages=pkgs, note=note)


def _setup_cfg(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile | None:
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(root / rel, encoding="utf-8")
    if not cfg.has_section("options"):
        return None
    raw = cfg.get("options", "install_requires", fallback="")
    pkgs = [r for line in raw.splitlines() if (r := parse_requirement(line))]
    if pr := cfg.get("options", "python_requires", fallback=None):
        hints.append(VersionHint(value=pr.strip(), source=rel, kind="python_requires"))
    return DependencyFile(path=rel, kind="setup_cfg", packages=pkgs)


def _conda(root: Path, rel: str, hints: list[VersionHint]) -> DependencyFile:
    pkgs = []
    for line in (root / rel).read_text(encoding="utf-8", errors="replace").splitlines():
        m = CONDA_LINE_RE.match(line)
        if not m:
            continue
        name, spec = m.group(1), (m.group(2) or "").strip()
        if name == "python":
            hints.append(VersionHint(value=spec.lstrip("="), source=rel, kind="conda"))
        elif name != "pip":
            pkgs.append(Requirement(name=name, spec=spec))
    return DependencyFile(path=rel, kind="conda", packages=pkgs)


def scan_dependencies(root: Path, files: dict[str, int]) -> tuple[list[DependencyFile], list[VersionHint], list[tuple[str, str]]]:
    """Returns (dependency files, version hints, parse errors)."""
    deps: list[DependencyFile] = []
    hints: list[VersionHint] = []
    errors: list[tuple[str, str]] = []
    for rel in files:
        name = PurePosixPath(rel).name
        try:
            if _is_requirements(rel):
                deps.append(_requirements(root, rel))
            elif name == "Pipfile":
                deps.append(_pipfile(root, rel, hints))
            elif name == "Pipfile.lock":
                deps.append(_pipfile_lock(root, rel, hints))
            elif name == "pyproject.toml":
                if d := _pyproject(root, rel, hints):
                    deps.append(d)
            elif name == "setup.py":
                deps.append(_setup_py(root, rel, hints))
            elif name == "setup.cfg":
                if d := _setup_cfg(root, rel, hints):
                    deps.append(d)
            elif name in {"environment.yml", "environment.yaml", "conda.yml", "conda.yaml"}:
                deps.append(_conda(root, rel, hints))
            elif name == "poetry.lock":
                deps.append(DependencyFile(path=rel, kind="poetry_lock"))
            elif name == "uv.lock":
                deps.append(DependencyFile(path=rel, kind="uv_lock"))
        except (SyntaxError, ValueError, tomllib.TOMLDecodeError, configparser.Error, OSError) as e:
            errors.append((rel, f"{type(e).__name__}: {e}"))
    return deps, hints, errors


def declared_names(deps: list[DependencyFile]) -> set[str]:
    return {normalize(r.name) for d in deps for r in d.packages}
