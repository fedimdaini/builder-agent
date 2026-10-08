"""Scan layer: facts about a repository, no LLM.

    from builder_agent.scan import scan_repo
    ctx = scan_repo("../taxi-trip-regression")
    print(ctx.summary())
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

from . import code, deps, hints, layout, targets
from .known import FRAMEWORKS, IMPORT_TO_DIST
from .models import (
    EntryPoint, Framework, Notebook, ParseError, PathReference, RepoContext, ThirdPartyImport,
)

__all__ = ["scan_repo", "RepoContext"]

MAX_PY_BYTES = 1_000_000
MAX_NOTEBOOK_BYTES = 50_000_000
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _distribution(module: str) -> str:
    for key, dist in IMPORT_TO_DIST.items():
        if module == key or module.startswith(key + "."):
            return dist
    top = module.split(".")[0]
    return IMPORT_TO_DIST.get(top, top)


def _is_declared(dist: str, module: str, declared: set[str]) -> bool:
    candidates = {deps.normalize(dist), deps.normalize(module.split(".")[0])}
    # also accept variants such as tensorflow-cpu, opencv-python-headless
    return any(d in candidates or any(d.startswith(c + "-") for c in candidates) for d in declared)


def _local_names(py_files: list[str]) -> set[str]:
    names = set()
    for f in py_files:
        p = PurePosixPath(f)
        names.update(p.parts[:-1])
        names.add(p.stem)
    return names


def _resolve(root: Path, literal: str, sources: set[str]) -> dict:
    """Try the literal relative to the repo root, then to each file that mentions it."""
    bases = [root] + [root / PurePosixPath(s).parent for s in sorted(sources)]
    for base in bases:
        p = Path(os.path.normpath(base / literal))
        if p.exists() and p.is_relative_to(root):
            return {"exists": True, "resolved": p.relative_to(root).as_posix()}
    return {"exists": False, "resolved": None}


def scan_repo(path: str | Path) -> RepoContext:
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    walk = layout.walk_repo(root)
    files = walk.files
    errors: list[ParseError] = []

    dep_files, version_hints, dep_errors = deps.scan_dependencies(root, files)
    errors += [ParseError(path=p, error=e) for p, e in dep_errors]
    version_hints += hints.version_hints(root, files, walk.pyc_names)
    declared = deps.declared_names(dep_files)

    # --- code ---
    py_files = [f for f in files if f.endswith(".py")]
    test_files = set(layout.find_test_files(files))
    module_facts: list[code.ModuleFacts] = []
    for f in py_files:
        if files[f] > MAX_PY_BYTES:
            errors.append(ParseError(path=f, error=f"skipped: larger than {MAX_PY_BYTES} bytes"))
            continue
        try:
            module_facts.append(code.analyze_python((root / f).read_text(encoding="utf-8", errors="replace"), f))
        except SyntaxError as e:
            errors.append(ParseError(path=f, error=f"SyntaxError line {e.lineno}: {e.msg}"))

    notebooks: list[Notebook] = []
    notebook_facts: list[code.ModuleFacts] = []
    for f in (f for f in files if f.endswith(".ipynb")):
        nb = Notebook(path=f, size_bytes=files[f])
        if files[f] > MAX_NOTEBOOK_BYTES:
            nb.parse_ok = False
            errors.append(ParseError(path=f, error=f"skipped: larger than {MAX_NOTEBOOK_BYTES} bytes"))
        else:
            try:
                facts, nb.n_code_cells, nb.parse_ok = code.analyze_notebook(
                    (root / f).read_text(encoding="utf-8", errors="replace"), f)
                notebook_facts.append(facts)
                if not nb.parse_ok:
                    errors.append(ParseError(path=f, error="some code cells have syntax errors (skipped)"))
            except ValueError as e:
                nb.parse_ok = False
                errors.append(ParseError(path=f, error=f"invalid notebook JSON: {e}"))
        notebooks.append(nb)

    # --- imports ---
    local = _local_names(py_files)
    third: dict[str, set[str]] = defaultdict(set)    # top-level module -> files
    full_name: dict[str, str] = {}                    # top-level -> a full dotted name seen
    stdlib_seen, local_seen = set(), set()
    for facts in module_facts + notebook_facts:
        if PurePosixPath(facts.path).name == "setup.py":
            continue  # build-time imports (setuptools), not runtime dependencies
        for imp in facts.imports:
            top = imp.split(".")[0]
            if top in STDLIB:
                stdlib_seen.add(top)
            elif top in local:
                local_seen.add(top)
            else:
                third[top].add(facts.path)
                full_name.setdefault(top, imp)
    third_party = []
    for top in sorted(third, key=str.lower):
        dist = deps.normalize(_distribution(full_name[top]))
        third_party.append(ThirdPartyImport(
            module=top, distribution=dist, declared=_is_declared(dist, top, declared),
            files=sorted(third[top]),
        ))

    # --- frameworks ---
    imported_dists = {t.distribution for t in third_party}
    frameworks = []
    for dist, category in FRAMEWORKS.items():
        via = [v for v, present in (("import", dist in imported_dists), ("dependency", dist in declared)) if present]
        if via:
            frameworks.append(Framework(name=dist, category=category, via=via))
    order = {"ml": 0, "serving": 1, "tracking": 2, "data": 3}
    frameworks.sort(key=lambda fw: (order[fw.category], fw.name))

    # --- entry points, web apps, ports ---
    entry_points = []
    web_apps = []
    port_hints = []
    for facts in module_facts:
        web_apps += facts.web_apps
        port_hints += facts.ports
        name = PurePosixPath(facts.path).name
        if facts.path in test_files or name in {"setup.py", "__init__.py", "conftest.py"}:
            continue
        role = code.role_for(facts.path)
        if role is None and (facts.has_main_guard or facts.web_apps):
            role = "other"
        if role is not None:
            entry_points.append(EntryPoint(
                path=facts.path, role=role, has_main_guard=facts.has_main_guard,
                cli=facts.cli, functions=facts.functions,
                signatures=facts.signatures[:code.MAX_FUNCTIONS],
            ))
    port_hints += hints.infra_port_hints(root, files)

    # --- paths referenced in code ---
    refs: dict[str, set[str]] = defaultdict(set)
    for facts in module_facts + notebook_facts:
        for lit in facts.path_literals:
            refs[lit].add(facts.path)
    path_refs = [PathReference(literal=lit, files=sorted(srcs), **_resolve(root, lit, srcs))
                 for lit, srcs in sorted(refs.items())]

    # --- layout ---
    data_dirs = layout.find_data_dirs(files)
    model_dirs = layout.find_model_dirs(files)
    lfs = layout.scan_lfs(root, files)

    # --- slot candidates: data columns, target, target transforms ---
    data_columns = layout.read_headers(root, files, data_dirs, {p.path for p in lfs.pointer_files})
    sigs: dict = {}
    for facts in module_facts:
        if facts.path not in test_files:
            for sig in facts.signatures:
                sigs.setdefault(sig.name, sig)
    target_candidates, target_transforms = targets.aggregate(
        [(f.path, f.target) for f in module_facts + notebook_facts], sigs,
        {c for d in data_columns for c in d.columns},
    )

    return RepoContext(
        root=root.as_posix(),
        name=root.name,
        is_git_repo=(root / ".git").exists(),
        n_python_files=len(py_files),
        dependency_files=dep_files,
        python_version_hints=version_hints,
        third_party_imports=third_party,
        stdlib_modules=sorted(stdlib_seen),
        local_modules=sorted(local_seen),
        frameworks=frameworks,
        entry_points=entry_points,
        web_apps=web_apps,
        port_hints=port_hints,
        path_references=path_refs,
        data_dirs=data_dirs,
        model_dirs=model_dirs,
        model_files=layout.find_model_files(files, model_dirs),
        loose_data_files=layout.find_loose_data_files(files, data_dirs),
        notebooks=notebooks,
        lfs=lfs,
        existing_pipeline_files=layout.find_pipeline_files(files),
        test_files=sorted(test_files),
        has_pytest_config=layout.has_pytest_config(root, files),
        parse_errors=errors,
        data_columns=data_columns,
        target_candidates=target_candidates,
        target_transforms=target_transforms,
    )
