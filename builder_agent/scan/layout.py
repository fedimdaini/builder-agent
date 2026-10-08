"""File walk and folder-level facts: data/model folders, LFS, pipeline files, tests."""

from __future__ import annotations

import fnmatch
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .models import ArtifactDir, FileInfo, LfsInfo, LfsPointer

PRUNE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".ipynb_checkpoints", "site-packages", ".idea",
}
DATA_DIR_NAMES = {"data", "dataset", "datasets", "raw", "processed", "interim", "external", "input", "inputs"}
MODEL_DIR_NAMES = {"models", "model", "artifacts", "checkpoints", "saved_models", "weights"}
DATA_EXTS = {".csv", ".tsv", ".parquet", ".feather", ".jsonl", ".xlsx", ".xls",
             ".arrow", ".avro", ".orc", ".npy", ".npz"}
MODEL_EXTS = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".keras", ".onnx",
              ".cbm", ".ubj", ".safetensors", ".ckpt", ".sav"}
CODE_EXTS = {".py", ".pyc", ".ipynb"}
IGNORED_NAMES = {".gitkeep", ".gitattributes", ".gitignore", ".DS_Store", "README.md"}

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
LFS_SIZE_RE = re.compile(rb"^size (\d+)", re.M)
PYC_TAG_RE = re.compile(r"\.cpython-(\d)(\d+)\.pyc$")


@dataclass
class WalkResult:
    files: dict[str, int]   # rel posix path -> size
    pyc_names: list[str]    # rel paths of .pyc files in __pycache__ (for version hints)


def walk_repo(root: Path) -> WalkResult:
    files: dict[str, int] = {}
    pyc: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep = []
        for d in dirnames:
            if d == "__pycache__":
                rel = (here / d).relative_to(root).as_posix()
                pyc.extend(f"{rel}/{n}" for n in os.listdir(here / d) if n.endswith(".pyc"))
            elif d in PRUNE_DIRS or d.endswith(".egg-info") or (here / d / "pyvenv.cfg").exists():
                continue
            else:
                keep.append(d)
        dirnames[:] = sorted(keep)
        for f in sorted(filenames):
            p = here / f
            try:
                files[p.relative_to(root).as_posix()] = p.stat().st_size
            except OSError:
                continue
    return WalkResult(files=files, pyc_names=sorted(pyc))


def _suffix(path: str) -> str:
    return PurePosixPath(path).suffix.lower()


def _is_artifact(path: str) -> bool:
    p = PurePosixPath(path)
    return p.name not in IGNORED_NAMES and p.suffix.lower() not in CODE_EXTS


def _collect_dirs(files: dict[str, int], names: set[str], kind: str,
                  require_artifacts: bool) -> list[ArtifactDir]:
    # every directory path, then keep the outermost ones whose name matches
    all_dirs = sorted({str(PurePosixPath(f).parent) for f in files} - {"."})
    expanded = set()
    for d in all_dirs:
        parts = PurePosixPath(d).parts
        expanded.update("/".join(parts[: i + 1]) for i in range(len(parts)))
    candidates = [d for d in sorted(expanded) if PurePosixPath(d).name.lower() in names]
    outermost = [d for d in candidates
                 if not any(d.startswith(c + "/") for c in candidates if c != d)]

    result = []
    for d in outermost:
        inside = {f: s for f, s in files.items() if f.startswith(d + "/") and _is_artifact(f)}
        if require_artifacts and not inside:
            continue  # e.g. src/models with only .py files is code, not models
        subdirs = sorted({f[len(d) + 1:].split("/")[0] for f in files
                          if f.startswith(d + "/") and f[len(d) + 1:].count("/") >= 1})
        exts = Counter(_suffix(f) or PurePosixPath(f).name for f in inside)
        result.append(ArtifactDir(
            path=d, kind=kind, subdirs=subdirs, n_files=len(inside),
            total_bytes=sum(inside.values()), extensions=dict(exts.most_common()),
        ))
    return result


def find_data_dirs(files: dict[str, int]) -> list[ArtifactDir]:
    return _collect_dirs(files, DATA_DIR_NAMES, "data", require_artifacts=False)


def find_model_dirs(files: dict[str, int]) -> list[ArtifactDir]:
    return _collect_dirs(files, MODEL_DIR_NAMES, "model", require_artifacts=True)


def find_model_files(files: dict[str, int], model_dirs: list[ArtifactDir]) -> list[FileInfo]:
    out = []
    for f, s in files.items():
        in_model_dir = any(f.startswith(d.path + "/") for d in model_dirs) and _is_artifact(f)
        if in_model_dir or _suffix(f) in MODEL_EXTS:
            out.append(FileInfo(path=f, size_bytes=s))
    return out


def find_loose_data_files(files: dict[str, int], data_dirs: list[ArtifactDir]) -> list[FileInfo]:
    return [FileInfo(path=f, size_bytes=s) for f, s in files.items()
            if _suffix(f) in DATA_EXTS and not any(f.startswith(d.path + "/") for d in data_dirs)]


# --- Git LFS --------------------------------------------------------------

def _lfs_patterns(root: Path, files: dict[str, int]) -> list[tuple[str, str]]:
    """(base dir, pattern) for every `filter=lfs` line in any .gitattributes."""
    out = []
    for f in files:
        if PurePosixPath(f).name != ".gitattributes":
            continue
        base = str(PurePosixPath(f).parent)
        text = (root / f).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#") and "filter=lfs" in parts[1:]:
                out.append(("" if base == "." else base, parts[0]))
    return out


def _matches(path: str, base: str, pattern: str) -> bool:
    if base:
        if not path.startswith(base + "/"):
            return False
        path = path[len(base) + 1:]
    pattern = pattern.lstrip("/")
    if "/" not in pattern:  # gitattributes: no slash -> match basename at any depth
        return fnmatch.fnmatchcase(PurePosixPath(path).name, pattern)
    return fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(path, pattern + "/*")


def scan_lfs(root: Path, files: dict[str, int]) -> LfsInfo:
    patterns = _lfs_patterns(root, files)
    tracked = [f for f in files if any(_matches(f, b, p) for b, p in patterns)]
    pointers = []
    for f, size in files.items():
        if size > 1024:
            continue
        try:
            head = (root / f).read_bytes()
        except OSError:
            continue
        if head.startswith(LFS_POINTER_PREFIX):
            m = LFS_SIZE_RE.search(head)
            pointers.append(LfsPointer(path=f, object_size=int(m.group(1)) if m else None))
    return LfsInfo(
        patterns=[f"{(b + '/') if b else ''}.gitattributes: {p}" for b, p in patterns],
        tracked_files=tracked,
        pointer_files=pointers,
    )


# --- existing pipeline files and tests -------------------------------------

PIPELINE_FILE_PATTERNS = [
    "Dockerfile", "Dockerfile.*", "*.Dockerfile", ".dockerignore",
    "docker-compose*.yml", "docker-compose*.yaml", "compose.yml", "compose.yaml",
    "Makefile", "Procfile", "MLproject", "dvc.yaml", "Jenkinsfile",
    ".gitlab-ci.yml", "tox.ini", "noxfile.py", "agents.yaml",
]


def find_pipeline_files(files: dict[str, int]) -> list[str]:
    out = []
    for f in files:
        name = PurePosixPath(f).name
        if f.startswith(".github/workflows/") and _suffix(f) in {".yml", ".yaml"}:
            out.append(f)
        elif any(fnmatch.fnmatchcase(name, p) for p in PIPELINE_FILE_PATTERNS):
            out.append(f)
    return sorted(out)


def find_test_files(files: dict[str, int]) -> list[str]:
    out = []
    for f in files:
        p = PurePosixPath(f)
        if p.suffix != ".py":
            continue
        if (p.name.startswith("test_") or p.stem.endswith("_test")
                or p.name == "conftest.py" or "tests" in p.parts[:-1]):
            out.append(f)
    return out


def has_pytest_config(root: Path, files: dict[str, int]) -> bool:
    if "pytest.ini" in files or "conftest.py" in files:
        return True
    for name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"),
                         ("tox.ini", "[pytest]")):
        if name in files and marker in (root / name).read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def pyc_version_tags(pyc_names: list[str]) -> dict[str, str]:
    """'3.9' -> first .pyc path that shows it."""
    out: dict[str, str] = {}
    for n in pyc_names:
        m = PYC_TAG_RE.search(n)
        if m:
            out.setdefault(f"{m.group(1)}.{m.group(2)}", n)
    return out
