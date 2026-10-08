"""The only way the Builder writes into a target repo.

Hard rule (CLAUDE.md): the Builder never modifies a file that existed in the target repo
before its run. It adds new files only, and in a git repo only on its own branch
builder/<date>, never on main. Problems in existing files go into a findings report and a
patch file, which are never applied.

    writer = RepoWriter(repo_root)        # creates and switches to builder/<date> if it's a git repo
    writer.write_new({"Dockerfile": text, "pipeline/train.py": text2})
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

BRANCH_PREFIX = "builder/"


class WriteRefused(Exception):
    """The Builder tried to change an existing file, or to write outside its branch."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def current_branch(root: Path) -> str | None:
    r = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_builder_branch(root: Path, today: date | None = None) -> str:
    """Create builder/<date> (or builder/<date>-N) from the current HEAD and switch to it.

    Never switches to an existing branch: that could change files in the working tree.
    Already on a builder/ branch: stay there.
    """
    branch = current_branch(root)
    if branch and branch.startswith(BRANCH_PREFIX):
        return branch
    base = f"{BRANCH_PREFIX}{(today or date.today()).isoformat()}"
    existing = set(_git(root, "branch", "--list", f"{base}*", "--format=%(refname:short)").stdout.split())
    name, n = base, 1
    while name in existing:
        n += 1
        name = f"{base}-{n}"
    r = _git(root, "switch", "-c", name)          # new branch from HEAD: the working tree is unchanged
    if r.returncode != 0:
        raise WriteRefused(f"could not create branch {name}: {r.stderr.strip()}")
    return name


class RepoWriter:
    """Writes new files into a target repo. Refuses any path that existed before this run."""

    def __init__(self, root: str | Path, today: date | None = None):
        self.root = Path(root)
        self.branch = ensure_builder_branch(self.root, today) if is_git_repo(self.root) else None
        self.created: set[str] = set()     # files this run created (it may update those)

    def _check(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise WriteRefused(f"{rel} is outside the target repo")
        if rel.startswith(".git/") or rel == ".git":
            raise WriteRefused(f"{rel} is inside .git")
        if path.exists() and rel not in self.created:
            raise WriteRefused(f"refusing to overwrite {rel}: it already exists in the target repo "
                               "(the Builder only adds new files; problems go into a findings report)")
        return path

    def write_new(self, files: dict[str, str]) -> list[str]:
        """Write all files or none. Paths are repo-relative with forward slashes."""
        if self.branch is not None:
            branch = current_branch(self.root)
            if branch != self.branch:
                raise WriteRefused(f"the target repo is on {branch!r}, not the Builder's branch {self.branch!r}")
        paths = {rel: self._check(rel) for rel in files}       # check everything before writing anything
        for rel, text in files.items():
            paths[rel].parent.mkdir(parents=True, exist_ok=True)
            paths[rel].write_text(text, encoding="utf-8", newline="\n")
            self.created.add(rel)
        return list(files)
