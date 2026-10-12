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


def tracked_changes(root: Path) -> list[str]:
    """Uncommitted changes to tracked files (git status lines); untracked files are not listed."""
    return _git(root, "status", "--porcelain", "--untracked-files=no").stdout.splitlines()


def worktree_state(root: Path) -> list[str]:
    """Every change git sees, untracked and ignored files included: to check a run left it unchanged."""
    return sorted(_git(root, "status", "--porcelain", "--untracked-files=all", "--ignored").stdout.splitlines())


def ignored_paths(root: Path, paths: list[str]) -> list[str]:
    """Paths the repo's .gitignore excludes (they couldn't be committed)."""
    if not paths:
        return []
    r = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"], input="\n".join(paths),
                       capture_output=True, text=True)
    return r.stdout.split()


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
        self.created_dirs: set[Path] = set()   # folders this run created (restore removes them if empty)

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

    def commit(self, message: str, author: str) -> str:
        """Commit exactly the files this run created, on the Builder's branch. Returns the commit sha.
        author: "Name <email>" (contracts.yaml integration.builder_commit_author)."""
        if self.branch is None or current_branch(self.root) != self.branch:
            raise WriteRefused(f"commit refused: not on the Builder's branch {self.branch!r}")
        name, _, email = author.partition("<")
        if not self.created or not name.strip() or not email.endswith(">"):
            raise WriteRefused(f"commit refused: no files, or author {author!r} is not 'Name <email>'")
        paths = sorted(self.created)
        for args in (["add", "--", *paths],
                     ["-c", f"user.name={name.strip()}", "-c", f"user.email={email[:-1]}", "commit", "--quiet",
                      "--no-verify", "-m", message, "--", *paths]):
            r = _git(self.root, *args)
            if r.returncode != 0:
                raise WriteRefused(f"git {args[0] if args[0] != '-c' else 'commit'} failed: {r.stderr.strip()}")
        return _git(self.root, "rev-parse", "HEAD").stdout.strip()

    def restore(self, original_branch: str) -> None:
        """Back to the branch the repo was on, whatever happened: files this run wrote but didn't commit
        are removed (they are its own new files), the switch back takes committed ones out of the working
        tree, and a Builder branch with no commit of its own is deleted. Never touches other files."""
        if self.branch is None:
            return
        tracked = set(_git(self.root, "ls-files", "--", *sorted(self.created)).stdout.split()) if self.created else set()
        for rel in sorted(self.created - tracked):
            (self.root / rel).unlink(missing_ok=True)
        _git(self.root, "reset", "--quiet")                          # unstage anything a failed commit staged
        if current_branch(self.root) != original_branch:
            r = _git(self.root, "switch", original_branch)
            if r.returncode != 0:
                raise WriteRefused(f"could not switch back to {original_branch}: {r.stderr.strip()}")
        mine = _git(self.root, "rev-parse", self.branch).stdout.strip()
        if mine and mine == _git(self.root, "rev-parse", "HEAD").stdout.strip():
            _git(self.root, "branch", "-D", self.branch)              # nothing was committed on it
        for d in sorted(self.created_dirs, key=lambda p: -len(p.parts)):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()                                             # empty folders this run created

    def write_new(self, files: dict[str, str]) -> list[str]:
        """Write all files or none. Paths are repo-relative with forward slashes."""
        if self.branch is not None:
            branch = current_branch(self.root)
            if branch != self.branch:
                raise WriteRefused(f"the target repo is on {branch!r}, not the Builder's branch {self.branch!r}")
        paths = {rel: self._check(rel) for rel in files}       # check everything before writing anything
        for rel, text in files.items():
            missing = [d for d in (paths[rel].parent, *paths[rel].parent.parents) if not d.exists()]
            paths[rel].parent.mkdir(parents=True, exist_ok=True)
            self.created_dirs.update(missing)
            paths[rel].write_text(text, encoding="utf-8", newline="\n")
            self.created.add(rel)
        return list(files)
