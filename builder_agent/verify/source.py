"""Read-only view of a target repo: its working tree, or the files at a git commit."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


class RepoSource:
    """Reads files; never writes. ref=None reads the working tree, ref="HEAD" reads that commit."""

    def __init__(self, root: str | Path, ref: str | None = None):
        self.root = Path(root)
        self.ref = ref
        self.commit = self._git("rev-parse", "--short", ref).strip() if ref else None
        self._files: list[str] | None = None

    @property
    def label(self) -> str:
        return f"git {self.ref} ({self.commit})" if self.ref else "working tree"

    def _git(self, *args: str) -> str:
        r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode("utf-8", errors="replace")

    def files(self) -> list[str]:
        if self._files is None:
            if self.ref:
                self._files = sorted(self._git("ls-tree", "-r", "--name-only", self.ref).splitlines())
            else:
                out = []
                for dirpath, dirnames, filenames in os.walk(self.root):
                    dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                    rel = Path(dirpath).relative_to(self.root)
                    out += [(rel / f).as_posix() for f in filenames]
                self._files = sorted(out)
        return self._files

    def read_bytes(self, rel: str) -> bytes:
        if self.ref:
            r = subprocess.run(["git", "-C", str(self.root), "show", f"{self.ref}:{rel}"], capture_output=True)
            if r.returncode != 0:
                raise FileNotFoundError(rel)
            return r.stdout
        return (self.root / rel).read_bytes()

    def read_text(self, rel: str) -> str:
        return self.read_bytes(rel).decode("utf-8", errors="replace")

    def exists(self, rel: str) -> bool:
        return rel in set(self.files())
