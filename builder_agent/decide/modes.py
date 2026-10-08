"""Builder mode per pipeline artifact: create it (nothing there yet) or verify what exists.

The Builder never modifies an existing file (CLAUDE.md): an artifact that already exists in
any form is verified (findings report + patch, never applied), and only missing artifacts
are created as new files.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..scan.models import RepoContext
from .contracts import Contracts

Mode = Literal["create", "verify"]


class ArtifactMode(BaseModel):
    artifact: str
    mode: Mode
    existing: list[str]            # files that make it "verify"
    reason: str


def _artifact_patterns(c: Contracts) -> dict[str, list[str]]:
    """Artifact -> file patterns (repo-relative). Paths the contract names come from contracts.yaml."""
    compose = (c.model_extra or {}).get("compose", {})
    adapters = c.paths["adapters_dir"].strip("/")
    return {
        "Dockerfile": ["Dockerfile", "*/Dockerfile", "Dockerfile.*", "*.Dockerfile", "*/*.Dockerfile"],
        "docker-compose": [compose.get("base_file", "compose.base.yml"), "docker-compose*.yml",
                           "docker-compose*.yaml", "compose.yml", "compose.yaml", "*/docker-compose*.y*ml"],
        "Makefile": ["Makefile"],
        "CI workflow": [".github/workflows/*.yml", ".github/workflows/*.yaml"],
        "agents.yaml": ["agents.yaml"],
        "schema draft": ["schema.py"],
        "contract adapter": [f"{adapters}/serve.py"],
    }


def select_modes(ctx: RepoContext, c: Contracts) -> list[ArtifactMode]:
    root = Path(ctx.root)
    known = set(ctx.existing_pipeline_files)
    out = []
    for artifact, patterns in _artifact_patterns(c).items():
        hits = sorted(f for f in known if any(fnmatch.fnmatchcase(f, p) for p in patterns))
        # paths named by the contract (adapter, schema) aren't scanner patterns: look them up (read-only)
        hits += [p for p in patterns if "*" not in p and p not in hits and (root / p).is_file()]
        hits = sorted(set(hits))
        if hits:
            out.append(ArtifactMode(artifact=artifact, mode="verify", existing=hits,
                                    reason=f"exists: {', '.join(hits)}; the Builder only reports and proposes patches"))
        else:
            out.append(ArtifactMode(artifact=artifact, mode="create", existing=[],
                                    reason="not found; the Builder adds it as a new file"))
    return out


def modes_text(modes: list[ArtifactMode]) -> str:
    return "\n".join(f"  {m.artifact:<17} {m.mode:<7} {', '.join(m.existing) or '-'}" for m in modes)


__all__ = ["ArtifactMode", "select_modes", "modes_text"]
