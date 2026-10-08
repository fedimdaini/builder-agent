"""Findings of verify mode. A finding never changes the repo: it carries a proposed fix as a diff."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning"]


class LineEdit(BaseModel):
    """Replace one line (1-based) and/or insert lines after it."""
    line: int
    replace: list[str] | None = None      # None: keep the line
    insert_after: list[str] = Field(default_factory=list)


class FileFix(BaseModel):
    path: str
    edits: list[LineEdit] = Field(default_factory=list)
    new_content: str | None = None        # set for a file that doesn't exist yet


class Finding(BaseModel):
    rule: str                              # e.g. floating-image-tag
    fault_id: str                          # fault case it comes from, e.g. fault-002
    severity: Severity
    file: str
    line: int
    message: str
    evidence: str                          # the offending line (or a summary)
    fix: str                               # what to do, in words
    fix_diff: str | None = None            # unified diff, or None when no safe change can be proposed
    fixes: list[FileFix] = Field(default_factory=list, exclude=True)   # machine form, used to build the patch


class VerifyReport(BaseModel):
    repo: str
    source: str                            # "git HEAD (e157717)" or "working tree"
    generated_at: str
    rules: dict[str, str]                  # rule -> fault id
    findings: list[Finding]
    patch_file: str | None = None
    patch_check: str | None = None         # result of git apply --check on a clean export

    @property
    def errors(self) -> int:
        return sum(f.severity == "error" for f in self.findings)

    @property
    def warnings(self) -> int:
        return sum(f.severity == "warning" for f in self.findings)
