"""Verify mode, static layer: check a repo's existing pipeline files against the fault cases.

    from builder_agent.verify import verify_repo
    report = verify_repo("C:/dev/mlops-zoomcamp-project", ref="HEAD", out_dir="experiments/verify/...")

Read-only on the target repo. Output: findings.json, findings.md and fixes.patch in out_dir
(builder-agent's own folder by default), never in the target. The patch is checked with
`git apply --check` against a clean export of the same commit, and never applied.
"""

from __future__ import annotations

import difflib
import io
import json
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .models import FileFix, Finding, LineEdit, VerifyReport
from .rules import RULES, run_rules
from .source import RepoSource

__all__ = ["verify_repo", "RepoSource", "VerifyReport", "Finding"]


# --- diffs -------------------------------------------------------------------------------

def _apply(lines: list[str], edits: list[LineEdit]) -> list[str]:
    by_line: dict[int, LineEdit] = {}
    for e in edits:                       # findings that fix the same line the same way share one edit
        prev = by_line.get(e.line)
        if prev is None:
            by_line[e.line] = e.model_copy(deep=True)
        else:
            if prev.replace is None:
                prev.replace = e.replace
            prev.insert_after += [x for x in e.insert_after if x not in prev.insert_after]
    out = []
    for i, line in enumerate(lines, 1):
        e = by_line.get(i)
        if e is None:
            out.append(line)
            continue
        ending = "\n" if line.endswith("\n") else ""
        out += [r + "\n" if not r.endswith("\n") else r for r in e.replace] if e.replace is not None else [line]
        if not out[-1].endswith("\n"):
            out[-1] += "\n"
        out += [x + "\n" for x in e.insert_after]
        if not ending and i == len(lines):
            out[-1] = out[-1].rstrip("\n")
    return out


def file_diff(src: RepoSource, fixes: list[FileFix]) -> str:
    """One unified diff (git style) for all the fixes, file by file."""
    by_path: dict[str, list[FileFix]] = {}
    for fx in fixes:
        by_path.setdefault(fx.path, []).append(fx)
    parts = []
    for path in sorted(by_path):
        group = by_path[path]
        new_content = next((fx.new_content for fx in group if fx.new_content is not None), None)
        if new_content is not None and not src.exists(path):
            body = list(difflib.unified_diff([], new_content.splitlines(keepends=True),
                                             fromfile="/dev/null", tofile=f"b/{path}"))
            parts.append(f"diff --git a/{path} b/{path}\nnew file mode 100644\n" + "".join(body))
            continue
        old = src.read_text(path).splitlines(keepends=True)
        new = _apply(old, [e for fx in group for e in fx.edits])
        body = list(difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}"))
        if body:
            parts.append(f"diff --git a/{path} b/{path}\n" + "".join(
                line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in body))
    return "".join(parts)


def check_patch(src: RepoSource, patch: str) -> str:
    """git apply --check on a clean export of the commit (never on the target repo)."""
    if not patch:
        return "no patch"
    if src.ref is None:
        return "not checked (working tree source)"
    # export the blobs as committed: no eol conversion from the user's git settings
    archive = subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(src.root), "archive", "--format=tar",
                              src.ref], capture_output=True)
    if archive.returncode != 0:
        return f"not checked: git archive failed: {archive.stderr.decode(errors='replace').strip()}"
    with tempfile.TemporaryDirectory(prefix="builder-verify-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        patch_path = Path(tmp) / "fixes.patch"
        patch_path.write_bytes(patch.encode("utf-8"))
        r = subprocess.run(["git", "apply", "--check", "--verbose", "fixes.patch"], cwd=tmp,
                           capture_output=True, text=True)
    if r.returncode == 0:
        return f"applies cleanly to {src.label} (git apply --check on a clean export)"
    return "DOES NOT APPLY: " + (r.stderr.strip() or r.stdout.strip())


# --- report ------------------------------------------------------------------------------

def verify_repo(root: str | Path, ref: str | None = "HEAD", out_dir: str | Path | None = None) -> VerifyReport:
    src = RepoSource(root, ref)
    findings = run_rules(src)
    for f in findings:
        f.fix_diff = file_diff(src, f.fixes) or None
    patch = file_diff(src, [fx for f in findings for fx in f.fixes])
    report = VerifyReport(repo=Path(root).resolve().name, source=src.label,
                          generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          rules=RULES, findings=findings, patch_check=check_patch(src, patch))
    if out_dir is not None:
        out = Path(out_dir)
        if out.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("verify writes its report outside the target repo (it is read-only there)")
        out.mkdir(parents=True, exist_ok=True)
        (out / "fixes.patch").write_text(patch, encoding="utf-8", newline="\n")
        report.patch_file = str(out / "fixes.patch")
        (out / "findings.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        (out / "findings.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    return report


def markdown(r: VerifyReport) -> str:
    out = [f"# Verify findings: {r.repo}", "",
           f"Source: {r.source}. Generated {r.generated_at}. Static rules only; nothing was run or changed.", "",
           f"**{r.errors} error(s), {r.warnings} warning(s).** Patch: `fixes.patch`, {r.patch_check}. "
           "The Builder never applies it.", "",
           "| # | Severity | File | Line | Rule | Fault | Evidence |", "|---|---|---|---|---|---|---|"]
    for i, f in enumerate(r.findings, 1):
        evidence = f.evidence if len(f.evidence) <= 80 else f.evidence[:77] + "..."
        out.append(f"| {i} | {f.severity} | `{f.file}` | {f.line or '-'} | {f.rule} | {f.fault_id} | "
                   f"`{evidence.replace('|', '/')}` |")
    for i, f in enumerate(r.findings, 1):
        out += ["", f"## {i}. {f.rule} ({f.fault_id}, {f.severity}): `{f.file}`" + (f" line {f.line}" if f.line else ""),
                "", f.message, "", f"Evidence: `{f.evidence}`", "", f"Fix: {f.fix}"]
        if f.fix_diff:
            out += ["", "```diff", f.fix_diff.rstrip("\n"), "```"]
    return "\n".join(out) + "\n"
