"""Compact, deterministic text view of a RepoContext for the LLM planner."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RepoContext


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _cap(items: list[str], n: int, sep: str = ", ") -> str:
    if not items:
        return "none"
    more = f"{sep}+{len(items) - n} more" if len(items) > n else ""
    return sep.join(items[:n]) + more


def _slot_candidates(ctx: RepoContext, add, n: int) -> None:
    # columns: group files with the same header; show processed data in full, count the rest
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for d in ctx.data_columns:
        groups[tuple(d.columns)].append(d.path)
    shown = {cols: paths for cols, paths in groups.items()
             if any("/processed/" in f"/{p}" for p in paths)} or dict(list(groups.items())[:2])
    for cols, paths in shown.items():
        add(f"COLUMNS of {_cap(paths, 4)} ({len(cols)}): {_cap(list(cols), 40)}")
    if len(groups) > len(shown):
        add(f"  (+{len(groups) - len(shown)} other header groups in other data files)")
    if not groups:
        add("COLUMNS: no CSV/TSV headers in data folders")

    if ctx.target_candidates:
        def fmt(c):
            kinds = sorted({e.split(":")[0] for e in c.evidence})
            where = "" if c.in_data else ", not in data headers"
            return f"{c.column} (score {c.score}{where}; {', '.join(kinds)})"
        add("TARGET CANDIDATES: " + _cap([fmt(c) for c in ctx.target_candidates], 3, "; "))
    else:
        add("TARGET CANDIDATES: none")

    for t in ctx.target_transforms:
        inv = (f"inverse {t.inverse} in {_cap(t.inverse_files, n)}" if t.inverse_files
               else f"inverse {t.inverse} NEVER applied")
        add(f"TARGET TRANSFORM: {t.forward} on {', '.join(t.applied_to)} "
            f"(in {_cap(t.forward_files, n)}) | {inv}")
    if not ctx.target_transforms:
        add("TARGET TRANSFORM: none found")


def render_summary(ctx: RepoContext, n: int = 8) -> str:
    out: list[str] = []
    add = out.append

    add(f"REPO {ctx.name} | git: {'yes' if ctx.is_git_repo else 'no'} | "
        f"{ctx.n_python_files} .py files, {len(ctx.notebooks)} notebooks")

    dep_parts = []
    for d in ctx.dependency_files:
        detail = f"{len(d.packages)} pkgs" if d.packages else (d.note or "empty")
        if d.packages and d.note:
            detail += f"; {d.note}"
        dep_parts.append(f"{d.path} [{detail}]")
    add(f"DEPENDENCY FILES: {_cap(dep_parts, n, '; ')}")

    by_value: dict[str, list[str]] = defaultdict(list)
    for h in ctx.python_version_hints:
        by_value[h.value].append(h.kind)
    add("PYTHON HINTS: " + _cap([f"{v} ({', '.join(sorted(set(k)))})" for v, k in by_value.items()], n))

    fw_by_cat: dict[str, list[str]] = defaultdict(list)
    for fw in ctx.frameworks:
        tag = "" if fw.via == ["import", "dependency"] else f" ({fw.via[0]} only)"
        fw_by_cat[fw.category].append(fw.name + tag)
    add("FRAMEWORKS: " + (" | ".join(f"{c}: {', '.join(v)}" for c, v in fw_by_cat.items()) or "none"))

    add(f"THIRD-PARTY IMPORTS ({len(ctx.third_party_imports)}): "
        + _cap([t.distribution for t in ctx.third_party_imports], 2 * n))
    undeclared = ctx.undeclared_imports
    add("UNDECLARED IMPORTS: " + _cap(
        [f"{t.distribution} ({'notebooks only' if t.notebooks_only else 'pipeline code'}, "
         f"{len(t.files)} file{'s' if len(t.files) > 1 else ''})" for t in undeclared], n))

    add("ENTRY POINTS:" if ctx.entry_points else "ENTRY POINTS: none")
    for e in ctx.entry_points[:n]:
        bits = [f"[{e.role}]", "__main__" if e.has_main_guard else "no __main__"]
        if e.cli:
            bits.append(f"cli={e.cli}")
        if e.signatures:
            bits.append("defs: " + "; ".join(
                sig.render() + (f" [{sig.model_api}: {', '.join(sig.api_evidence)}]" if sig.model_api else "")
                for sig in e.signatures[:4]))
        elif e.functions:
            bits.append("defs: " + ", ".join(e.functions[:4]))
        add(f"  - {e.path} " + " ".join(bits))
    if len(ctx.entry_points) > n:
        add(f"  - +{len(ctx.entry_points) - n} more")

    add("WEB APPS:" if ctx.web_apps else "WEB APPS: none")
    for w in ctx.web_apps[:n]:
        port = str(w.port) if w.port is not None else "not set in code"
        if w.port_env_var:
            port += f" (env {w.port_env_var})"
        routes = _cap([f"{'/'.join(r.methods)} {r.path}" for r in w.routes], n)
        add(f"  - {w.framework} {w.target} | port {port} | routes: {routes}")

    ports: dict[int, list[str]] = defaultdict(list)
    for p in ctx.port_hints:
        ports[p.port].append(f"{p.kind}@{p.source}")
    add("PORTS SEEN: " + _cap([f"{p} ({', '.join(src)})" for p, src in ports.items()], n))

    for d in ctx.data_dirs + ctx.model_dirs:
        exts = ", ".join(f"{e.lstrip('.')}:{c}" for e, c in list(d.extensions.items())[:5])
        sub = f" subdirs: {', '.join(d.subdirs)};" if d.subdirs else ""
        add(f"{d.kind.upper()} DIR: {d.path}/ —{sub} {d.n_files} files, {human_size(d.total_bytes)} ({exts or 'empty'})")
    if not ctx.data_dirs:
        add("DATA DIR: none")
    add("MODEL FILES: " + _cap([f"{m.path} ({human_size(m.size_bytes)})" for m in ctx.model_files], n))
    if ctx.loose_data_files:
        add("DATA FILES OUTSIDE DATA DIRS: " + _cap([f.path for f in ctx.loose_data_files], n))
    found = sorted({r.resolved for r in ctx.path_references if r.resolved})
    missing = [r.literal for r in ctx.path_references if not r.exists]
    add("PATHS IN CODE (found): " + _cap(found, n))
    if missing:
        add("PATHS IN CODE (MISSING): " + _cap(missing, n))

    _slot_candidates(ctx, add, n)

    add("NOTEBOOKS: " + _cap([f"{nb.path} ({nb.n_code_cells} code cells)" for nb in ctx.notebooks], n))

    lfs = ctx.lfs
    if lfs.patterns or lfs.pointer_files:
        ptr = (f"{len(lfs.pointer_files)} NOT PULLED (pointer files): "
               + _cap([p.path for p in lfs.pointer_files], n)) if lfs.pointer_files else "0 pointer files (content pulled)"
        add(f"GIT LFS: {len(lfs.tracked_files)} tracked files, {ptr}")
    else:
        add("GIT LFS: not used")

    add("EXISTING PIPELINE FILES: " + _cap(ctx.existing_pipeline_files, n))
    add(f"TESTS: {len(ctx.test_files)} files" + (", pytest config" if ctx.has_pytest_config else ""))
    if ctx.parse_errors:
        add("PARSE ERRORS: " + _cap([f"{e.path}: {e.error}" for e in ctx.parse_errors], n, "; "))
    return "\n".join(out)
