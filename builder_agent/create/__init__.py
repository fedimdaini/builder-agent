"""Create mode: the whole Builder on one repo, and the files written only if the sandbox passes.

    python -m builder_agent.create <repo> [--dry-run] [--model M] [--slots FILE] [--expected FILE] [--no-write-back]

scan -> decide -> slot filling (the approved prompt for the repo's training mode) -> sandbox; if it fails
because of a slot answer (fix/origin.py), the error goes back to the slot LLM (prompts/slot_runtime_retry_v1.md),
at most MAX_RUNTIME_RETRIES times -> fix loop
(diagnose_v3 + advanced retriever over the base memory, Python rule on) -> if every stage passed: render the
same files with the fixes applied, write them through RepoWriter on a new builder/<date> branch, commit
them there, and switch back to the original branch (in a finally block, whatever happens).

Never writes into the repo if: the repo isn't git, has uncommitted changes to tracked files, isn't on a
branch, already has pipeline files (use verify mode), would get a file where any file already exists
(tracked or untracked) or where .gitignore excludes it, or any stage still fails after the fix loop.
--dry-run does everything except the write. Each run's report goes to logs/create/ (outside the target).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .. import LOG_DIR
from ..decide import load_contracts, plan_build
from ..decide.modes import select_modes
from ..fix import DEFAULT_FIX_LOG, FixLoopResult, run_fix_loop
from ..fix.origin import GROUP_SLOTS, changed_slots, classify, outside_group, slot_files
from ..llm import DEFAULT_CALL_LOG, ChatClient, SlotFillResult, fill_slots, load_prompt, runtime_retry, slot_messages
from ..render import Expected, render_adapters, validate_slots
from ..render.configs import render_configs
from ..repo_writer import (RepoWriter, WriteRefused, current_branch, ignored_paths, is_git_repo, tracked_changes,
                           worktree_state)
from ..sandbox import DEFAULT_CONTRACTS, DEFAULT_LOG, run_sandbox, subprocess_runner
from ..scan import scan_repo

# the approved prompts (docs/RESULTS.md): slots per training mode, and the default diagnosis setup
SLOT_PROMPTS = {"function": "slots_v3", "script": "train_script_v1"}
DIAGNOSE_PROMPT = "diagnose_v3"
RUNTIME_RETRY_PROMPT = "slot_runtime_retry_v1"
MAX_RUNTIME_RETRIES = 2                  # slot answers corrected from a sandbox error, before the fix loop
ERROR_LINES = 30                         # last lines of the failed stage shown in a runtime retry
RETRIEVER = "advanced"
MEMORY_SET = "base"
REPORT_DIR = LOG_DIR / "create"
BUILDER_ROOT = Path(__file__).resolve().parents[2]


class CreateResult(BaseModel):
    repo: str
    status: str                          # "written", "dry-run", "refused", "failed"
    reason: str = ""                     # why it was refused or failed
    original_branch: str | None = None
    branch: str | None = None            # the builder branch, if a commit was made
    commit: str | None = None
    files: list[str] = Field(default_factory=list)
    commit_message: str | None = None
    slots_source: str | None = None      # "llm" or "file:<path>"
    slots: dict | None = None
    slot_attempts: list[dict] = Field(default_factory=list)
    runtime_retries: list[dict] = Field(default_factory=list)   # slot answers corrected from a sandbox error
    fix_loop: dict | None = None         # FixLoopResult dump
    worktree_unchanged: bool | None = None
    report_path: str | None = None

    def text(self) -> str:
        out = [f"CREATE {self.repo}: {self.status.upper()}" + (f": {self.reason}" if self.reason else "")]
        for a in self.slot_attempts:
            out.append(f"  slots try {a['attempt']}: " + ("valid" if a["valid"] else "rejected: " + "; ".join(a["reasons"])))
        out += ["  " + runtime_retry_text(r) for r in self.runtime_retries]
        if self.fix_loop:
            fl = FixLoopResult.model_validate(self.fix_loop)
            out += ["  " + line for line in fl.text().splitlines()]
            if not fl.final_ok and fl.final_failed_stage:
                out.append(f"  still failing at {fl.final_failed_stage}; last lines:")
                out += ["    | " + line for line in fl.final_error_tail[-15:]]
        if self.files:
            out.append(f"  files ({len(self.files)}): " + ", ".join(self.files))
        if self.status == "dry-run" and self.commit_message:
            out += ["  would commit on a new builder/<date> branch:"] + ["    " + x for x in self.commit_message.splitlines()]
        if self.commit:
            out.append(f"  committed {self.commit[:12]} on {self.branch}; back on {self.original_branch}"
                       + ("; working tree unchanged" if self.worktree_unchanged else "; WORKING TREE CHANGED"))
        if self.report_path:
            out.append(f"  report: {self.report_path}")
        return "\n".join(out)


def _builder_version() -> str:
    r = subprocess.run(["git", "-C", str(BUILDER_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _prediction(sandbox_log, attempt_id: str | None) -> float | None:
    if not (sandbox_log and attempt_id and Path(sandbox_log).exists()):
        return None
    for line in Path(sandbox_log).read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        if a.get("attempt_id") == attempt_id:
            for s in a["stages"]:
                for t in s.get("output_tail", []):
                    if t.startswith("prediction: "):
                        return float(t.split(": ", 1)[1])
    return None


def render_files(ctx, c, plan, slots: dict, overrides, expected: dict | None) -> dict[str, str]:
    """The files the sandbox checked (adapters, configs, manifest), rendered into a temp folder: path -> text."""
    tmp = Path(tempfile.mkdtemp(prefix="builder-create-"))
    try:
        writer = RepoWriter(tmp)                                     # no .git: new files only
        a = render_adapters(ctx, c, slots, tmp, writer=writer,
                            expected=Expected.model_validate(expected) if expected else None)
        render_configs(ctx, c, plan, slots, tmp, None, writer=writer, overrides=overrides)
        if a.lint:
            raise WriteRefused("lint: " + "; ".join(a.lint))
        return {rel: (tmp / rel).read_text(encoding="utf-8") for rel in sorted(writer.created)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def runtime_retry_text(r: dict) -> str:
    o = r["origin"]
    head = (f"runtime slot retry {r['retry']}: {r['stage']} failed, blamed {o['group']} ({o['frame']}: "
            f"{o['exception']})")
    if not r["valid"]:
        return head + " -> no valid answer"
    flag = f"; CHANGED OUTSIDE THE BLAMED GROUP: {', '.join(r['outside_group'])}" if r["outside_group"] else ""
    return head + f" -> changed {', '.join(r['changed']) or 'nothing'}{flag} -> sandbox {r['sandbox_after']}"


def commit_message(ctx, plan, files: list[str], slots: dict, slots_source: str, slot_fill: SlotFillResult | None,
                   fl: FixLoopResult, prediction: float | None, runtime_retries: list[dict] = ()) -> str:
    p, t, m = plan.python, plan.task, plan.train_mode
    task = (f"{t.task} (rule: {t.reason})" if t and t.status == "decided"
            else f"{slots.get('task', '?')} (chosen by the LLM; the scan signals didn't settle it)")
    fixes = [f"- fix {a.n} ({a.source}): {a.action_text} -> {a.outcome}" for a in fl.attempts if a.action_text]
    source = (f"LLM: {slot_fill.prompt_version} x {slot_fill.model}, valid on try {len(slot_fill.attempts)}"
              if slot_fill else f"from {slots_source.removeprefix('file:')} (not the LLM)")
    return "\n".join([
        f"Builder: pipeline files for {ctx.name} (sandbox: all stages ok)",
        "",
        f"Files ({len(files)}):", *[f"- {f}" for f in files],
        "",
        "Plan assumptions:",
        f"- python: {fl.applied.python_version or p.version}" + (" (ASSUMED: Builder default)" if p.assumed else "")
        + f" ({p.reason})",
        f"- task: {task}",
        f"- training mode: {m.mode if m else 'function'} ({m.reason if m else 'no rule'})",
        "",
        f"Slot answers ({source}):", json.dumps(slots, indent=2, sort_keys=True),
        "",
        "Runtime slot retries:", *([f"- {runtime_retry_text(r)}" for r in runtime_retries] or ["- none"]),
        "",
        "Fixes applied:", *(fixes or ["- none: the first sandbox run passed"]),
        "",
        f"Sandbox: attempt {fl.final_attempt_id}" + (f", prediction {prediction}" if prediction is not None else ""),
        f"Diagnosis: {fl.prompt_version} x {fl.model}, retriever {fl.retriever or 'none'}, "
        f"python rule {'on' if fl.python_rule else 'off'}",
        f"Builder: builder-agent {_builder_version()}",
    ])


def create(repo: str | Path, client: ChatClient, slots_file: str | Path | None = None, expected: dict | None = None,
           dry_run: bool = False, contracts_path: str | Path = DEFAULT_CONTRACTS, runner=subprocess_runner,
           sandbox_log=DEFAULT_LOG, fix_log=DEFAULT_FIX_LOG, call_log=DEFAULT_CALL_LOG, retriever=None,
           retriever_name: str | None = None, write_back=None, report_dir: str | Path | None = REPORT_DIR,
           prompts_dir: str | Path | None = None) -> CreateResult:
    """write_back: called with the FixLoopResult after the loop (None: no write-back)."""
    root = Path(repo).resolve()
    res = CreateResult(repo=root.name, status="refused")

    def done(status: str, reason: str = "") -> CreateResult:
        res.status, res.reason = status, reason
        if report_dir:
            d = Path(report_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{res.repo}.json"
            res.report_path = str(path)
            path.write_text(res.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        return res

    # 1. preconditions: nothing is written if any of these fails
    if not is_git_repo(root):
        return done("refused", "not a git repo: create mode commits on a builder/<date> branch")
    res.original_branch = current_branch(root)
    if res.original_branch is None:
        return done("refused", "HEAD is detached: check out a branch first")
    if dirty := tracked_changes(root):
        return done("refused", f"uncommitted changes to tracked files ({len(dirty)}): commit or stash them first")
    ctx = scan_repo(root)
    c = load_contracts(contracts_path)
    existing = [m for m in select_modes(ctx, c) if m.mode == "verify"]
    if existing:
        return done("refused", "the repo already has pipeline files ("
                    + "; ".join(f"{m.artifact}: {', '.join(m.existing)}" for m in existing)
                    + "): use verify mode, python -m builder_agent.verify <repo>")
    plan = plan_build(ctx, c)

    # 2. slots: the LLM with the approved prompt for the training mode, or a file
    slot_fill = None
    mode = plan.train_mode.mode if plan.train_mode else "function"
    if not slots_file and plan.train_mode and plan.train_mode.status != "decided":
        return done("failed", f"training mode not decided: {plan.train_mode.reason}")
    prompt = load_prompt(SLOT_PROMPTS[mode], prompts_dir) if prompts_dir else load_prompt(SLOT_PROMPTS[mode])
    if slots_file:
        from ..render.slots import load_answer
        slots, _ = load_answer(slots_file)
        res.slots_source = f"file:{slots_file}"
    else:
        slot_fill = fill_slots(ctx, plan, prompt, client, log_path=call_log, strict=True)
        res.slot_attempts = [{"attempt": a.attempt, "valid": a.valid, "reasons": a.reasons} for a in slot_fill.attempts]
        res.slots_source = "llm"
        if not slot_fill.valid:
            return done("failed", f"slot filling ({prompt.version}) gave no valid answer"
                        + (f": {slot_fill.error}" if slot_fill.error else f" after {len(slot_fill.attempts)} tries"))
        slots = slot_fill.answer
    res.slots = slots
    v = validate_slots(ctx, slots, strict=True)        # with the create-mode checks (render.slots.create_checks)
    if not v.ok:
        return done("failed", "invalid slot answers: " + "; ".join(v.reasons))

    # where the files would go (fixes change their content, never their paths): checked before any sandbox run
    try:
        paths = list(render_files(ctx, c, plan, slots, None, expected))
    except Exception as e:  # noqa: BLE001
        return done("failed", f"rendering failed: {type(e).__name__}: {e}")
    if taken := [f for f in paths if (root / f).exists()]:
        return done("refused", "a file already exists (tracked or untracked) where the Builder would write: "
                    + ", ".join(taken))
    if ignored := ignored_paths(root, paths):
        return done("refused", "the repo's .gitignore excludes files the Builder would commit: " + ", ".join(ignored))
    author = c.i("reviewer", "builder_commit_author")

    # 3. the sandbox, and the slot answers corrected from its error when the slots caused it
    def sandbox(s: dict):
        return run_sandbox(root, s, expected=expected, contracts_path=contracts_path, log_path=sandbox_log,
                           runner=runner)
    first = sandbox(slots)
    retry_prompt = (load_prompt(RUNTIME_RETRY_PROMPT, prompts_dir, required=("user",)) if prompts_dir
                    else load_prompt(RUNTIME_RETRY_PROMPT, required=("user",)))
    conversation = (slot_fill.conversation if slot_fill else
                    slot_messages(ctx, plan, prompt) + [{"role": "assistant", "content": json.dumps(slots)}])
    adapters_dir = c.paths["adapters_dir"].strip("/")
    for k in range(1, MAX_RUNTIME_RETRIES + 1):
        failed = next((s for s in first.stages if s.status == "failed"), None)
        if first.ok or failed is None:
            break
        origin = classify(failed.name, failed.output_tail, slot_files(validate_slots(ctx, slots).slots, adapters_dir))
        if origin.kind != "slot":
            break                                 # the environment: the fix loop's job
        variables = {"stage": failed.name, "origin": origin.frame + (f" (`{origin.code}`)" if origin.code else ""),
                     "slot_group": ", ".join(GROUP_SLOTS[origin.group]),
                     "error_tail": "\n".join(failed.output_tail[-ERROR_LINES:]),
                     "previous_answer": json.dumps(slots, indent=2)}
        r = runtime_retry(ctx, plan, prompt, retry_prompt, client, conversation, variables, log_path=call_log,
                          strict=True)
        entry = {"retry": k, "stage": failed.name, "origin": origin.model_dump(), "calls": len(r.attempts),
                 "valid": r.valid, "rejected": [a.reasons for a in r.attempts if not a.valid],
                 "changed": [], "outside_group": [], "sandbox_before": first.attempt_id}
        if not r.valid:
            res.runtime_retries.append(entry)
            break
        changed = changed_slots(slots, r.answer)
        entry |= {"changed": changed, "outside_group": outside_group(changed, origin.group)}
        slots, conversation = r.answer, r.conversation
        first = sandbox(slots)
        entry["sandbox_after"] = "passed" if first.ok else f"failed at {first.failed_stage}"
        res.runtime_retries.append(entry)
    res.slots = slots

    # 4. the fix loop, starting from the last sandbox run
    fl = run_fix_loop(root, slots, DIAGNOSE_PROMPT, client, expected=expected, contracts_path=contracts_path,
                      runner=runner, sandbox_log=sandbox_log, fix_log=fix_log, retriever=retriever,
                      retriever_name=retriever_name, python_rule=True, initial=first,
                      **({"prompts_dir": prompts_dir} if prompts_dir else {}))
    if write_back:
        write_back(fl)
    res.fix_loop = fl.model_dump()
    if not fl.final_ok:
        return done("failed", f"the sandbox still fails at {fl.final_failed_stage} after {len(fl.attempts)} fix(es)")

    # 5. the files, rendered with the fixes applied (RepoWriter checks again that none exists)
    try:
        files = render_files(ctx, c, plan, slots, fl.applied, expected)
    except Exception as e:  # noqa: BLE001
        return done("failed", f"rendering the final files failed: {type(e).__name__}: {e}")
    res.files = list(files)
    if sorted(files) != sorted(paths):
        return done("failed", "the final files differ from the planned paths: " + ", ".join(sorted(set(files) ^ set(paths))))
    res.commit_message = commit_message(ctx, plan, res.files, slots, res.slots_source, slot_fill, fl,
                                        _prediction(sandbox_log, fl.final_attempt_id), res.runtime_retries)
    if dry_run:
        return done("dry-run")

    # 6. write and commit on builder/<date>; back to the original branch whatever happens
    before = worktree_state(root)
    writer, error = None, None
    try:
        writer = RepoWriter(root)
        writer.write_new(files)
        res.commit, res.branch = writer.commit(res.commit_message, author), writer.branch
    except WriteRefused as e:
        error = f"write refused: {e}"
    finally:
        if writer is not None:
            writer.restore(res.original_branch)
        res.worktree_unchanged = current_branch(root) == res.original_branch and worktree_state(root) == before
    return done("failed", error) if error else done("written")
