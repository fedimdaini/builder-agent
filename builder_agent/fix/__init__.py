"""Build-failure diagnosis loop: when a sandbox stage fails, the LLM picks one fix from a fixed menu.

    from builder_agent.fix import run_fix_loop
    result = run_fix_loop(repo, slots, "diagnose_v1", OllamaClient("qwen2.5-coder:7b"), mlflow_client="mlflow")

A fix is a setting (Overrides) that changes only what the Builder generates. Every attempt
re-renders the Builder's files into a fresh sandbox copy (through RepoWriter), so the repo's own
files are never touched. At most 3 fixes; every attempt is logged.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ..decide import load_contracts, plan_build
from ..llm import ChatClient
from ..llm.ollama import inline_refs
from ..llm.prompts import PromptFile, load_prompt
from ..render import validate_slots
from ..render.configs import MLFLOW_IMAGE, Overrides, config_context
from ..sandbox import DEFAULT_CONTRACTS, DEFAULT_LOG, STAGES, SandboxResult, run_sandbox, subprocess_runner
from ..scan import scan_repo
from ..scan.deps import declared_names, normalize
from ..scan.known import FRAMEWORKS, IMPORT_TO_DIST
from .models import Diagnosis, DiagnosisWithAnalysis, GiveUp, answer_model, apply_fix, describe, validate_fix

MAX_FIXES = 3
MAX_CALLS = 3                         # LLM calls per diagnosis (rejected answers are retried with reasons)
DEFAULT_FIX_LOG = Path(__file__).resolve().parents[2] / "logs" / "fix_attempts.jsonl"

MENU = {
    "pin_package": "{name, version}: install exactly this version of a Python package",
    "add_system_package": "{name}: apt-get install a Debian package in the image",
    "set_python_version": "{version}: change the Python version of the image, e.g. 3.11",
    "set_env_var": "{name, value}: set an environment variable in the image",
    "add_dependency": "{name}: pip install a Python package that is missing",
    "give_up": "{reason}: none of the above can fix this error",
}


class Call(BaseModel):
    messages: list[dict]
    response: str | None
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    valid: bool
    reasons: list[str] = Field(default_factory=list)
    error: str | None = None


class FixAttempt(BaseModel):
    n: int
    stage: str
    error_tail: list[str]
    calls: list[Call]
    action: dict | None = None        # the accepted fix
    action_text: str | None = None
    analysis: str | None = None       # the accepted answer's step-by-step analysis (prompts that ask for it)
    reason: str | None = None
    outcome: str                      # "passed", "failed at <stage>", "not applied: ...", "reverted: regression ..."
    sandbox_attempt_id: str | None = None
    retrieved: list[dict] | None = None   # RAG: the memory cases put into the prompt (id, variant, score, ranks)


class FixLoopResult(BaseModel):
    repo: str
    prompt_version: str
    prompt_sha256: str
    model: str
    model_digest: str
    initial: str                      # outcome of the first sandbox run
    initial_attempt_id: str
    attempts: list[FixAttempt]
    final_ok: bool
    stopped_because: str
    applied: Overrides                # the injected fault (if any) plus the fixes on top
    injection: Overrides | None = None
    retriever: str | None = None      # RAG retriever name (builder_agent.memory), if any

    def text(self) -> str:
        out = [f"FIX LOOP {self.repo} ({self.prompt_version} x {self.model}): "
               + ("PASSED" if self.final_ok else "NOT FIXED") + f", stopped because: {self.stopped_because}",
               f"  initial run: {self.initial}"]
        for a in self.attempts:
            out.append(f"  fix {a.n}: stage {a.stage} -> {a.action_text or 'no valid action'} -> {a.outcome}")
            if a.retrieved is not None:
                out.append("         retrieved: " + ", ".join(f"{r['id']} {r['variant']}" for r in a.retrieved))
            if a.analysis:
                out.append(f"         analysis: {a.analysis}")
            if a.reason:
                out.append(f"         reason: {a.reason}")
            for i, c in enumerate(a.calls, 1):
                if not c.valid:
                    out.append(f"         call {i} rejected: {c.error or '; '.join(c.reasons)}")
        return "\n".join(out)


# --- prompt variables ----------------------------------------------------------------------

def _failed(r: SandboxResult):
    return next((s for s in r.stages if s.status == "failed"), None)


def _outcome(r: SandboxResult) -> str:
    return "passed" if r.ok else f"failed at {r.failed_stage}"


def prompt_variables(ctx, r: SandboxResult, tctx: dict, attempt: int, history: list[FixAttempt]) -> dict:
    stage = _failed(r)
    prev = [f"- {h.action_text}: then {h.outcome}" for h in history if h.action_text]
    return {
        "stage": stage.name,                                       # the failed sandbox stage
        "error_tail": "\n".join(stage.output_tail),               # its last output lines (<= 50)
        "scan_summary": ctx.summary(),                             # facts about the repo
        "attempt": attempt, "max_attempts": MAX_FIXES,
        "previous_fixes": [h.model_dump() for h in history],
        "previous_fixes_text": "\n".join(prev) or "none",
        "menu_text": "\n".join(f"- {k} {v}" for k, v in MENU.items()),
        # what the Builder generated (not used by diagnose_v1: "from the error alone")
        "pipeline_text": "\n".join([
            f"base image: python:{tctx['python_version']}-slim",
            *[f"install: {c}" for c in tctx["install_commands"]],
            f"MLflow server image: {MLFLOW_IMAGE}",
            f"serve: {' '.join(tctx['serve_cmd'])}",
        ]),
    }


def _installed(ctx, tctx: dict) -> set[str]:
    names = set(declared_names(ctx.dependency_files))
    for cmd in tctx["install_commands"]:      # in order: a later uninstall removes, a later install adds back
        words = cmd.split()
        if cmd.startswith("pip install "):
            names |= {normalize(t.split("==")[0]) for t in words[2:] if not t.startswith("-")}
        elif cmd.startswith("pip uninstall "):
            names -= {normalize(t) for t in words[2:] if not t.startswith("-")}
    return names


def _python_packages(ctx) -> set[str]:
    """Names known to be Python packages: what the repo imports, plus the scanner's known ML packages."""
    return ({normalize(t.distribution) for t in ctx.third_party_imports} | {normalize(n) for n in FRAMEWORKS}
            | {normalize(d) for d in IMPORT_TO_DIST.values()})


def regressed(before: SandboxResult, after: SandboxResult) -> bool:
    """True if `after` failed at an earlier stage than `before` (a fix made things worse)."""
    if after.ok or before.ok or before.failed_stage is None:
        return False
    return STAGES.index(after.failed_stage) < STAGES.index(before.failed_stage)


# --- one diagnosis ------------------------------------------------------------------------------

def diagnose(prompt: PromptFile, client: ChatClient, variables: dict, applied: Overrides,
             installed: set[str], python_packages: set[str] = frozenset(), reverted: list[dict] = ()
             ) -> tuple[Diagnosis | DiagnosisWithAnalysis | None, list[Call]]:
    model = answer_model("\n".join(prompt.sections.values()))
    schema = inline_refs(model.model_json_schema())
    messages = [{"role": "system", "content": prompt.render("system", variables)},
                {"role": "user", "content": prompt.render("user", variables)}]
    calls: list[Call] = []
    for n in range(1, MAX_CALLS + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat(messages, schema=schema)
        except Exception as e:  # noqa: BLE001 - logged, then the loop stops
            calls.append(Call(messages=messages, response=None, latency_s=round(time.perf_counter() - t0, 2),
                              valid=False, error=f"{type(e).__name__}: {e}"))
            return None, calls
        try:
            answer = json.loads(resp.content)
            v = validate_fix(answer, applied, installed, model, python_packages, reverted) \
                if isinstance(answer, dict) else None
            reasons = v.reasons if v else ["the answer must be one JSON object"]
        except json.JSONDecodeError as e:
            v, reasons = None, [f"the answer is not valid JSON: {e}"]
        calls.append(Call(messages=messages, response=resp.content, latency_s=resp.latency_s,
                          prompt_tokens=resp.prompt_tokens, output_tokens=resp.output_tokens,
                          valid=not reasons, reasons=reasons))
        if v and v.ok:
            return v.diagnosis, calls
        if n < MAX_CALLS:
            retry = prompt.render("retry", variables | {"reasons": reasons,
                                                        "reasons_text": "\n".join(f"- {x}" for x in reasons),
                                                        "previous_answer": resp.content})
            messages = messages + [{"role": "assistant", "content": resp.content}, {"role": "user", "content": retry}]
    return None, calls


# --- the loop -------------------------------------------------------------------------------------

def run_fix_loop(repo: str | Path, slots: dict, prompt_version: str, client: ChatClient,
                 expected: dict | None = None, mlflow_client: str | None = None, max_fixes: int = MAX_FIXES,
                 contracts_path: str | Path = DEFAULT_CONTRACTS, prompts_dir: str | Path | None = None,
                 runner=subprocess_runner, sandbox_log: str | Path | None = DEFAULT_LOG,
                 fix_log: str | Path | None = DEFAULT_FIX_LOG, injection: Overrides | None = None,
                 retriever=None, retriever_name: str | None = None) -> FixLoopResult:
    """injection: a generated fault (builder_agent/faults) the loop starts from; fixes are applied on top.
    retriever: (stage, error_tail) -> memory hits (builder_agent.memory), rendered into {{ retrieved_text }}."""
    prompt = load_prompt(prompt_version, prompts_dir) if prompts_dir else load_prompt(prompt_version)
    ctx = scan_repo(repo)
    c = load_contracts(contracts_path)
    plan = plan_build(ctx, c)
    answers = validate_slots(ctx, slots).slots
    applied = injection.model_copy(deep=True) if injection else Overrides()
    reverted: list[dict] = []         # fixes undone by the regression guard; offering one again is rejected

    def sandbox(o: Overrides) -> SandboxResult:
        return run_sandbox(repo, slots, expected=expected, contracts_path=contracts_path, log_path=sandbox_log,
                           runner=runner, mlflow_client=mlflow_client, overrides=o)

    r = sandbox(applied)
    initial, initial_id = _outcome(r), r.attempt_id
    attempts: list[FixAttempt] = []
    stopped = "passed" if r.ok else "max fixes reached"
    digest = client.digest
    for n in range(1, max_fixes + 1):
        if r.ok:
            stopped = "passed"
            break
        tctx = config_context(ctx, c, plan, answers, mlflow_client, applied)
        variables = prompt_variables(ctx, r, tctx, n, attempts)
        stage = _failed(r)
        retrieved = None
        if retriever is not None:
            from ..memory import render_retrieved      # chromadb only when RAG is used
            hits = retriever(stage.name, stage.output_tail)
            variables["retrieved_text"] = render_retrieved(hits)
            retrieved = [{"id": h.doc.id, "variant": h.doc.variant, "score": h.score, "ranks": h.ranks} for h in hits]
        diag, calls = diagnose(prompt, client, variables, applied, _installed(ctx, tctx), _python_packages(ctx),
                               reverted)
        attempt = FixAttempt(n=n, stage=stage.name, error_tail=stage.output_tail, calls=calls, outcome="",
                             retrieved=retrieved)
        if diag is None:
            attempt.outcome = "not applied: " + ("LLM call failed" if calls and calls[-1].error else
                                                 f"no valid fix after {len(calls)} call(s)")
            stopped = "invalid diagnosis" if not (calls and calls[-1].error) else "LLM call failed"
        else:
            attempt.action, attempt.action_text, attempt.reason = (diag.fix.model_dump(), describe(diag.fix),
                                                                   diag.reason)
            attempt.analysis = getattr(diag, "analysis", None)
            if isinstance(diag.fix, GiveUp):
                attempt.outcome = "not applied: the LLM gave up"
                stopped = "give_up"
            else:
                trial = sandbox(apply_fix(applied, diag.fix))
                attempt.sandbox_attempt_id = trial.attempt_id
                if regressed(r, trial):
                    # the fix broke an earlier stage than the one it was meant to fix: undo it and keep
                    # the previous state (and its error) for the next diagnosis; the attempt counts as failed
                    attempt.outcome = (f"reverted: regression (failed at {trial.failed_stage}, "
                                       f"before {r.failed_stage})")
                    reverted.append(diag.fix.model_dump())
                else:
                    applied, r = apply_fix(applied, diag.fix), trial
                    attempt.outcome = _outcome(r)
        attempts.append(attempt)
        _log(fix_log, ctx.name, prompt, client, digest, attempt)
        if diag is None or isinstance(diag.fix, GiveUp):
            break
    else:
        stopped = "passed" if r.ok else "max fixes reached"
    return FixLoopResult(repo=ctx.name, prompt_version=prompt.version, prompt_sha256=prompt.sha256,
                         model=client.model, model_digest=digest, initial=initial, initial_attempt_id=initial_id,
                         attempts=attempts, final_ok=r.ok, stopped_because=stopped, applied=applied,
                         injection=injection, retriever=retriever_name)


def _log(path, repo: str, prompt: PromptFile, client, digest: str, a: FixAttempt) -> None:
    if not path:
        return
    entry = {"time": datetime.now(timezone.utc).isoformat(), "task": "build_fix", "repo": repo,
             "prompt_version": prompt.version, "prompt_sha256": prompt.sha256, "model": client.model,
             "model_digest": digest, "options": client.options,
             "keep_alive": getattr(client, "keep_alive", None)} | a.model_dump()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def score_attempts(attempts: list[dict], expected_fix: dict) -> dict:
    """Score one run from its fix attempts (FixAttempt dumps or fix_attempts.jsonl entries).
    Main: did the first fix pass the sandbox, and how many fixes until it passed (None: not fixed).
    Secondary: was the first accepted action the expected one (exact: same action and arguments;
    loose: same action on the same package, any version)."""
    passed = next((a["n"] for a in attempts if a["outcome"] == "passed"), None)
    first = next((a["action"] for a in attempts if a.get("action")), None)
    same = first is not None and first.get("action") == expected_fix.get("action") and \
        normalize(str(first.get("name", ""))) == normalize(str(expected_fix.get("name", "")))
    return {"first_fix_passed": passed == 1, "fixes_to_pass": passed,
            "exact": same and all(first.get(k) == v for k, v in expected_fix.items()), "loose": same,
            "first_action": first}


def score_fix(result: FixLoopResult, expected_fix: dict) -> dict:
    return score_attempts([a.model_dump() for a in result.attempts], expected_fix)


def export_run(result: FixLoopResult, out: str | Path, summary: str, fix_log: str | Path = DEFAULT_FIX_LOG,
               sandbox_log: str | Path = DEFAULT_LOG, score: dict | None = None) -> Path:
    """Copy one run's log lines (its fix attempts and sandbox attempts) and its summary to out/.
    Call it right after the run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sandbox_ids = {result.initial_attempt_id} | {a.sandbox_attempt_id for a in result.attempts if a.sandbox_attempt_id}
    # runs go one at a time, so this run's fix attempts are the last lines of the fix log
    fixes = Path(fix_log).read_text(encoding="utf-8").splitlines()[-len(result.attempts):] if result.attempts else []
    sandboxes = [line for line in Path(sandbox_log).read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("attempt_id") in sandbox_ids]
    (out / "fix_attempts.jsonl").write_text("\n".join(fixes) + "\n", encoding="utf-8", newline="\n")
    (out / "sandbox_attempts.jsonl").write_text("\n".join(sandboxes) + "\n", encoding="utf-8", newline="\n")
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8", newline="\n")
    if score is not None:
        write_score(out, score)
    return out


def write_score(out: str | Path, score: dict) -> None:
    (Path(out) / "score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8", newline="\n")
