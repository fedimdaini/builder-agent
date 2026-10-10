"""Prompt files: prompts/<version>.md with three sections, rendered with Jinja2.

File format (headings are case-insensitive, any level):

    ## system
    ...
    ## user
    ... {{ placeholders }} ...
    ## retry
    ... {{ reasons_text }} ...

The prompt wording lives only in the file. Placeholders are filled from the
variables documented in slot_variables() and retry_variables(); an unknown
placeholder is an error that names it (StrictUndefined).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from pydantic import BaseModel

from ..decide.models import BuildPlan
from ..decide.rules import choose_task
from ..render.script_slots import ScriptSlotAnswers, script_candidates
from ..render.slots import TASKS, SlotAnswers, SlotAnswersV3, slot_candidates
from ..scan.models import RepoContext

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SECTIONS = ("system", "user", "retry")
HEADING_RE = re.compile(r"^#{1,6}\s*(system|user|retry)\s*$", re.I | re.M)


class PromptError(Exception):
    pass


class PromptFile(BaseModel):
    version: str                   # file stem, e.g. "slots_v1"
    path: str
    sha256: str                    # of the file, so a logged call pins the exact wording
    sections: dict[str, str]       # system / user / retry -> template text

    def render(self, section: str, variables: dict) -> str:
        env = Environment(undefined=StrictUndefined, keep_trailing_newline=False)
        try:
            return env.from_string(self.sections[section]).render(**variables).strip()
        except UndefinedError as e:
            raise PromptError(f"{self.version}.md [{section}]: {e.message}; "
                              f"available: {', '.join(sorted(variables))}") from e
        except TemplateSyntaxError as e:
            raise PromptError(f"{self.version}.md [{section}] line {e.lineno}: {e.message}") from e


def load_prompt(version: str, prompts_dir: str | Path = PROMPTS_DIR) -> PromptFile:
    path = Path(prompts_dir) / f"{version}.md"
    if not path.exists():
        raise PromptError(f"prompt file not found: {path}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    heads = list(HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, h in enumerate(heads):
        name = h.group(1).lower()
        if name in sections:
            raise PromptError(f"{path.name}: section '{name}' appears twice")
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        sections[name] = text[h.end():end].strip("\n")
    missing = [s for s in SECTIONS if s not in sections]
    if missing:
        raise PromptError(f"{path.name}: missing section(s) {', '.join(missing)} "
                          f"(expected headings: {', '.join('## ' + s for s in SECTIONS)})")
    return PromptFile(version=version, path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      sections=sections)


def slots_model(prompt: PromptFile) -> type:
    """The answer schema a slot prompt asks for: script mode if it shows {{ script_sources }}
    (train_script_v1 on), with the task (slots_v3 on) if it shows {{ task_facts }}, else SlotAnswers."""
    user = prompt.sections.get("user", "")
    if "{{ script_sources }}" in user:
        return ScriptSlotAnswers
    return SlotAnswersV3 if "{{ task_facts }}" in user else SlotAnswers


MAX_SCRIPT_LINES = 150     # lines of each training script shown to the LLM (script mode)


def script_facts(ctx: RepoContext) -> str:
    """Per training script: its CLI options, sys.argv positions and where it writes a model."""
    out = []
    for t in ctx.train_scripts:
        out.append(f"SCRIPT {t.path} (cli: {t.cli or 'none'}, main guard: {'yes' if t.has_main_guard else 'no'})")
        for a in t.cli_args:
            out.append(f"  - option {' / '.join(a.flags)}: {'required' if a.required else 'optional'}"
                       + (f", default {a.default}" if a.default else "") + (f", type {a.type}" if a.type else "")
                       + (f", action {a.action}" if a.action else "") + (f" ({a.help})" if a.help else ""))
        if t.argv:
            out.append(f"  - reads sys.argv[{'], sys.argv['.join(str(i) for i in t.argv)}]")
        for o in t.outputs:
            out.append(f"  - line {o.line}: {o.kind} -> {o.target}")
    return "\n".join(out) or "none"


def script_sources(ctx: RepoContext) -> str:
    """The training scripts' code with line numbers, at most MAX_SCRIPT_LINES lines each."""
    blocks = []
    for t in ctx.train_scripts:
        lines = (Path(ctx.root) / t.path).read_text(encoding="utf-8", errors="replace").splitlines()
        shown = "\n".join(f"{i:4d}  {line}" for i, line in enumerate(lines[:MAX_SCRIPT_LINES], 1))
        more = f"\n      ... ({len(lines) - MAX_SCRIPT_LINES} more lines)" if len(lines) > MAX_SCRIPT_LINES else ""
        blocks.append(f"--- {t.path} ---\n{shown}{more}")
    return "\n\n".join(blocks) or "none"


def task_facts(ctx: RepoContext) -> str:
    """The task rule's decision and its evidence, for slots_v3."""
    choice, _ = choose_task(ctx)
    head = (f"TASK: decided by the scan signals: {choice.task} ({choice.reason})" if choice.status == "decided"
            else f"TASK: not decided ({choice.reason})")
    return "\n".join([head] + [f"  - {line}" for line in choice.evidence])


def slot_variables(ctx: RepoContext, plan: BuildPlan) -> dict:
    """Variables for the system and user sections."""
    needs = [n.model_dump() for n in plan.needs_llm]
    needs_text = "\n".join(
        f"- {n['item']}: {n['reason']}" + "".join(f"\n    - {c}" for c in n["context"]) for n in needs)
    candidates = slot_candidates(ctx)
    schema = SlotAnswers.model_json_schema()
    return {
        "repo": ctx.name,
        "scan_summary": ctx.summary(),                 # compact RepoContext text
        "needs_llm": needs,                            # [{item, reason, context: [...]}, ...]
        "needs_llm_text": needs_text,                  # the same, as a bullet list
        "candidates": candidates,                      # slot_candidates(ctx) dict
        "candidates_json": json.dumps(candidates, indent=2),
        "schema": schema,                              # SlotAnswers JSON schema
        "schema_json": json.dumps(schema, indent=2),
        # slots_v3: the task (regression or classification); v1/v2 don't use these
        "task_facts": task_facts(ctx),
        "task_candidates": TASKS,
        # train_script_v1 (script-training mode); earlier prompts don't use these
        "script_candidates": script_candidates(ctx),
        "script_facts": script_facts(ctx),
        "script_sources": script_sources(ctx),
    }


def retry_variables(reasons: list[str], previous_answer: str, attempt: int) -> dict:
    """Variables for the retry section (on top of slot_variables)."""
    return {
        "reasons": reasons,                            # validator rejection reasons
        "reasons_text": "\n".join(f"- {r}" for r in reasons),
        "previous_answer": previous_answer,            # the model's last raw answer
        "attempt": attempt,                            # number of the attempt about to be made (2 or 3)
    }
