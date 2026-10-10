"""Slot-filling loop: the LLM answers SlotAnswers as strict JSON, validated against scanner facts.

    from builder_agent.llm import fill_slots, load_prompt, OllamaClient
    result = fill_slots(ctx, plan, load_prompt("slots_v1"), OllamaClient("qwen2.5-coder:7b"))

One conversation: system + user, then up to two retries that add the validator's
reasons (the prompt's retry section). Every call is logged as one JSON line.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .. import LOG_DIR
from ..decide.models import BuildPlan
from ..render.slots import validate_slots
from ..scan.models import RepoContext
from .ollama import ChatResponse, OllamaClient, inline_refs
from .prompts import PromptError, PromptFile, load_prompt, retry_variables, slot_variables, slots_model

__all__ = ["fill_slots", "load_prompt", "OllamaClient", "PromptError", "SlotFillResult", "DEFAULT_CALL_LOG"]

MAX_ATTEMPTS = 3
DEFAULT_CALL_LOG = LOG_DIR / "llm_calls.jsonl"


class ChatClient(Protocol):
    model: str
    options: dict

    @property
    def digest(self) -> str: ...

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResponse: ...


class Attempt(BaseModel):
    attempt: int
    latency_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    valid: bool
    reasons: list[str] = Field(default_factory=list)
    answer: dict | None = None     # parsed JSON, if it parsed
    raw: str


class SlotFillResult(BaseModel):
    prompt_version: str
    prompt_sha256: str
    model: str
    model_digest: str
    valid: bool
    answer: dict | None            # the last valid answer, else the last parsed one
    attempts: list[Attempt]
    error: str | None = None       # a call failed for good (after the transport retry): the loop stopped

    @property
    def valid_on_first_try(self) -> bool:
        return bool(self.attempts) and self.attempts[0].valid

    @property
    def total_tokens(self) -> int:
        return sum((a.prompt_tokens or 0) + (a.output_tokens or 0) for a in self.attempts)

    @property
    def total_s(self) -> float:
        return round(sum(a.latency_s for a in self.attempts), 2)


def _parse(raw: str) -> tuple[dict | None, list[str]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, [f"the answer is not valid JSON: {e}"]
    if not isinstance(value, dict):
        return None, ["the answer must be one JSON object"]
    return value, []


def fill_slots(ctx: RepoContext, plan: BuildPlan, prompt: PromptFile, client: ChatClient,
               max_attempts: int = MAX_ATTEMPTS, log_path: str | Path | None = DEFAULT_CALL_LOG) -> SlotFillResult:
    variables = slot_variables(ctx, plan)
    messages = [{"role": "system", "content": prompt.render("system", variables)},
                {"role": "user", "content": prompt.render("user", variables)}]
    schema = inline_refs(slots_model(prompt).model_json_schema())   # v1/v2: SlotAnswers, unchanged
    digest = client.digest
    attempts: list[Attempt] = []
    answer: dict | None = None

    error = None
    for n in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat(messages, schema=schema)
        except Exception as e:  # noqa: BLE001 - log every failed call, then stop the loop
            error = f"{type(e).__name__}: {e}"
            _log(log_path, prompt, client, digest, ctx, n, messages, None, [], error=error,
                 latency_s=round(time.perf_counter() - t0, 2))
            break
        parsed, reasons = _parse(resp.content)
        if parsed is not None:
            answer = parsed
            reasons = validate_slots(ctx, parsed).reasons
        attempts.append(Attempt(attempt=n, latency_s=resp.latency_s, prompt_tokens=resp.prompt_tokens,
                                output_tokens=resp.output_tokens, valid=not reasons, reasons=reasons,
                                answer=parsed, raw=resp.content))
        _log(log_path, prompt, client, digest, ctx, n, messages, resp, reasons)
        if not reasons:
            break
        if n < max_attempts:
            retry = prompt.render("retry", variables | retry_variables(reasons, resp.content, n + 1))
            messages = messages + [{"role": "assistant", "content": resp.content},
                                   {"role": "user", "content": retry}]

    return SlotFillResult(prompt_version=prompt.version, prompt_sha256=prompt.sha256, model=client.model,
                          model_digest=digest, valid=error is None and bool(attempts) and attempts[-1].valid,
                          answer=answer, attempts=attempts, error=error)


def _log(log_path, prompt: PromptFile, client: ChatClient, digest: str, ctx: RepoContext, attempt: int,
         messages: list[dict], resp: ChatResponse | None, reasons: list[str], error: str | None = None,
         latency_s: float | None = None) -> None:
    if not log_path:
        return
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "task": "slot_filling",
        "repo": ctx.name,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "model": client.model,
        "model_digest": digest,
        "options": client.options,
        "attempt": attempt,
        "messages": messages,
        "response": resp.content if resp else None,
        "latency_s": resp.latency_s if resp else latency_s,
        "prompt_tokens": resp.prompt_tokens if resp else None,
        "output_tokens": resp.output_tokens if resp else None,
        "transport_errors": resp.transport_errors if resp else [],
        "error": error,                  # the call failed for good (no response)
        "valid": error is None and not reasons,
        "reasons": reasons,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
