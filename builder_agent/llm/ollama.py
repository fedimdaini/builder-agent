"""Minimal Ollama chat client (stdlib only) with JSON-schema structured output."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from pydantic import BaseModel

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}


class ChatResponse(BaseModel):
    content: str
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


def inline_refs(schema: dict) -> dict:
    """Resolve $defs/$ref so the schema is self-contained for the grammar converter."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


class OllamaClient:
    def __init__(self, model: str, host: str = DEFAULT_HOST, options: dict | None = None, timeout: float = 900):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        self._digest: str | None = None

    @property
    def digest(self) -> str:
        if self._digest is None:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                tags = json.load(r)["models"]
            match = next((m for m in tags if m["name"] == self.model), None)
            if match is None:
                raise RuntimeError(f"model {self.model!r} is not pulled in Ollama "
                                   f"(have: {', '.join(m['name'] for m in tags)})")
            self._digest = match["digest"]
        return self._digest

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResponse:
        body = {"model": self.model, "stream": False, "options": self.options, "messages": messages}
        if schema is not None:
            body["format"] = schema
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        return ChatResponse(content=resp["message"]["content"], latency_s=round(time.perf_counter() - t0, 2),
                            prompt_tokens=resp.get("prompt_eval_count"), output_tokens=resp.get("eval_count"))
