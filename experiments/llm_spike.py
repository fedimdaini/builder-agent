"""Throwaway spike, not part of the agent.

Asks a local Ollama model to write two adapters for the taxi repo from the
BuildPlan's needs_llm items: pipeline/train.py and pipeline/serve.py.
The generated code is saved and printed, never run.

    .venv/Scripts/python experiments/llm_spike.py [../taxi-trip-regression]
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from builder_agent.decide import load_contracts, plan_build
from builder_agent.scan import scan_repo

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5-coder:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
ROOT = HERE.parent

SYSTEM = (
    "You write thin adapter scripts for an ML repository. Rules: the adapter lives in the "
    "repo's pipeline/ folder and is run from the repo root; it only imports and calls functions "
    "the repo already has; it never modifies the repo's files. Answer with one Python file in a "
    "single ```python code block and nothing else."
)


def ollama(prompt: str) -> dict:
    body = json.dumps({"model": MODEL, "stream": False, "options": OPTIONS,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    resp["latency_s"] = round(time.perf_counter() - t0, 1)
    return resp


def model_digest() -> str:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        tags = json.load(r)["models"]
    return next((m["digest"] for m in tags if m["name"] == MODEL), "unknown")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def item_block(plan, item_id: str) -> str:
    item = next(n for n in plan.needs_llm if n.item == item_id)
    return f"Item: {item.item}\nReason: {item.reason}\nContext:\n" + "\n".join(f"- {c}" for c in item.context)


def train_prompt(plan, contracts, repo: Path) -> str:
    p = contracts.paths
    return f"""Write pipeline/train.py for this repo.

Why the Builder needs it (from its build plan):
{item_block(plan, "target:train")}

src/models/train_model.py:
```python
{(repo / "src/models/train_model.py").read_text(encoding="utf-8")}
```

Requirements (from the team contract):
- Call train_xgboost() from src/models/train_model.py; do not reimplement training.
- Training data lives under {p["processed_data"]}/.
- Log the model and its metrics (rmse, mae, r2) to MLflow: tracking URI from env MLFLOW_TRACKING_URI,
  default {p["mlflow_uri"]}; experiment "{p["mlflow_experiment"]}".
- Sample mode: when env {contracts.sample_mode.variable}=1, use only a {contracts.sample_mode.fraction} fraction of the rows.
- Runnable as `python pipeline/train.py`.
"""


def serve_prompt(plan, contracts, repo: Path) -> str:
    s = contracts.serving
    extra = s.model_extra or {}
    ep = "\n".join(f"- {n}: {e.method} {e.path}" for n, e in s.endpoints.items())
    return f"""Write pipeline/serve.py for this repo.

Why the Builder needs it (from its build plan):
{item_block(plan, "serving")}

{plan.serving.app_path}:
```python
{(repo / plan.serving.app_path).read_text(encoding="utf-8")}
```

Requirements (from the team contract):
- Reuse the repo's prediction function from {plan.serving.app_path}; do not reimplement it.
- Load the model from MLflow using the model URI in env {s.model_uri_env}, not from a local file.
- Endpoints:
{ep}
- Request: {extra.get("request", {}).get("format")}, one record per request, features sent BY NAME, never by position.
- Response JSON fields: "{extra["response"]["prediction_field"]}" (float) and "{extra["response"]["model_version_field"]}".
- Errors must return HTTP 4xx or 5xx, never 200.
- It will be started with `gunicorn --bind 0.0.0.0:{s.port} pipeline.serve:app`.
"""


def run(name: str, prompt: str, digest: str) -> None:
    print(f"\n{'=' * 30} {name} {'=' * 30}\nasking {MODEL} ...", flush=True)
    resp = ollama(prompt)
    text = resp["message"]["content"]
    code = extract_code(text)
    try:
        ast.parse(code)
        syntax = "ok"
    except SyntaxError as e:
        syntax = f"SyntaxError line {e.lineno}: {e.msg}"

    (OUT / f"pipeline_{name}.py").write_text(code, encoding="utf-8")
    (OUT / f"pipeline_{name}.log.json").write_text(json.dumps({
        "model": MODEL, "model_digest": digest, "options": OPTIONS, "system": SYSTEM,
        "prompt": prompt, "response": text, "latency_s": resp["latency_s"],
        "prompt_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count"),
        "syntax": syntax,
    }, indent=2), encoding="utf-8")

    print(f"latency {resp['latency_s']} s | prompt {resp.get('prompt_eval_count')} tok | "
          f"output {resp.get('eval_count')} tok | syntax {syntax}")
    print(f"saved experiments/output/pipeline_{name}.py\n")
    print(code)


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "taxi-trip-regression").resolve()
    contracts = load_contracts(ROOT / "contracts.yaml")
    plan = plan_build(scan_repo(repo), contracts)
    OUT.mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    digest = model_digest()
    run("train", train_prompt(plan, contracts, repo), digest)
    run("serve", serve_prompt(plan, contracts, repo), digest)


if __name__ == "__main__":
    main()
