"""agents.yaml (format version 1, docs/INTEGRATION.md): for each agent, what to watch in this repo and the
board topics it subscribes to. Every value comes from the plan, the scan, the slot answers or contracts.yaml
(subscriptions, paths, commands and the board URL are integration fields). Written once by the Builder;
the board owns it afterwards.

watch entries are {path, kind}: kind "file" or "folder" (exists in the repo or is written by this Builder run)
or "runtime" (appears after the first pipeline run: the predictions log, the eval report, the latest model URI).
check_agents is the static check: the format, and every non-runtime path exists.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

import yaml

from ..decide.contracts import Contracts
from ..decide.models import BuildPlan
from ..scan.models import RepoContext

FORMAT_VERSION = 1
AGENTS = ("builder", "reviewer", "inspector", "trainer", "deployer", "monitor")


def _entry(path: str, kind: str) -> dict:
    return {"path": path, "kind": kind}


def runtime_paths(c: Contracts) -> list[str]:
    return [c.i("monitor", "predictions_log"), c.i("trainer", "eval_report"), c.i("trainer", "latest_model_uri")]


def agents_data(ctx: RepoContext, c: Contracts, plan: BuildPlan, slots, builder_files: list[str]) -> dict:
    subs = c.i("board", "subscriptions")
    if missing := [a for a in AGENTS if a not in subs]:
        raise ValueError(f"integration.board.subscriptions has no entry for {', '.join(missing)}")
    adapters = c.paths["adapters_dir"].strip("/")
    script = getattr(slots, "train_script", None)
    train_file = script.train_file if script else slots.train.train_file
    eval_file = slots.evaluate.eval_file
    sources = sorted({e.path.split("/")[0] for e in ctx.entry_points if not e.path.startswith(adapters + "/")})
    endpoints = c.serving.endpoints
    current = c.i("deployer", "current_service")
    port = c.serving.port
    data_dirs = [d for d in (c.i("inspector", "raw_data"), c.i("inspector", "processed_data"))
                 if (Path(ctx.root) / d).is_dir()]
    response = (c.serving.model_extra or {}).get("response", {})
    return {
        "format_version": FORMAT_VERSION,
        "repo": ctx.name,
        "board_url": c.i("board", "url"),
        "agents": {
            "builder": {"subscriptions": subs["builder"],
                        "watch": [_entry(f, "file") for f in builder_files] + [_entry(adapters, "folder")]},
            "reviewer": {"subscriptions": subs["reviewer"],
                         "watch": [_entry(s, "folder" if (Path(ctx.root) / s).is_dir() else "file") for s in sources]
                         + [_entry(t, "file") for t in ctx.test_files] + [_entry(adapters, "folder")],
                         "ci_workflow": {"path": c.i("reviewer", "ci_workflow"), "status": "not written yet"}},
            "inspector": {"subscriptions": subs["inspector"],
                          "watch": [_entry(d, "folder") for d in data_dirs]
                          + [_entry(f, "file") for f in dict.fromkeys([train_file, eval_file])],
                          "schema": {"path": c.i("inspector", "schema_path"), "status": "not generated yet"}},
            "trainer": {"subscriptions": subs["trainer"],
                        "commands": {"train": c.i("trainer", "train_command"),
                                     "evaluate": c.i("trainer", "evaluate_command")},
                        "mlflow": {"uri": c.paths["mlflow_uri"], "experiment": _experiment(ctx, c)},
                        "watch": [_entry(c.i("trainer", "eval_report"), "runtime"),
                                  _entry(c.i("trainer", "latest_model_uri"), "runtime")]},
            "deployer": {"subscriptions": subs["deployer"],
                         "commands": {"serve": c.i("deployer", "serve_command")},
                         "service": {"name": current, "port": port,
                                     "endpoints": {k: f"{e.method.upper()} {e.path}" for k, e in endpoints.items()},
                                     "model_uri_env": c.serving.model_uri_env},
                         "files_you_create": [c.i("deployer", "deploy_file"), c.i("deployer", "router_config")],
                         "watch": [_entry(f, "file") for f in builder_files
                                   if f in ("Dockerfile", (c.model_extra or {})["compose"]["base_file"])]},
            "monitor": {"subscriptions": subs["monitor"],
                        "health_url": f"http://{current}:{port}{endpoints['health'].path}",
                        "fields": {"prediction": response.get("prediction_field"),
                                   "model_version": response.get("model_version_field"),
                                   "log_line": c.i("monitor", "predictions_log_fields")},
                        "watch": [_entry(c.i("monitor", "predictions_log"), "runtime")]},
        },
    }


def _experiment(ctx: RepoContext, c: Contracts) -> str:
    from . import experiment_name
    return experiment_name(ctx, c)


def render_agents(ctx: RepoContext, c: Contracts, plan: BuildPlan, slots, builder_files: list[str]) -> str:
    head = (f"# {c.i('board', 'agents_file')} for {ctx.name}: what each agent watches in this repo and the board\n"
            f"# topics it subscribes to. Format version {FORMAT_VERSION}: docs/INTEGRATION.md. Written once by the\n"
            "# Builder (owner afterwards: board). kind: runtime = appears after the first pipeline run.\n")
    return head + yaml.safe_dump(agents_data(ctx, c, plan, slots, builder_files), sort_keys=False, width=110)


def check_agents(text: str, repo_root: str | Path, out_root: str | Path, c: Contracts) -> list[str]:
    """The format, and every watched path: exists in the repo or in this run's output, or is a runtime output."""
    problems = []
    me = c.i("board", "agents_file")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return [f"{me}: not valid YAML: {e}"]
    if data.get("format_version") != FORMAT_VERSION:
        problems.append(f"{me}: format_version must be {FORMAT_VERSION}")
    agents = data.get("agents") or {}
    if sorted(agents) != sorted(AGENTS):
        problems.append(f"{me}: agents must be {', '.join(AGENTS)}")
    runtime = set(runtime_paths(c))
    for name, a in agents.items():
        if not isinstance(a.get("subscriptions"), list) or not a["subscriptions"]:
            problems.append(f"{me}: {name} has no subscriptions")
        for w in a.get("watch", []):
            path, kind = w.get("path", ""), w.get("kind")
            if kind == "runtime":
                if path not in runtime:
                    problems.append(f"{me}: {name} watches {path} as runtime, but it isn't a pipeline output")
            elif kind in ("file", "folder"):
                found = [r / path for r in (Path(repo_root), Path(out_root)) if (r / path).exists()]
                if not found:
                    problems.append(f"{me}: {name} watches {path}, which doesn't exist")
                elif kind == "folder" and not found[0].is_dir():
                    problems.append(f"{me}: {name} watches {path} as a folder, but it is a file")
            else:
                problems.append(f"{me}: {name} watch entry {w} has no kind file/folder/runtime")
            if posixpath.isabs(path) or ".." in path.split("/"):
                problems.append(f"{me}: {name} watches {path}, outside the repo")
    return problems
