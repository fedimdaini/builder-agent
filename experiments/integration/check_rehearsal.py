"""Integration rehearsal check: in the files create committed on builder/<date> with contracts.rehearsal.yaml,
every changed value appears where it should and no old default remains.

    python experiments/integration/check_rehearsal.py <name>=<repo> ... --out experiments/integration/check.json

Reads the repos only (git show of the Builder's branch, and the sandbox log for the evaluate output).
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
REH = yaml.safe_load((HERE / "contracts.rehearsal.yaml").read_text(encoding="utf-8"))["integration"]
REAL = yaml.safe_load((ROOT / "contracts.yaml").read_text(encoding="utf-8"))["integration"]


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8").stdout


def builder_branch(repo) -> str:
    return [b for b in git(repo, "branch", "--format=%(refname:short)").split() if b.startswith("builder/")][-1]


def files(repo, branch) -> dict[str, str]:
    names = git(repo, "show", "--name-only", "--format=", branch).split()
    return {n: git(repo, "show", f"{branch}:{n}") for n in names}


def evaluate_output(repo_name: str) -> str:
    """The evaluate stage's printed metrics in this repo's create run (sandbox log, last attempt for the repo)."""
    lines = (ROOT / "logs" / "sandbox_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        a = json.loads(line)
        if a.get("repo") == repo_name and a.get("ok"):
            s = next(s for s in a["stages"] if s["name"] == "evaluate")
            return "\n".join(s["output_tail"])
    return ""


def checks(f: dict[str, str], author: str, eval_out: str, task: str) -> list[dict]:
    serve, train, ev = f["pipeline/serve.py"], f["pipeline/train.py"], f["pipeline/evaluate.py"]
    model = f["pipeline/_model.py"] if "pipeline/_model.py" in f else serve
    compose, docker, smoke = f["compose.base.yml"], f["Dockerfile"], f["pipeline/smoke_test.py"]
    agents_name = REH["board"]["agents_file"]["value"]
    # the agents file at its rehearsal path; else (a run with the real contracts, as a negative control) any agents file
    other = next((n for n in f if n.endswith("agents.yaml")), None)
    agents = yaml.safe_load(f.get(agents_name) or (f[other] if other else "{}")) or {}
    a = agents.get("agents", {})
    every = "\n".join(f.values())
    out = []

    def add(field, value, where, ok, old_absent=True, note=""):
        out.append({"field": field, "value": value, "where": where, "ok": bool(ok and old_absent), "note": note})

    v = lambda g, n: REH[g][n]["value"]          # noqa: E731
    d = lambda g, n: REAL[g][n]["default"]       # noqa: E731
    absent = lambda *olds: all(o not in every for o in olds)  # noqa: E731

    add("monitor.predictions_log", v("monitor", "predictions_log"), "serve.py PREDICTIONS_LOG; compose volume ./monitoring; agents.yaml monitor watch (runtime)",
        f'PREDICTIONS_LOG = "{v("monitor", "predictions_log")}"' in serve and "./monitoring:/app/monitoring" in compose
        and {"path": v("monitor", "predictions_log"), "kind": "runtime"} in a["monitor"]["watch"], absent(d("monitor", "predictions_log")))
    lf = v("monitor", "predictions_log_fields")
    add("monitor.predictions_log_fields", lf, "serve.py log line keys (and the response's request id key)",
        all(f'"{lf[k]}": ' in serve for k in lf) and f'"{lf["request_id"]}": request_id}}), 200' in serve,
        all(f'"{k}": ' not in serve for k in ("time", "features", "request_id")))
    add("trainer.eval_report", v("trainer", "eval_report"), "evaluate.py EVAL_REPORT; compose volume ./metrics_out; agents.yaml trainer watch (runtime)",
        f'EVAL_REPORT = "{v("trainer", "eval_report")}"' in ev and "./metrics_out:/app/metrics_out" in compose, absent(d("trainer", "eval_report")))
    rf = v("trainer", "eval_report_fields")
    add("trainer.eval_report_fields", rf, "evaluate.py report keys" + (" (task key: classification only)" if task == "regression" else ""),
        all(f'"{rf[k]}": ' in ev for k in ("model_version", "metrics", "per_group")), '"metrics": metrics(' not in ev)
    add("trainer.eval_metrics", v("trainer", "eval_metrics"), "evaluate.py keep filter; the evaluate stage printed only these",
        f'keep = {json.dumps(v("trainer", "eval_metrics"))}' in ev and all(f'"{m}"' in eval_out for m in v("trainer", "eval_metrics")),
        '"r2"' not in eval_out, note=f"evaluate printed: {eval_out.strip().splitlines()[-1] if eval_out.strip() else '?'}")
    add("trainer.eval_classification_metrics", v("trainer", "eval_classification_metrics"), "evaluate.py, classification repos only",
        task == "regression", note="not rendered for a regression repo (the filter is per task); covered by tests")
    add("trainer.latest_model_uri", v("trainer", "latest_model_uri"), "train/evaluate/serve MODEL_URI_FILE; compose volume ./state; evaluate and serve passed reading it",
        f'"{v("trainer", "latest_model_uri")}"' in train and "./state:/app/state" in compose, absent(d("trainer", "latest_model_uri")))
    add("trainer.mlflow_experiment", v("trainer", "mlflow_experiment"), "train.py EXPERIMENT; agents.yaml trainer.mlflow",
        f'EXPERIMENT = "{v("trainer", "mlflow_experiment")}"' in train and a["trainer"]["mlflow"]["experiment"] == v("trainer", "mlflow_experiment"),
        absent(d("trainer", "mlflow_experiment")))
    raw_listed = {"path": v("inspector", "raw_data"), "kind": "folder"} in a["inspector"]["watch"]
    add("inspector.raw_data", v("inspector", "raw_data"), "agents.yaml inspector watch (only if the folder exists in the repo)",
        True, d("inspector", "raw_data") not in [w["path"] for w in a["inspector"]["watch"]],
        note="listed" if raw_listed else "folder doesn't exist in this repo, so not listed")
    add("inspector.schema_path", v("inspector", "schema_path"), "agents.yaml inspector.schema.path",
        a["inspector"]["schema"]["path"] == v("inspector", "schema_path"), "path: schema.py\n" not in every)
    port = v("deployer", "serving_port")
    add("deployer.serving_port", port, "Dockerfile EXPOSE, gunicorn bind, compose ports and healthcheck, agents.yaml",
        f"EXPOSE {port}" in docker and f"0.0.0.0:{port}" in docker + f["Makefile"] and f'"{port}:{port}"' in compose
        and a["deployer"]["service"]["port"] == port, "8000" not in every)
    ep = v("deployer", "endpoints")
    add("deployer.endpoints", ep, "serve.py routes, smoke_test.py, compose healthcheck, agents.yaml",
        f'"{ep["predict"]["path"]}"' in serve and f'"{ep["health"]["path"]}"' in serve and ep["health"]["path"] in compose
        and f'PREDICT = ("POST", "{ep["predict"]["path"]}")' in smoke, '"/predict"' not in every and '"/health"' not in every)
    add("deployer.prediction_field", v("deployer", "prediction_field"), "serve.py response key; smoke_test.py checks it; predict stage passed",
        f'PREDICTION_FIELD = "{v("deployer", "prediction_field")}"' in serve and f'"{v("deployer", "prediction_field")}"' in smoke,
        'BODY_HAS = "prediction"' not in smoke)
    add("deployer.model_version_field", v("deployer", "model_version_field"), "serve.py response key",
        f'VERSION_FIELD = "{v("deployer", "model_version_field")}"' in serve, 'VERSION_FIELD = "model_version"' not in serve)
    env = v("deployer", "model_uri_env")
    add("deployer.model_uri_env", env, "compose environment; the adapters read it",
        f"{env}: " in compose and f'"{env}"' in model, re.search(r'(?<!_)"MODEL_URI"', every) is None)
    add("deployer.current_service", v("deployer", "current_service"), "compose service; agents.yaml deployer.service and monitor.health_url",
        f"  {v('deployer', 'current_service')}:\n" in compose and a["deployer"]["service"]["name"] == v("deployer", "current_service"),
        absent(d("deployer", "current_service")))
    add("deployer.candidate_service", v("deployer", "candidate_service"), "nowhere yet: used once the candidate container exists (ROADMAP D)",
        True, absent(d("deployer", "candidate_service")), note="not used while candidate_and_router is off")
    add("deployer.deploy_file", v("deployer", "deploy_file"), "compose header comment; agents.yaml deployer.files_you_create",
        v("deployer", "deploy_file") in compose and v("deployer", "deploy_file") in a["deployer"]["files_you_create"],
        absent(d("deployer", "deploy_file")))
    add("deployer.router_config", v("deployer", "router_config"), "agents.yaml deployer.files_you_create",
        v("deployer", "router_config") in a["deployer"]["files_you_create"], "- router.yaml" not in every)
    add("board.url", v("board", "url"), "agents.yaml board_url; compose board service name and port",
        agents.get("board_url") == v("board", "url") and "  board-svc:\n" in compose and '"5678:5678"' in compose, absent(d("board", "url")))
    add("board.image", v("board", "image"), "compose board service image", f"image: {v('board', 'image')}" in compose)
    add("board.agents_file", agents_name, "the agents file's path in the commit", agents_name in f, "agents.yaml" not in f)
    add("board.subscriptions", "6 agents changed", "agents.yaml subscriptions",
        all(a[k]["subscriptions"] == s for k, s in v("board", "subscriptions").items()))
    add("reviewer.builder_commit_author", v("reviewer", "builder_commit_author"), "the commit's author", author == v("reviewer", "builder_commit_author"))
    add("reviewer.ci_workflow", v("reviewer", "ci_workflow"), "agents.yaml reviewer.ci_workflow (create doesn't write the workflow yet)",
        a["reviewer"]["ci_workflow"]["path"] == v("reviewer", "ci_workflow"), absent(d("reviewer", "ci_workflow")))
    add("team.mlflow_uri", v("team", "mlflow_uri"), "compose service tracking on 5050 and MLFLOW_TRACKING_URI; train/evaluate/serve default; agents.yaml",
        "  tracking:\n" in compose and "--port 5050" in compose and f'"{v("team", "mlflow_uri")}"' in train, absent(d("team", "mlflow_uri")))
    add("team.mlflow_server_image", v("team", "mlflow_server_image"), "compose mlflow image; Dockerfile client pin mlflow==2.17.1",
        v("team", "mlflow_server_image") in compose and "mlflow==2.17.1" in docker, absent("v2.17.2", "mlflow==2.17.2"))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repos", nargs="+", help="name=path")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = {}
    for spec in args.repos:
        name, path = spec.split("=", 1)
        branch = builder_branch(path)
        f = files(path, branch)
        author = git(path, "log", "-1", "--format=%an <%ae>", branch).strip()
        task = "regression"            # both rehearsal repos are regression (decided by the task rule)
        result[name] = {"branch": branch, "files": sorted(f),
                        "checks": checks(f, author, evaluate_output(Path(path).name), task)}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, r in result.items():
        bad = [c["field"] for c in r["checks"] if not c["ok"]]
        print(f"{name}: {len(r['checks']) - len(bad)}/{len(r['checks'])} ok" + (f"; NOT OK: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
