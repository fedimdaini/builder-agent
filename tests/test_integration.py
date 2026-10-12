"""The integration section of contracts.yaml: one source for every team-dependent value, its shape, the
listing command, agents.yaml, and no team-dependent default left hard-coded in the Builder's code."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from builder_agent.decide import load_contracts, plan_build
from builder_agent.decide.contracts import ContractError
from builder_agent.integration import report, unconfirmed
from builder_agent.render import RenderError, render_adapters
from builder_agent.render.agents import check_agents, runtime_paths
from builder_agent.render.configs import render_configs
from builder_agent.repo_writer import RepoWriter

from test_render import CONTRACTS, MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]


def edited(tmp_path, change):
    data = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    change(data)
    p = tmp_path / "contracts.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_contracts(p)


def setv(group, name, value):
    def change(d):
        d["integration"][group][name]["value"] = value
    return change


def render_all(mini, c, out):  # noqa: F811
    w = RepoWriter(out)
    render_adapters(mini, c, MINI_GOLD, out, writer=w)
    return render_configs(mini, c, plan_build(mini, c), MINI_GOLD, out, writer=w)


# --- one source, the code's views -----------------------------------------------------------------

def test_views_are_filled_from_integration(contracts):  # noqa: F811
    c = contracts
    assert c.serving.port == 8000 and c.serving.endpoints["predict"].path == "/predict"
    assert c.serving.model_uri_env == "MODEL_URI"
    assert c.paths["predictions_log"] == "logs/predictions.jsonl" and c.paths["eval_report"] == "reports/eval.json"
    assert c.paths["mlflow_uri"] == "http://mlflow:5000" and c.paths["raw_data"] == "data/raw"
    extra = c.model_extra
    assert extra["router"]["services"] == {"current": "serving-current", "candidate": "serving-candidate"}
    assert extra["compose"]["deploy_file"] == "compose.deploy.yml"
    assert set(extra["formats"]["eval_report"]["metrics"]) == {"rmse", "mae", "r2"}
    assert c.smoke_test["predict_body_has"] == "prediction"
    assert all(f.group in {"monitor", "trainer", "inspector", "deployer", "board", "reviewer", "team"}
               for f in c.integration_fields())


@pytest.mark.parametrize("change,message", [
    (lambda d: d["paths"].update(eval_report="x.json"), "paths.eval_report moved to the integration section"),
    (lambda d: d["serving"].update(port=9000), "serving.port moved"),
    (lambda d: d.update(router={"config_file": "r.yaml"}), "router moved"),
    (lambda d: d["integration"]["monitor"]["predictions_log"].pop("default"), "must be {value, default, confirmed}"),
    (lambda d: d["integration"]["monitor"]["predictions_log"].update(confirmed="yes"), "confirmed true/false"),
    (lambda d: d["integration"]["team"].pop("mlflow_server_image"), "integration.team.mlflow_server_image is missing"),
    (setv("trainer", "train_command", "python train.py"), "must be `make <target>`"),
])
def test_malformed_contracts_are_refused(tmp_path, change, message):
    with pytest.raises(ContractError, match=message.replace("{", r"\{").replace("}", r"\}")):
        edited(tmp_path, change)


# --- integration values reach the generated files -------------------------------------------------

def test_renamed_fields_and_paths_reach_the_files(mini, tmp_path):  # noqa: F811
    c = edited(tmp_path, lambda d: [setv("monitor", "predictions_log_fields",
                                         {"time": "ts", "model_version": "model_version", "features": "x",
                                          "prediction": "y_hat", "request_id": "rid"})(d),
                                    setv("trainer", "eval_report_fields",
                                         {"model_version": "version", "task": "task", "metrics": "scores",
                                          "per_group": "groups"})(d),
                                    setv("deployer", "current_service", "model-live")(d),
                                    setv("team", "mlflow_server_image", "ghcr.io/mlflow/mlflow:v2.16.0")(d)])
    out = tmp_path / "out"
    r = render_all(mini, c, out)
    assert r.ok, r.lint
    serve = (out / "pipeline/serve.py").read_text(encoding="utf-8")
    assert '"ts": datetime' in serve and '"y_hat": prediction' in serve and '"rid": request_id' in serve
    evaluate = (out / "pipeline/evaluate.py").read_text(encoding="utf-8")
    assert '"scores": metrics(' in evaluate and '"groups": per_group' in evaluate
    compose = yaml.safe_load((out / "compose.base.yml").read_text(encoding="utf-8"))
    assert "model-live" in compose["services"] and compose["services"]["mlflow"]["image"].endswith("v2.16.0")
    assert "mlflow==2.16.0" in (out / "Dockerfile").read_text(encoding="utf-8")


def test_board_image_adds_the_board_service(mini, tmp_path):  # noqa: F811
    c = edited(tmp_path, lambda d: [setv("board", "image", "team/board:1.0")(d), setv("board", "url", "http://msgs:9100")(d)])
    out = tmp_path / "out"
    render_all(mini, c, out)
    compose = yaml.safe_load((out / "compose.base.yml").read_text(encoding="utf-8"))
    assert compose["services"]["msgs"] == {"image": "team/board:1.0", "ports": ["9100:9100"]}


def test_candidate_switch_on_is_refused_until_implemented(mini, tmp_path):  # noqa: F811
    c = edited(tmp_path, setv("deployer", "candidate_and_router", True))
    with pytest.raises(RenderError, match="not implemented yet"):
        render_all(mini, c, tmp_path / "out")


# --- agents.yaml ----------------------------------------------------------------------------------

def test_agents_yaml(mini, contracts, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    r = render_all(mini, contracts, out)
    assert "agents.yaml" in r.files and r.ok, r.lint
    data = yaml.safe_load((out / "agents.yaml").read_text(encoding="utf-8"))
    a = data["agents"]
    assert data["format_version"] == 1 and list(a) == ["builder", "reviewer", "inspector", "trainer", "deployer",
                                                      "monitor"]
    assert a["inspector"]["subscriptions"] == ["dataset:*", "feature:*", "type:question"]
    runtime = {w["path"] for x in a.values() for w in x.get("watch", []) if w["kind"] == "runtime"}
    assert runtime == set(runtime_paths(contracts))           # the three runtime outputs, marked as such
    assert a["trainer"]["commands"] == {"train": "make train", "evaluate": "make evaluate"}
    assert {"path": "data/processed/train.csv", "kind": "file"} in a["inspector"]["watch"]


def test_agents_check_catches_missing_and_fake_runtime_paths(mini, contracts, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    render_all(mini, contracts, out)
    data = yaml.safe_load((out / "agents.yaml").read_text(encoding="utf-8"))
    data["agents"]["reviewer"]["watch"].append({"path": "nowhere", "kind": "folder"})
    data["agents"]["monitor"]["watch"].append({"path": "logs/other.jsonl", "kind": "runtime"})
    problems = check_agents(yaml.safe_dump(data), mini.root, out, contracts)
    assert any("nowhere, which doesn't exist" in p for p in problems)
    assert any("logs/other.jsonl as runtime" in p for p in problems)


def test_subscriptions_are_integration_values(mini, tmp_path):  # noqa: F811
    subs = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))["integration"]["board"]["subscriptions"]["value"]
    c = edited(tmp_path, setv("board", "subscriptions", subs | {"monitor": ["model:*"]}))
    out = tmp_path / "out"
    render_all(mini, c, out)
    assert yaml.safe_load((out / "agents.yaml").read_text(encoding="utf-8"))["agents"]["monitor"]["subscriptions"] == ["model:*"]


# --- the listing command --------------------------------------------------------------------------

def test_listing(tmp_path):
    c = edited(tmp_path, lambda d: [d["integration"]["deployer"]["serving_port"].update(value=9000),
                                    d["integration"]["reviewer"]["builder_commit_author"].update(confirmed=True)])
    text = report(c)
    assert "serving_port = 9000  [changed, default 8000]" in text and "builder_commit_author" not in text
    assert len(unconfirmed(c)) == len(c.integration_fields()) - 1
    r = subprocess.run([sys.executable, "-m", "builder_agent.integration", "--strict"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "not confirmed yet" in r.stdout


# --- nothing team-dependent hard-coded -------------------------------------------------------------

def _defaults(contracts) -> set[str]:  # noqa: F811
    """Default values of the integration fields that are specific enough to grep for."""
    out = set()
    for f in contracts.integration_fields():
        for v in ([f.default] if isinstance(f.default, str) else []):
            if "/" in v or "." in v or "-" in v or "@" in v:
                out.add(v)
    return out - {"make train", "make evaluate", "make serve"}


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def test_no_integration_default_is_hard_coded(contracts):  # noqa: F811
    defaults = _defaults(contracts)
    assert {"serving-current", "reports/eval.json", "ghcr.io/mlflow/mlflow:v2.17.2", "router.yaml"} <= defaults
    pkg = ROOT / "builder_agent"
    found = []
    for path in pkg.rglob("*.py"):
        if "faults" in path.parts or "scan" in path.parts:
            continue        # recorded fault cases describe past runs; the scanner detects files by conventional names
        allowed = {"agents.yaml"} if path.name == "modes.py" else set()     # an artifact's label, not its path
        found += [(path.relative_to(ROOT).as_posix(), s) for s in _string_constants(path) for d in defaults
                  if d in s and s not in allowed]
    for path in (pkg / "render" / "templates").glob("*.j2"):
        text = path.read_text(encoding="utf-8")
        found += [(path.relative_to(ROOT).as_posix(), d) for d in defaults if d in text]
    assert not found, found


# --- bugs the integration rehearsal found (experiments/integration/report.md) -----------------------

def test_runtime_output_folders_are_volumes(mini, tmp_path):  # noqa: F811
    """Bug 1: only models/ and the predictions log folder were mounted, so a latest model URI or an eval
    report in another folder never reached the next stage's container, or the host."""
    c = edited(tmp_path, lambda d: [setv("trainer", "latest_model_uri", "state/latest.txt")(d),
                                    setv("trainer", "eval_report", "metrics_out/eval.json")(d)])
    out = tmp_path / "out"
    render_all(mini, c, out)
    vols = yaml.safe_load((out / "compose.base.yml").read_text(encoding="utf-8"))["services"]["serving-current"]["volumes"]
    assert {"./state:/app/state", "./metrics_out:/app/metrics_out", "./logs:/app/logs"} <= set(vols)


def test_a_shorter_metric_list_filters_the_report(mini, contracts, tmp_path):  # noqa: F811
    """Bug 2: the report kept every supported metric whatever the Trainer listed."""
    c = edited(tmp_path, setv("trainer", "eval_metrics", ["rmse"]))
    render_all(mini, c, tmp_path / "a")
    assert 'keep = ["rmse"]' in (tmp_path / "a/pipeline/evaluate.py").read_text(encoding="utf-8")
    render_all(mini, contracts, tmp_path / "b")                        # the full default list: no filter rendered
    assert "keep = " not in (tmp_path / "b/pipeline/evaluate.py").read_text(encoding="utf-8")


def test_the_smoke_test_checks_the_contract_prediction_field(mini, tmp_path):  # noqa: F811
    """Bug 3: the smoke test read smoke_test from the untyped extras, so it always looked for "prediction"."""
    c = edited(tmp_path, setv("deployer", "prediction_field", "y_pred"))
    render_all(mini, c, tmp_path / "out")
    smoke = (tmp_path / "out/pipeline/smoke_test.py").read_text(encoding="utf-8")
    assert '"y_pred"' in smoke and '"prediction"' not in smoke
