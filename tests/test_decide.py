from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from builder_agent.decide import load_contracts, plan_build
from builder_agent.decide.contracts import Contracts
from builder_agent.scan import scan_repo
from builder_agent.scan.models import (
    DependencyFile, EntryPoint, PathReference, RepoContext, Requirement, Route, ThirdPartyImport,
    VersionHint, WebApp,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts.yaml"
TAXI = ROOT.parent / "taxi-trip-regression"


@pytest.fixture
def contracts() -> Contracts:
    return load_contracts(CONTRACTS)


def edit_contracts(tmp_path, change) -> Contracts:
    data = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    change(data)
    p = tmp_path / "contracts.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_contracts(p)


def hint(value, kind, source="x"):
    return VersionHint(value=value, kind=kind, source=source)


def imp(dist, *files, declared=True):
    return ThirdPartyImport(module=dist, distribution=dist, declared=declared, files=list(files))


def ctx(**kw) -> RepoContext:
    base = dict(root="/r", name="r", is_git_repo=True,
                dependency_files=[DependencyFile(path="requirements.txt", kind="requirements",
                                                 packages=[Requirement(name="pandas")])],
                python_version_hints=[hint("3.11", "python-version-file")])
    return RepoContext(**(base | kw))


FULL_APP = WebApp(path="serve.py", framework="fastapi", app_var="app", target="serve:app", port=9000,
                  routes=[Route(methods=["POST"], path="/predict"), Route(methods=["GET"], path="/health")])


# --- python ----------------------------------------------------------------

@pytest.mark.parametrize("hints,version,reason_part", [
    ([hint("3.9", "pipfile"), hint("3.9.18", "pipfile-lock"), hint("3.9", "pyc-cache")], "3.9", "agree"),
    ([hint("3.10", "ci"), hint("3.11", "pipfile")], "3.11", "pipfile wins"),
    ([hint(">=3.10", "requires-python")], "3.10", "lowest version"),
    ([hint("^3.9", "poetry"), hint(">=3.10", "python_requires")], "3.10", "lowest version"),
    ([hint("3.11", "dockerfile"), hint(">=3.9,<3.13", "requires-python")], "3.11", "agree"),
])
def test_python_decided(contracts, hints, version, reason_part):
    plan = plan_build(ctx(python_version_hints=hints), contracts)
    assert (plan.python.status, plan.python.version) == ("decided", version)
    assert reason_part in plan.python.reason


@pytest.mark.parametrize("hints,reason_part", [
    ([], "no Python version hint"),
    ([hint("3.10", "ci"), hint("3.11", "ci"), hint("3.9", "pyc-cache")], "several versions"),
    ([hint("3.8", "pipfile"), hint(">=3.10", "requires-python")], "violates >=3.10"),
    ([hint("<3.12", "requires-python")], "without a lower bound"),
])
def test_python_needs_llm(contracts, hints, reason_part):
    plan = plan_build(ctx(python_version_hints=hints), contracts)
    assert plan.python.status == "needs_llm" and reason_part in plan.python.reason
    assert any(n.item == "python" for n in plan.needs_llm)


def test_python_forced_by_contract(tmp_path):
    c = edit_contracts(tmp_path, lambda d: d["project"].update(python_version="3.12"))
    plan = plan_build(ctx(python_version_hints=[hint("3.9", "pipfile")]), c)
    assert (plan.python.version, plan.python.status) == ("3.12", "decided")


def test_unquoted_python_version_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="quoted string"):
        edit_contracts(tmp_path, lambda d: d["project"].update(python_version=3.10))


# --- serving ---------------------------------------------------------------

def test_serving_generates_when_repo_has_no_app(contracts):
    s = plan_build(ctx(), contracts).serving
    assert (s.status, s.mode, s.framework) == ("decided", "generate", "flask")
    assert s.command == "gunicorn --bind 0.0.0.0:8000 pipeline.serve:app"


def test_serving_uses_conforming_repo_app(contracts):
    s = plan_build(ctx(web_apps=[FULL_APP], third_party_imports=[imp("mlflow", "serve.py")]), contracts).serving
    assert (s.status, s.mode, s.app_target, s.repo_port) == ("decided", "repo", "serve:app", 9000)
    assert s.command == "uvicorn serve:app --host 0.0.0.0 --port 8000"
    assert s.contract_gaps == [] and "response has field 'prediction'" in s.unverified


def test_serving_gaps_need_llm(contracts):
    app = FULL_APP.model_copy(update={"framework": "flask", "routes": FULL_APP.routes[:1]})
    plan = plan_build(ctx(web_apps=[app]), contracts)
    s = plan.serving
    assert (s.status, s.mode) == ("needs_llm", "repo")
    assert s.command == "gunicorn --bind 0.0.0.0:8000 serve:app"
    assert s.contract_gaps[0].startswith("no GET /health route")
    assert "does not import mlflow" in s.contract_gaps[1]
    assert plan.target("serve").status == "needs_llm"


@pytest.mark.parametrize("apps,reason_part", [
    ([FULL_APP, FULL_APP.model_copy(update={"target": "other:app"})], "2 apps serve POST /predict"),
    ([FULL_APP.model_copy(update={"routes": [Route(methods=["GET"], path="/")]})], "none serves POST /predict"),
])
def test_serving_ambiguous(contracts, apps, reason_part):
    s = plan_build(ctx(web_apps=apps), contracts).serving
    assert s.status == "needs_llm" and s.mode is None and reason_part in s.reason


def test_serving_forced_repo_without_app(tmp_path):
    c = edit_contracts(tmp_path, lambda d: d["serving"].update(mode="repo"))
    assert "forces serving.mode=repo" in plan_build(ctx(), c).serving.reason


# --- install ---------------------------------------------------------------

def test_install_prefers_lock_and_adds_extras(contracts):
    deps = [DependencyFile(path="Pipfile", kind="pipfile", packages=[Requirement(name="flask")]),
            DependencyFile(path="Pipfile.lock", kind="pipfile_lock"),
            DependencyFile(path="requirements.txt", kind="requirements", packages=[Requirement(name="x")])]
    imports = [imp("scipy", "train.py", declared=False),          # pipeline code: added
               imp("requests", "nb.ipynb", declared=False),       # notebook only: skipped
               imp("pytest", "tests/test_a.py", declared=False)]  # tests only: skipped
    app = FULL_APP.model_copy(update={"framework": "flask"})
    plan = plan_build(ctx(dependency_files=deps, third_party_imports=imports, web_apps=[app],
                          test_files=["tests/test_a.py"]), contracts)
    assert (plan.install.tool, plan.install.source) == ("pipenv", "Pipfile.lock")
    assert plan.install.extra_packages == ["gunicorn", "mlflow", "scipy"]
    assert plan.target("install").command.endswith("pip install gunicorn mlflow scipy")


@pytest.mark.parametrize("deps,tool,source", [
    ([DependencyFile(path="requirements.txt", kind="requirements", packages=[Requirement(name="a")])],
     "pip", "requirements.txt"),
    ([DependencyFile(path="pyproject.toml", kind="pyproject", packages=[Requirement(name="a")]),
      DependencyFile(path="poetry.lock", kind="poetry_lock")], "poetry", "poetry.lock"),
    ([DependencyFile(path="setup.py", kind="setup_py", packages=[Requirement(name="a")])], "pip", "setup.py"),
])
def test_install_tools(contracts, deps, tool, source):
    i = plan_build(ctx(dependency_files=deps), contracts).install
    assert (i.status, i.tool, i.source) == ("decided", tool, source)


def test_install_without_dependency_files(contracts):
    plan = plan_build(ctx(dependency_files=[]), contracts)
    assert plan.install.status == "needs_llm" and "no dependency file" in plan.install.reason


# --- make targets ----------------------------------------------------------

def test_script_targets_decided_when_they_meet_the_contract(contracts):
    eps = [EntryPoint(path="src/train.py", role="train", has_main_guard=True),
           EntryPoint(path="src/evaluate.py", role="evaluate", has_main_guard=True),
           EntryPoint(path="src/make_dataset.py", role="data", has_main_guard=True)]
    refs = [PathReference(literal="reports/eval.json", files=["src/evaluate.py"], exists=False),
            PathReference(literal="../data/processed/train.parquet", files=["src/make_dataset.py"], exists=False)]
    plan = plan_build(ctx(entry_points=eps, path_references=refs,
                          third_party_imports=[imp("mlflow", "src/train.py")]), contracts)
    for name, cmd in [("train", "python src/train.py"), ("evaluate", "python src/evaluate.py"),
                      ("data", "python src/make_dataset.py")]:
        t = plan.target(name)
        assert (t.status, t.command) == ("decided", cmd), t.reason


def test_script_targets_need_llm_with_reason(contracts):
    eps = [EntryPoint(path="train.py", role="train", has_main_guard=True),
           EntryPoint(path="eval_a.py", role="evaluate", has_main_guard=True),
           EntryPoint(path="eval_b.py", role="evaluate", has_main_guard=True),
           EntryPoint(path="features.py", role="features", has_main_guard=False, functions=["f"])]
    plan = plan_build(ctx(entry_points=eps), contracts)
    train = plan.target("train")
    assert train.status == "needs_llm" and train.command == "python train.py"
    assert "does not import mlflow" in train.reason
    assert "2 runnable scripts" in plan.target("evaluate").reason
    assert "no __main__ block" in plan.target("data").reason
    data_need = next(n for n in plan.needs_llm if n.item == "target:data")
    assert data_need.context == ["features.py [features] no __main__, defs: f"]


def test_all_order_and_unknown_target(tmp_path):
    c = edit_contracts(tmp_path, lambda d: d["make_targets"].update(lint="check code style"))
    plan = plan_build(ctx(), c)
    all_t = plan.target("all")
    assert all_t.depends_on == ["install", "data", "train", "evaluate", "test"]
    assert all_t.command == "$(MAKE) install data train evaluate test"
    assert plan.target("lint").status == "needs_llm"
    assert [t.name for t in plan.make_targets][:7] == list(c.make_targets)[:7]


# --- integration -----------------------------------------------------------

@pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")
def test_taxi_plan(contracts):
    plan = plan_build(scan_repo(TAXI), contracts)
    assert (plan.python.status, plan.python.version) == ("decided", "3.9")
    assert (plan.install.tool, plan.install.extra_packages) == ("pipenv", ["mlflow"])
    s = plan.serving
    assert (s.mode, s.app_target, s.repo_port, s.port) == ("repo", "src.models.predict_model:app", 9696, 8000)
    assert s.command == "gunicorn --bind 0.0.0.0:8000 src.models.predict_model:app"
    assert len(s.contract_gaps) == 2
    status = {t.name: t.status for t in plan.make_targets}
    assert status == {"install": "decided", "data": "needs_llm", "train": "needs_llm",
                      "evaluate": "needs_llm", "serve": "needs_llm", "test": "needs_llm", "all": "decided"}
    assert "no __main__" in plan.target("train").reason
    assert {n.item for n in plan.needs_llm} == {"serving", "target:data", "target:train",
                                                "target:evaluate", "target:test"}
    assert len(plan.summary()) < 3000

    ctx_of = {n.item: n.context for n in plan.needs_llm}
    for item in ("target:train", "target:evaluate", "target:test", "serving"):
        assert any(c.startswith("target candidates: trip_duration") for c in ctx_of[item]), item
        assert any(c.startswith("target transform: log1p on trip_duration") and "expm1 in src/models/predict_model.py" in c
                   for c in ctx_of[item]), item
    assert any("train_xgboost(train_path, target_col, num_rounds=371)" in c for c in ctx_of["target:train"])
    assert any("single_prediction(features, model)" in c for c in ctx_of["serving"])
    assert any(c.startswith("columns of data/processed/") and "trip_duration" in c for c in ctx_of["target:test"])
