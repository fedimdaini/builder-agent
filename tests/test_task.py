"""Regression vs classification: scan signals, the decide rule, slot validation, rendering (step 2c)."""

import ast
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from builder_agent.decide import load_contracts, plan_build
from builder_agent.decide.rules import choose_task
from builder_agent.llm.prompts import load_prompt, slots_model
from builder_agent.render import RenderError, lint_files, render_adapters, validate_slots
from builder_agent.render.slots import SlotAnswers, SlotAnswersV3
from builder_agent.scan import scan_repo
from builder_agent.scan.task import code_signals, target_values

from test_render import CONTRACTS, TAXI, TAXI_GOLD
from test_scan import write

ROOT = Path(__file__).resolve().parents[1]


def signals(code: str) -> set[tuple[str, str, str]]:
    return {(s.kind, s.value, s.task) for s in code_signals(ast.parse(code), "x.py")}


def test_code_signals():
    code = ("import xgboost as xgb\nfrom sklearn.metrics import accuracy_score\n"
            "m = xgb.XGBClassifier(objective='binary:logistic').fit(X, y)\n"
            "print(accuracy_score(y, m.predict(X)))\n"
            "b = xgb.train({'objective': 'reg:squarederror'}, d)\nr = Ridge()\n")
    assert signals(code) == {("model", "XGBClassifier", "classification"),
                             ("objective", "binary:logistic", "classification"),
                             ("metric", "accuracy_score", "classification"),
                             ("objective", "reg:squarederror", "regression"),
                             ("model", "Ridge", "regression")}
    assert signals("from sklearn.metrics import r2_score\n") == set()   # imported, never called


@pytest.mark.parametrize("values,task", [
    (["0", "1"] * 20, "classification"),
    (["yes", "no", "maybe"] * 15, "classification"),
    ([str(1.5 + i * 0.37) for i in range(40)], "regression"),
    ([str(100 + i * 7) for i in range(40)], "regression"),          # many distinct integers: a quantity
    (["1.5", "2.0", "2.5"] * 15, None),                             # few distinct decimals: unclear
    (["0", "1", "1"], None),                                        # too few rows to say anything
])
def test_target_values(tmp_path, values, task):
    write(tmp_path, {"d.csv": "a,label\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values))})
    assert target_values(tmp_path, "d.csv", "label").task == task


def classifier_repo(root: Path, model: str = "LogisticRegression", label=lambda i: i % 2) -> Path:
    rows = "".join(f"{i % 7},{(i * 13) % 11 / 3:.3f},{i % 3},{label(i)}\n" for i in range(60))
    write(root, {
        "requirements.txt": "scikit-learn\npandas\nnumpy\n",
        ".python-version": "3.11\n",
        "data/processed/train.csv": "a,b,g,label\n" + rows,
        "data/processed/test.csv": "a,b,g,label\n" + rows,
        "src/train_model.py": (
            "import pandas as pd\n"
            f"from sklearn.linear_model import {model}\n"
            "def train(train_path, target_col='label'):\n"
            "    df = pd.read_csv(train_path)\n"
            "    y = df[target_col]\n"
            "    X = df.drop(target_col, axis=1)\n"
            f"    return {model}().fit(X, y)\n"
        ),
    })
    return root


GOLD = {"target_column": "label", "target_transform": "none", "transform_inside_train_fn": False,
        "model_flavor": "sklearn", "model_input": "dataframe",
        "train": {"train_function": "src.train_model:train", "train_file": "data/processed/train.csv",
                  "arg_map": {"train_path": "$train_path", "target_col": "$target_column"}},
        "evaluate": {"eval_file": "data/processed/test.csv", "group_column": "g"},
        "data": {"data_step": "existing"}}


def test_rule_decides_when_the_signals_agree(tmp_path):
    ctx = scan_repo(classifier_repo(tmp_path))
    choice, need = choose_task(ctx)
    assert (choice.status, choice.task, need) == ("decided", "classification", None)
    assert any("LogisticRegression" in e for e in choice.evidence) and any("target values" in e for e in choice.evidence)


def test_rule_asks_the_llm_when_the_signals_disagree(tmp_path):
    ctx = scan_repo(classifier_repo(tmp_path, model="LinearRegression"))   # a regressor on 0/1 labels
    plan = plan_build(ctx, load_contracts(CONTRACTS))
    assert plan.task.status == "needs_llm" and "disagree" in plan.task.reason
    assert any(n.item == "task" for n in plan.needs_llm)


def test_rule_asks_the_llm_without_signals(tmp_path):
    write(tmp_path, {"data/d.csv": "a,y\n1,2\n", "src/train.py": "def train(X, y):\n    return None\n"})
    choice, need = choose_task(scan_repo(tmp_path))
    assert choice.status == "needs_llm" and "no task signal" in choice.reason and need.item == "task"


def test_taxi_is_regression():
    if not TAXI.is_dir():
        pytest.skip("taxi not checked out")
    choice, _ = choose_task(scan_repo(TAXI))
    assert (choice.status, choice.task) == ("decided", "regression")


def test_validator_task_rules(tmp_path):
    ctx = scan_repo(classifier_repo(tmp_path))
    assert validate_slots(ctx, GOLD).ok                                   # v2-style: the rule's task
    assert validate_slots(ctx, GOLD | {"task": "classification"}).ok      # v3, agreeing
    r = validate_slots(ctx, GOLD | {"task": "regression"})
    assert not r.ok and "the scan signals decide 'classification'" in r.reasons[0]
    r = validate_slots(ctx, GOLD | {"target_transform": "log1p"})
    assert any("must be 'none'" in x for x in r.reasons)


def test_classification_on_a_continuous_target_is_rejected(tmp_path):
    # a classifier in the code, but continuous target values: the rule asks, and the answer is checked
    ctx = scan_repo(classifier_repo(tmp_path, label=lambda i: f"{i * 1.37:.2f}"))
    choice, _ = choose_task(ctx)
    assert choice.status == "needs_llm"
    assert validate_slots(ctx, GOLD | {"task": "regression"}).ok
    r = validate_slots(ctx, GOLD | {"task": "classification"})
    assert not r.ok and any("is a quantity" in x for x in r.reasons)


def test_classification_adapters_render_clean(tmp_path):
    repo = classifier_repo(tmp_path / "repo")
    ctx = scan_repo(repo)
    out = tmp_path / "out"
    result = render_adapters(ctx, load_contracts(CONTRACTS), GOLD, out)
    assert result.ok, result.lint
    evaluate = (out / "pipeline" / "evaluate.py").read_text(encoding="utf-8")
    serve = (out / "pipeline" / "serve.py").read_text(encoding="utf-8")
    assert '"task": "classification"' in evaluate and "f1_macro" in evaluate and "rmse" not in evaluate
    assert "value.item()" in serve and "np.isfinite" not in serve
    assert not lint_files(out, ["pipeline/evaluate.py", "pipeline/serve.py"])


def test_classification_metrics_are_right(tmp_path):
    """The rendered metrics() and roc_auc(), run on numbers worked out by hand."""
    ctx = scan_repo(classifier_repo(tmp_path / "repo"))
    render_adapters(ctx, load_contracts(CONTRACTS), GOLD, tmp_path / "out")
    tree = ast.parse((tmp_path / "out" / "pipeline" / "evaluate.py").read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("metrics", "roc_auc", "average_ranks")]
    ns = {"np": np}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "evaluate", "exec"), ns)
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    score = np.array([0.9, 0.4, 0.2, 0.3, 0.8, 0.6])
    m = ns["metrics"](y_true, y_pred, score, 1)
    # accuracy 4/6; F1 class 1: tp 2, fp 1, fn 1 -> 4/6; class 0: tp 2, fp 1, fn 1 -> 4/6
    assert m["accuracy"] == pytest.approx(4 / 6) and m["f1_macro"] == pytest.approx(4 / 6)
    # ROC AUC: positive scores 0.9, 0.4, 0.8 vs negatives 0.2, 0.3, 0.6: 8 of 9 pairs ranked right
    assert m["roc_auc"] == pytest.approx(8 / 9)
    assert "roc_auc" not in ns["metrics"](np.array(["a", "b"]), np.array(["a", "a"]))
    # ties share their rank: two positives and two negatives all scored 0.5 -> AUC 0.5
    assert ns["roc_auc"](np.array([True, True, False, False]), np.full(4, 0.5)) == pytest.approx(0.5)


def test_render_refuses_an_undecided_task(tmp_path):
    ctx = scan_repo(classifier_repo(tmp_path / "repo", model="LinearRegression"))   # signals disagree
    with pytest.raises(RenderError, match="task not decided"):
        render_adapters(ctx, load_contracts(CONTRACTS), GOLD, tmp_path / "out")
    assert render_adapters(ctx, load_contracts(CONTRACTS), GOLD | {"task": "classification"},
                           tmp_path / "out2").ok


def test_taxi_render_is_byte_identical_to_the_snapshot():
    """Regression repos render exactly as before step 2c: past fault cases and prompts quote these files."""
    if not TAXI.is_dir():
        pytest.skip("taxi not checked out")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_configs import render_all
    root = Path(tempfile.mkdtemp())
    render_all(scan_repo(TAXI), load_contracts(CONTRACTS), TAXI_GOLD, root)
    want = json.loads((ROOT / "tests" / "gold" / "taxi_rendered_sha256.json").read_text(encoding="utf-8"))
    got = {k: hashlib.sha256((root / k).read_bytes()).hexdigest() for k in want}
    assert got == want


def test_v1_v2_keep_their_schema_and_v3_adds_the_task():
    for v in ("slots_v1", "slots_v2"):
        assert slots_model(load_prompt(v)) is SlotAnswers
    assert slots_model(load_prompt("slots_v3")) is SlotAnswersV3
    v2, v3 = load_prompt("slots_v2"), load_prompt("slots_v3")
    assert v3.sections["system"] == v2.sections["system"] and v3.sections["retry"] == v2.sections["retry"]
    assert "task" not in SlotAnswers.model_json_schema()["properties"]
