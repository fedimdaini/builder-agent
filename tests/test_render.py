"""Slot validation and adapter rendering (no LLM; rendered adapters are never run)."""

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from builder_agent.decide import load_contracts
from builder_agent.render import RenderError, py_literal, render_adapters, slot_candidates, validate_slots
from builder_agent.scan import scan_repo

from test_scan import notebook, write

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts.yaml"
TAXI = ROOT.parent / "taxi-trip-regression"
TAXI_GOLD = json.loads((ROOT / "tests/gold/taxi_slots.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contracts():
    return load_contracts(CONTRACTS)


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """A tiny repo shaped like the taxi one: log1p inside the train function, expm1 when serving."""
    root = tmp_path_factory.mktemp("mini")
    write(root, {
        "requirements.txt": "xgboost\nscikit-learn\npandas\nnumpy\nflask\n",
        ".python-version": "3.11\n",
        "data/processed/train.csv": "a,b,flag,price\n1,2.5,True,100\n",
        "data/processed/test.csv": "a,b,flag,price\n3,0.5,False,40\n",
        "data/processed/other.csv": "a,b\n1,2\n",
        "src/train_model.py": (
            "import numpy as np\nimport pandas as pd\nimport xgboost as xgb\n"
            "def train(train_path, target_col, rounds=10):\n"
            "    df = pd.read_csv(train_path)\n"
            "    y = np.log1p(df[target_col].values)\n"
            "    X = df.drop(target_col, axis=1).values\n"
            "    return xgb.train({}, xgb.DMatrix(X, label=y), rounds)\n"
            "def fit_arrays(X, y):\n"
            "    return xgb.XGBRegressor().fit(X, y)\n"
        ),
        "src/alt_train.py": "def train_alt(train_path, target_col):\n    pass\n",
        "src/predict_model.py": "import numpy as np\ndef predict(model, X):\n    return np.expm1(model.predict(X))\n",
        "src/make_data.py": "def build():\n    pass\n",
        "notebooks/nb.ipynb": notebook("from src.train_model import train\n"
                                       "train('data/processed/train.csv', 'price')"),
    })
    return scan_repo(root)


MINI_GOLD = {
    "target_column": "price",
    "target_transform": "log1p",
    "transform_inside_train_fn": True,
    "model_flavor": "xgboost",
    "model_input": "dmatrix",
    "train": {"train_function": "src.train_model:train", "train_file": "data/processed/train.csv",
              "arg_map": {"train_path": "$train_path", "target_col": "$target_column"}},
    "evaluate": {"eval_file": "data/processed/test.csv", "group_column": "flag"},
    "data": {"data_step": "existing"},
}


def changed(base: dict, path: str, value) -> dict:
    """Copy of base with one dotted path set (value=... deletes it)."""
    out = copy.deepcopy(base)
    *parents, key = path.split(".")
    node = out
    for p in parents:
        node = node[p]
    if value is ...:
        del node[key]
    else:
        node[key] = value
    return out


# --- candidates and validation ----------------------------------------------

def test_candidates_come_from_scan(mini):
    cand = slot_candidates(mini)
    assert cand["target_column"][0] == "price"
    assert cand["target_transform"] == ["none", "log1p"]
    assert cand["model_flavor"] == ["xgboost", "sklearn"]
    assert cand["train_function"]["src.train_model:train"] == "train(train_path, target_col, rounds=10)"
    assert "src.make_data:build" in cand["data_step"]
    assert "src.train_model:train" not in cand["data_step"]   # has required parameters


def test_mini_gold_validates(mini):
    v = validate_slots(mini, MINI_GOLD)
    assert v.ok, v.reasons


@pytest.mark.parametrize("path,value,reason", [
    ("target_column", "duration", "is not a scanned target candidate"),
    ("evaluate.eval_file", "data/processed/other.csv", "'price' is not in the header of data/processed/other.csv"),
    ("train.train_file", "data/missing.csv", "is not a data file with a readable header"),
    ("target_transform", "sqrt", "was not found in the code"),
    ("model_flavor", "lightgbm", "which the repo doesn't use"),
    ("train.train_function", "src.train_model:nope", "is not a repo function"),
    ("train.arg_map.target_col", ..., "required parameter(s) not mapped: target_col"),
    ("train.arg_map.epochs", 5, "has no parameter epochs"),
    ("train.arg_map.target_col", "$labels", "unknown token '$labels'"),
    ("evaluate.group_column", "zzz", "'zzz' is not in the header"),
    ("evaluate.group_column", "price", "can't be the target column"),
    ("data.data_step", "src.train_model:train", "must be 'existing' or a repo function"),
    ("transform_inside_train_fn", False, "it would train on the raw target"),
    ("transform_inside_train_fn", "yes", "transform_inside_train_fn: Input should be a valid boolean"),
    ("surprise", 1, "surprise: Extra inputs are not permitted"),
])
def test_rejections(mini, path, value, reason):
    v = validate_slots(mini, changed(MINI_GOLD, path, value))
    assert not v.ok and v.slots is None
    assert any(reason in r for r in v.reasons), v.reasons


def test_rejects_input_that_does_not_fit_flavor(mini):
    answer = changed(changed(MINI_GOLD, "model_flavor", "sklearn"), "model_input", "dmatrix")
    v = validate_slots(mini, answer)
    assert any("doesn't fit flavor sklearn" in r for r in v.reasons), v.reasons


def test_rejects_transform_inside_a_function_that_does_not_apply_it(mini):
    answer = changed(MINI_GOLD, "train.train_function", "src.alt_train:train_alt")
    v = validate_slots(mini, answer)
    assert any("log1p is not applied in src/alt_train.py" in r for r in v.reasons), v.reasons


def test_rejects_inside_flag_without_transform(mini):
    v = validate_slots(mini, changed(MINI_GOLD, "target_transform", "none"))
    assert any("target_transform is 'none'" in r for r in v.reasons), v.reasons


# --- rendering ---------------------------------------------------------------

def render_text(result, root: Path) -> dict[str, str]:
    return {Path(f).name: (root / f).read_text(encoding="utf-8") for f in result.files}


def test_mini_renders_clean(mini, contracts, tmp_path):
    r = render_adapters(mini, contracts, MINI_GOLD, tmp_path)
    assert r.ok, r.lint
    files = render_text(r, tmp_path)
    assert "{{" not in "".join(files.values()) and "{%" not in "".join(files.values())
    assert json.loads(files["sample_request.json"]) == {"a": 3, "b": 0.5, "flag": False}


def test_y_transform_applied_when_function_does_not(mini, contracts, tmp_path):
    answer = changed(MINI_GOLD, "transform_inside_train_fn", False)
    answer = changed(answer, "train.train_function", "src.train_model:fit_arrays")
    answer = changed(answer, "train.arg_map", {"X": "$X", "y": "$y"})
    answer = changed(answer, "model_input", "numpy")
    r = render_adapters(mini, contracts, answer, tmp_path)
    assert r.ok, r.lint
    files = render_text(r, tmp_path)
    assert "y = np.log1p(df[TARGET])" in files["train.py"]
    assert "model = fit_arrays(X=X, y=y)" in files["train.py"]
    assert "import xgboost as xgb" not in files["serve.py"]   # numpy input: no DMatrix, no unused import
    assert "raw = model.predict(X.values)" in files["serve.py"]


def test_data_step_function(mini, contracts, tmp_path):
    r = render_adapters(mini, contracts, changed(MINI_GOLD, "data.data_step", "src.make_data:build"), tmp_path)
    assert r.ok, r.lint
    assert "from src.make_data import build" in render_text(r, tmp_path)["data.py"]


def test_render_refuses_invalid_answers(mini, contracts, tmp_path):
    with pytest.raises(RenderError, match="invalid slot answers: target_column 'duration'"):
        render_adapters(mini, contracts, changed(MINI_GOLD, "target_column", "duration"), tmp_path)


def test_overwrite_guard(mini, contracts, tmp_path):
    render_adapters(mini, contracts, MINI_GOLD, tmp_path)
    render_adapters(mini, contracts, MINI_GOLD, tmp_path)          # unchanged: regenerating is fine
    train = tmp_path / "pipeline/train.py"
    train.write_text(train.read_text(encoding="utf-8") + "# my tweak\n", encoding="utf-8")
    with pytest.raises(RenderError, match="pipeline/train.py: it was edited after the Builder wrote it"):
        render_adapters(mini, contracts, MINI_GOLD, tmp_path)

    other = tmp_path / "other"
    write(other, {"pipeline/serve.py": "# hand-written\n"})
    with pytest.raises(RenderError, match="pipeline/serve.py: it was not generated by the Builder"):
        render_adapters(mini, contracts, MINI_GOLD, other)
    assert (other / "pipeline/serve.py").read_text(encoding="utf-8") == "# hand-written\n"
    assert not (other / "pipeline/train.py").exists()               # nothing written on refusal


def test_unsupported_contract_metric(mini, tmp_path):
    data = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    data["formats"]["eval_report"]["metrics"]["mape"] = "float"
    p = tmp_path / "contracts.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RenderError, match="mape"):
        render_adapters(mini, load_contracts(p), MINI_GOLD, tmp_path / "out")


def test_py_literal():
    assert py_literal("it's") == '"it\'s"'
    assert py_literal(["a", 1, True, None]) == '["a", 1, True, None]'
    assert py_literal(["x" * 30] * 3).startswith('[\n    "xxx')


# --- taxi gold ---------------------------------------------------------------

taxi_only = pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")


@pytest.fixture(scope="module")
def taxi():
    return scan_repo(TAXI)


@taxi_only
def test_taxi_gold_validates(taxi):
    v = validate_slots(taxi, TAXI_GOLD)
    assert v.ok, v.reasons


@taxi_only
def test_taxi_gold_renders_clean(taxi, contracts, tmp_path):
    r = render_adapters(taxi, contracts, TAXI_GOLD, tmp_path)
    assert r.ok, r.lint
    assert r.files == ["pipeline/train.py", "pipeline/evaluate.py", "pipeline/serve.py",
                       "pipeline/data.py", "pipeline/sample_request.json"]
    f = render_text(r, tmp_path)

    header = next(d.columns for d in taxi.data_columns if d.path == "data/processed/train_large.csv")
    features = [c for c in header if c != "trip_duration"]

    # sample request: row 1 of test.csv (the reference-run row, ~531 s), features only, by name
    sample = json.loads(f["sample_request.json"])
    assert list(sample) == features and "trip_duration" not in sample
    assert sample["geodesic_distance"] == 1484.8959727054666 and sample["store_and_fwd_flag"] is False

    # train: repo function, MLflow, sample mode
    assert "from src.models.train_model import train_xgboost" in f["train.py"]
    assert "model = train_xgboost(train_path=train_path, target_col=TARGET)" in f["train.py"]
    assert "np.log1p" not in f["train.py"]          # train_xgboost applies log1p itself
    assert 'os.environ.get("SAMPLE", "0") == "1"' in f["train.py"]
    assert "mlflow.xgboost.log_model(model" in f["train.py"]

    # serve: by name in training order, inverse transform, contract endpoints and errors
    serve = f["serve.py"]
    assert "FEATURES = [\n" + "".join(f'    "{c}",\n' for c in features) + "]" in serve
    assert "X = X[FEATURES]" in serve and "np.expm1(" in serve
    assert '@app.route("/health", methods=["GET"])' in serve
    assert '@app.route("/predict", methods=["POST"])' in serve
    errors = re.findall(r'jsonify\(\{"error".*?\}\), (\d+)', serve)
    assert errors and all(code in {"400", "500"} for code in errors)

    # evaluate: eval.json in the contract format
    ev = f["evaluate.py"]
    assert '"model_version": version, "metrics": metrics(y_true, y_pred), "per_group": per_group' in ev
    assert 'EVAL_REPORT = "reports/eval.json"' in ev and 'GROUP_COLUMN = "vendor_id"' in ev
