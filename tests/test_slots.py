"""Slot candidates: data columns, target candidates, target transforms, signatures."""

import ast
from pathlib import Path

import pytest

from builder_agent.scan import scan_repo
from builder_agent.scan.targets import signatures

from test_scan import LFS_POINTER, notebook, write

TAXI = Path(__file__).resolve().parents[2] / "taxi-trip-regression"


@pytest.fixture
def ml_repo(tmp_path):
    return write(tmp_path, {
        "data/processed/train.csv": "﻿sqft,rooms,price\n120,3,250000\n",
        "data/processed/test.tsv": "sqft\trooms\tprice\n",
        "data/raw/big.csv": LFS_POINTER,
        "src/train.py": (
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
            "TARGET = 'price'\n"
            "def fit_model(train_path, target_col, *, rounds=10):\n"
            "    '''Fit the model.\n\n    More text.'''\n"
            "    df = pd.read_csv(train_path)\n"
            "    y = np.log1p(df[TARGET])\n"
            "    X = df.drop(columns=['price', 'id'])\n"
            "    plt.plot(X, label='Train')\n"
            "    return model.fit(X, y)\n"
            "def run(*args, **kwargs): pass\n"
        ),
        "src/serve.py": "import numpy as np\ndef predict(m, x):\n    return np.expm1(m.predict(x))\n",
        "notebooks/nb.ipynb": notebook("from src.train import fit_model\n"
                                       "fit_model('data/processed/train.csv', 'price')"),
    })


def test_data_columns(ml_repo):
    cols = {d.path: d.columns for d in scan_repo(ml_repo).data_columns}
    # BOM stripped, TSV split on tabs, LFS pointer file not read as a header
    assert cols == {"data/processed/train.csv": ["sqft", "rooms", "price"],
                    "data/processed/test.tsv": ["sqft", "rooms", "price"]}


def test_target_candidates(ml_repo):
    ctx = scan_repo(ml_repo)
    best = ctx.target_candidates[0]
    assert (best.column, best.in_data) == ("price", True)
    kinds = {e.split(":")[0] for e in best.evidence}
    assert kinds == {"target-name", "y-assign", "fit", "drop", "target-param"}
    names = {c.column for c in ctx.target_candidates}
    assert "Train" not in names        # plot label, not a target
    assert "id" not in names           # drop-only and not in any data header


def test_target_transform(ml_repo):
    [t] = scan_repo(ml_repo).target_transforms
    assert (t.forward, t.inverse) == ("log1p", "expm1")
    assert t.applied_to == ["price"]
    assert (t.forward_files, t.inverse_files) == (["src/train.py"], ["src/serve.py"])


def test_transform_without_inverse_is_flagged(tmp_path):
    write(tmp_path, {"train.py": "import numpy as np\ny_train = np.log1p(df['target'])\n"})
    ctx = scan_repo(tmp_path)
    [t] = ctx.target_transforms
    assert t.inverse_files == []
    assert "inverse expm1 NEVER applied" in ctx.summary()


def test_transform_on_non_target_is_ignored(tmp_path):
    write(tmp_path, {"f.py": "import numpy as np\ndist = np.log1p(df['distance'])\ny = df['price']\n"})
    assert scan_repo(tmp_path).target_transforms == []


def test_signatures():
    tree = ast.parse("def f(a, /, b, c=1, *args, d, e='a very long default value', **kw):\n"
                     "    '''Doc line.'''\nasync def g(*, x): pass\n")
    f, g = signatures(tree)
    assert f.render() == "f(a, b, c=1, *args, d, e='a very long defa..., **kw)"
    assert f.doc == "Doc line."
    assert g.render() == "g(*, x)"


def test_entry_point_signatures_in_summary(ml_repo):
    ctx = scan_repo(ml_repo)
    train = next(e for e in ctx.entry_points if e.path == "src/train.py")
    assert train.signatures[0].render() == "fit_model(train_path, target_col, *, rounds=10)"
    assert train.signatures[0].doc == "Fit the model."
    s = ctx.summary()
    assert "fit_model(train_path, target_col, *, rounds=10)" in s
    assert "TARGET CANDIDATES: price (score" in s
    assert "TARGET TRANSFORM: log1p on price" in s


@pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")
def test_taxi_slots():
    ctx = scan_repo(TAXI)
    processed = {d.path: d.columns for d in ctx.data_columns if d.path.startswith("data/processed/")}
    assert len(processed) == 4 and all(len(c) == 25 and "trip_duration" in c for c in processed.values())

    assert "trip_duration" in {c.column for c in ctx.target_candidates}
    assert ctx.target_candidates[0].column == "trip_duration"

    t = next(t for t in ctx.target_transforms if t.forward == "log1p")
    assert t.inverse == "expm1"
    assert "trip_duration" in t.applied_to
    assert "src/models/train_model.py" in t.forward_files
    assert t.inverse_files == ["src/models/predict_model.py"]

    train = next(e for e in ctx.entry_points if e.path == "src/models/train_model.py")
    assert [s.render() for s in train.signatures] == ["train_xgboost(train_path, target_col, num_rounds=371)"]
