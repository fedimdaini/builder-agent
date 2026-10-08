import json
from pathlib import Path

import pytest

from builder_agent.scan import scan_repo
from builder_agent.scan.code import analyze_python, role_for
from builder_agent.scan.deps import parse_requirement
from builder_agent.scan.layout import _matches

LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
    "size 12345678\n"
)


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def notebook(*cells: str) -> str:
    return json.dumps({"cells": [{"cell_type": "code", "source": c} for c in cells]
                       + [{"cell_type": "markdown", "source": "# title"}]})


@pytest.fixture
def fastapi_repo(tmp_path):
    return write(tmp_path, {
        "requirements.txt": "fastapi==0.110\nuvicorn[standard]>=0.29 ; python_version>'3.8'\n"
                            "scikit-learn~=1.4  # model\n-r requirements-dev.txt\n"
                            "-e git+https://github.com/x/y.git#egg=mylib\n",
        "requirements-dev.txt": "pytest\n",
        "pyproject.toml": '[project]\nname="x"\nrequires-python=">=3.11"\ndependencies=["joblib>=1.3"]\n'
                          '[tool.pytest.ini_options]\naddopts="-q"\n',
        ".python-version": "3.11.9\n",
        ".github/workflows/ci.yml": "jobs:\n  t:\n    steps:\n      - uses: actions/setup-python@v5\n"
                                    "        with:\n          python-version: '3.11'\n",
        ".gitattributes": "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
        "app/main.py": (
            "import os\nimport uvicorn\nimport joblib\nfrom fastapi import FastAPI\n"
            "from app.utils import helper\nfrom . import schemas\n"
            "app = FastAPI()\nmodel = joblib.load('artifacts/model.joblib')\n"
            "@app.get('/health')\ndef health(): return {}\n"
            "@app.post('/predict')\nasync def predict(x: dict): return x\n"
            "if __name__ == '__main__':\n"
            "    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))\n"
        ),
        "app/utils.py": "def helper(): pass\n",
        "app/__init__.py": "",
        "app/schemas.py": "",
        "src/models/train.py": (
            "import argparse\nimport pandas as pd\nfrom sklearn.linear_model import Ridge\nimport cv2\n"
            "def main():\n    df = pd.read_csv('data/train.parquet')\n"
            "if __name__ == '__main__':\n    main()\n"
        ),
        "src/models/__init__.py": "",
        "src/legacy.py": "print 'python 2'\n",
        "data/train.parquet": LFS_POINTER,
        "data/raw/.gitkeep": "",
        "artifacts/model.joblib": "binary",
        "notebooks/eda.ipynb": notebook("%matplotlib inline\nimport seaborn as sns\n!pip install foo",
                                        "df = pd.read_csv('../data/train.parquet')", "this is not python ("),
        "tests/test_api.py": "import pytest\n",
        "Dockerfile": "FROM python:3.11-slim\nEXPOSE 8080\nCMD uvicorn app.main:app --port 8080\n",
    })


def test_dependency_files(fastapi_repo):
    ctx = scan_repo(fastapi_repo)
    by_path = {d.path: d for d in ctx.dependency_files}
    assert set(by_path) == {"requirements.txt", "requirements-dev.txt", "pyproject.toml"}
    reqs = {r.name: r.spec for r in by_path["requirements.txt"].packages}
    assert reqs == {"fastapi": "==0.110", "uvicorn": ">=0.29", "scikit-learn": "~=1.4",
                    "mylib": "-e git+https://github.com/x/y.git#egg=mylib"}
    assert all(r.dev for r in by_path["requirements-dev.txt"].packages)
    assert [r.name for r in by_path["pyproject.toml"].packages] == ["joblib"]


def test_python_version_hints(fastapi_repo):
    hints = {(h.kind, h.value) for h in scan_repo(fastapi_repo).python_version_hints}
    assert hints == {("requires-python", ">=3.11"), ("python-version-file", "3.11.9"),
                     ("ci", "3.11"), ("dockerfile", "3.11")}


def test_imports_and_frameworks(fastapi_repo):
    ctx = scan_repo(fastapi_repo)
    third = {t.module: t for t in ctx.third_party_imports}
    assert set(third) == {"uvicorn", "joblib", "fastapi", "pandas", "sklearn", "cv2", "seaborn", "pytest"}
    assert third["sklearn"].distribution == "scikit-learn" and third["sklearn"].declared
    assert third["seaborn"].files == ["notebooks/eda.ipynb"]
    assert {t.distribution for t in ctx.undeclared_imports} == {"pandas", "opencv-python", "seaborn"}
    assert "app" in ctx.local_modules and {"os", "argparse"} <= set(ctx.stdlib_modules)
    fws = {f.name: (f.category, f.via) for f in ctx.frameworks}
    assert fws["scikit-learn"] == ("ml", ["import", "dependency"])
    assert fws["pandas"] == ("data", ["import"])
    assert fws["fastapi"][0] == "serving"


def test_entry_points_and_web_app(fastapi_repo):
    ctx = scan_repo(fastapi_repo)
    eps = {e.path: e for e in ctx.entry_points}
    assert eps["src/models/train.py"].role == "train"
    assert eps["src/models/train.py"].cli == "argparse" and eps["src/models/train.py"].has_main_guard
    assert eps["app/main.py"].role == "main"
    assert "app/utils.py" not in eps and "tests/test_api.py" not in eps

    [web] = ctx.web_apps
    assert (web.framework, web.target, web.port, web.port_env_var) == ("fastapi", "app.main:app", 8080, "PORT")
    assert [(r.methods, r.path) for r in web.routes] == [(["GET"], "/health"), (["POST"], "/predict")]
    assert {(p.port, p.kind) for p in ctx.port_hints} == {
        (8080, "uvicorn.run"), (8080, "dockerfile-expose"), (8080, "port-flag")}


def test_layout_lfs_notebooks(fastapi_repo):
    ctx = scan_repo(fastapi_repo)
    assert [d.path for d in ctx.data_dirs] == ["data"]
    assert ctx.data_dirs[0].subdirs == ["raw"]
    # src/models only holds code, so it is not a model folder
    assert [d.path for d in ctx.model_dirs] == ["artifacts"]
    assert [m.path for m in ctx.model_files] == ["artifacts/model.joblib"]

    assert ctx.lfs.tracked_files == ["data/train.parquet"]
    [ptr] = ctx.lfs.pointer_files
    assert (ptr.path, ptr.object_size) == ("data/train.parquet", 12345678)

    [nb] = ctx.notebooks
    assert (nb.n_code_cells, nb.parse_ok) == (3, False)
    refs = {r.literal: r for r in ctx.path_references}
    assert refs["../data/train.parquet"].resolved == "data/train.parquet"
    assert refs["artifacts/model.joblib"].exists

    assert ctx.test_files == ["tests/test_api.py"] and ctx.has_pytest_config
    assert ctx.existing_pipeline_files == [".github/workflows/ci.yml", "Dockerfile"]
    assert {e.path for e in ctx.parse_errors} == {"src/legacy.py", "notebooks/eda.ipynb"}


def test_summary_is_compact(fastapi_repo):
    s = scan_repo(fastapi_repo).summary()
    assert "fastapi app.main:app | port 8080 (env PORT)" in s
    assert "1 NOT PULLED (pointer files): data/train.parquet" in s
    assert "UNDECLARED IMPORTS: opencv-python (pipeline code, 1 file)" in s
    assert "seaborn (notebooks only, 1 file)" in s
    assert len(s) < 3000


def test_ignores_virtualenv(tmp_path):
    write(tmp_path, {"main.py": "import requests\n",
                     "env/pyvenv.cfg": "home = x\n",
                     "env/lib/site.py": "import torch\n"})
    ctx = scan_repo(tmp_path)
    assert [t.module for t in ctx.third_party_imports] == ["requests"]


def test_flask_positional_port_and_pipfile(tmp_path):
    write(tmp_path, {
        "Pipfile": '[packages]\nflask = "*"\nxgboost = {version = "==2.0"}\n[requires]\npython_version = "3.9"\n',
        "setup.py": "from setuptools import setup\nsetup(name='x', install_requires=open('r.txt').read().split())\n",
        "serve.py": "import flask\napp = flask.Flask(__name__)\n"
                    "@app.route('/p', methods=['GET', 'post'])\ndef p(): pass\napp.run('0.0.0.0', 5001)\n",
    })
    ctx = scan_repo(tmp_path)
    pip = next(d for d in ctx.dependency_files if d.kind == "pipfile")
    assert {(r.name, r.spec) for r in pip.packages} == {("flask", "*"), ("xgboost", "==2.0")}
    setup = next(d for d in ctx.dependency_files if d.kind == "setup_py")
    assert "computed at runtime" in setup.note
    [web] = ctx.web_apps
    assert (web.port, web.routes[0].methods) == (5001, ["GET", "POST"])
    assert ctx.entry_points[0].role == "serve"
    assert [t.module for t in ctx.third_party_imports] == ["flask"]  # setuptools excluded


@pytest.mark.parametrize("line,expected", [
    ("numpy==1.26.4", ("numpy", "==1.26.4")),
    ("pandas>=2,<3 ; python_version >= '3.9'", ("pandas", ">=2,<3")),
    ("torch", ("torch", "")),
    ("# comment", None),
    ("--index-url https://x", None),
])
def test_parse_requirement(line, expected):
    r = parse_requirement(line)
    assert (r.name, r.spec) == expected if expected else r is None


@pytest.mark.parametrize("path,base,pattern,ok", [
    ("data/raw/a.csv", "", "*.csv", True),
    ("data/raw/a.csv", "", "data/raw/a.csv", True),
    ("data/raw/a.csv", "data", "raw/*.csv", True),
    ("other/raw/a.csv", "data", "raw/*.csv", False),
    ("data/big/x.bin", "", "data/big", True),
])
def test_lfs_pattern_matching(path, base, pattern, ok):
    assert _matches(path, base, pattern) is ok


@pytest.mark.parametrize("path,role", [
    ("src/models/train_model.py", "train"), ("predict_model.py", "predict"),
    ("api.py", "serve"), ("evaluate.py", "evaluate"), ("make_dataset.py", "data"),
    ("build_features.py", "features"), ("main.py", "main"), ("utils.py", None),
])
def test_role_for(path, role):
    assert role_for(path) == role


def test_relative_imports_are_ignored():
    facts = analyze_python("from . import a\nfrom ..b import c\nimport x.y\n", "pkg/m.py")
    assert facts.imports == {"x.y"}


TAXI = Path(__file__).resolve().parents[2] / "taxi-trip-regression"


@pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")
def test_taxi_repo():
    ctx = scan_repo(TAXI)
    assert {d.kind for d in ctx.dependency_files} >= {"pipfile", "pipfile_lock", "setup_py"}
    assert {h.value for h in ctx.python_version_hints} == {"3.9"}
    assert {f.name for f in ctx.frameworks if f.category == "ml"} == {"xgboost", "scikit-learn"}
    [web] = ctx.web_apps
    assert (web.framework, web.target, web.port) == ("flask", "src.models.predict_model:app", 9696)
    eps = {e.path: e for e in ctx.entry_points}
    assert eps["src/models/train_model.py"].role == "train"
    assert not eps["src/models/train_model.py"].has_main_guard
    assert [d.path for d in ctx.data_dirs] == ["data"]
    assert [d.path for d in ctx.model_dirs] == ["models"]
    assert len(ctx.notebooks) == 2
    assert len(ctx.lfs.tracked_files) == 7
    [req] = ctx.undeclared_imports
    assert req.distribution == "requests" and req.notebooks_only
    assert "requests (notebooks only, 2 files)" in ctx.summary()
