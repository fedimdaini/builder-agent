"""CI workflow (create mode): rules from scanner facts, needs_llm for the rest, YAML + actionlint before writing."""

import shutil
import subprocess

import pytest
import yaml

from builder_agent.decide import plan_build
from builder_agent.render import RenderError
from builder_agent.render.ci import CI_PATH, classify_tests, plan_ci, render_ci, validate_workflow
from builder_agent.repo_writer import RepoWriter
from builder_agent.scan import scan_repo
from builder_agent.verify.source import RepoSource

from test_render import contracts  # noqa: F401 (fixture)
from test_repo_rule import git_repo

PIPFILE = ('[packages]\nflask = "*"\nboto3 = "*"\n[dev-packages]\npylint = "*"\n'
           '[requires]\npython_version = "3.11"\n')
PRECOMMIT = ("repos:\n- repo: local\n  hooks:\n    - id: pylint\n      entry: pylint\n      language: system\n"
             "    - id: mystery\n      entry: mytool --x\n      language: system\n")


@pytest.fixture
def proj(tmp_path):
    return git_repo(tmp_path / "proj", {
        "Pipfile": PIPFILE,
        ".pre-commit-config.yaml": PRECOMMIT,
        "api/app.py": "import boto3\nfrom flask import Flask\napp = Flask(__name__)\n",
        "api/requirements.txt": "flask\nboto3\n",
        "api/test_app.py": "from app import app\n\ndef test_ok():\n    assert app\n",
        "api/test_live.py": "import requests\n\ndef test_live():\n    requests.get('http://x')\n",
        "core/model.py": "def f():\n    return 1\n",
        "core/test_model.py": "from model import f\n\ndef test_f():\n    assert f() == 1\n",
        "api/Dockerfile": "FROM python:3.11-slim\n",
        "tools/Dockerfile": "FROM alpine:3.20\n",
        "docker-compose.yaml": "services:\n  api:\n    build:\n      context: ./api\n    ports:\n"
                               "      - \"${API_PORT}:8000\"\n    environment:\n      - KEY=${KEY:-dev}\n",
    })


def test_classify_tests_follows_local_imports(proj):
    unit, service = classify_tests(RepoSource(proj), ["api/test_app.py", "api/test_live.py", "core/test_model.py"])
    assert unit == ["core/test_model.py"]
    assert service == {"api/test_live.py": "imports requests",
                       "api/test_app.py": "imports app (api/app.py), which imports boto3"}


def test_plan_from_facts(proj, contracts):  # noqa: F811
    ctx = scan_repo(proj)
    ci = plan_ci(ctx, contracts, plan_build(ctx, contracts))
    assert ci.python_version == "3.11"
    assert ci.install[-1] == "pipenv install --dev --system --skip-lock"     # pylint hook needs dev deps
    assert ci.unit_tests == ["core/test_model.py"] and ci.test_install == ["pip install pytest"]   # not declared
    needs = {n.item: n for n in ci.needs_llm}
    assert set(needs) == {"ci:pre-commit", "ci:service-tests", "ci:compose-env"}
    assert "mytool" in needs["ci:pre-commit"].reason
    assert needs["ci:compose-env"].context == ["API_PORT"]                  # KEY has a default
    builds = {b.dockerfile: b.context for b in ci.builds}
    assert builds == {"api/Dockerfile": "api", "tools/Dockerfile": "tools"}   # api's context from compose
    assert ci.precommit and ci.compose_files == ["docker-compose.yaml"]


def test_pytest_installed_when_not_declared(tmp_path, contracts):  # noqa: F811
    repo = git_repo(tmp_path / "p", {"requirements.txt": "numpy\n", ".python-version": "3.12\n",
                                     "lib/requirements.txt": "pandas\n", "lib/calc.py": "x = 1\n",
                                     "lib/test_calc.py": "from calc import x\n\ndef test_x():\n    assert x\n"})
    ctx = scan_repo(repo)
    ci = plan_ci(ctx, contracts, plan_build(ctx, contracts))
    assert ci.test_install == ["pip install -r lib/requirements.txt", "pip install pytest"]
    assert not ci.needs_llm


def test_smoke_test_when_data_is_committed_and_needs_llm_when_in_lfs(tmp_path, contracts):  # noqa: F811
    files = {"requirements.txt": "numpy\n", ".python-version": "3.12\n", "data/processed/train.csv": "a,y\n1,2\n"}
    plain = scan_repo(git_repo(tmp_path / "plain", files))
    created = {"pipeline/smoke_test.py": ""}
    ci = plan_ci(plain, contracts, plan_build(plain, contracts), created)
    assert ci.smoke == ["make data train evaluate SAMPLE=1", "make test"] and not ci.needs_llm

    lfs = scan_repo(git_repo(tmp_path / "lfs", files | {".gitattributes": "*.csv filter=lfs\n"}))
    ci = plan_ci(lfs, contracts, plan_build(lfs, contracts), created)
    assert not ci.smoke and [n.item for n in ci.needs_llm] == ["ci:smoke-data"]


def test_render_validates_and_writes_through_the_writer(proj, contracts, tmp_path):  # noqa: F811
    ctx = scan_repo(proj)
    r = render_ci(ctx, contracts, plan_build(ctx, contracts), proj, writer=RepoWriter(proj))
    assert r.yaml_ok and r.actionlint == []
    doc = yaml.safe_load((proj / CI_PATH).read_text(encoding="utf-8"))
    test_steps = [s.get("name", s.get("uses")) for s in doc["jobs"]["test"]["steps"]]
    assert test_steps == ["actions/checkout@v4.2.2", "actions/setup-python@v5.6.0", "Install dependencies",
                          "Unit tests (pytest)", "Install pre-commit",
                          "Pre-commit on the files this pull request changes", "Pre-commit on all files (push to main)"]
    steps = {s.get("name", s.get("uses")): s for s in doc["jobs"]["test"]["steps"]}
    assert steps["actions/checkout@v4.2.2"]["with"] == {"fetch-depth": 0}
    pr = steps["Pre-commit on the files this pull request changes"]
    assert pr["if"] == "github.event_name == 'pull_request'" and pr["env"] == {"BASE_REF": "${{ github.base_ref }}"}
    assert pr["run"] == 'pre-commit run --from-ref "origin/$BASE_REF" --to-ref HEAD --show-diff-on-failure'
    push = steps["Pre-commit on all files (push to main)"]
    assert push["if"] == "github.event_name == 'push'" and "--all-files" in push["run"]

    docker = {s.get("name", s.get("uses")): s for s in doc["jobs"]["docker"]["steps"]}
    assert "if" not in docker["Validate compose files (always)"]
    assert docker["Validate compose files (always)"]["run"].strip() == "docker compose -f docker-compose.yaml config -q"
    api = docker["Build api/Dockerfile"]
    assert api["if"] == "steps.changes.outputs.build_api == 'true'"
    assert api["run"] == "docker build -f api/Dockerfile -t ci/api:ci api"
    assert 'check build_api "api/Dockerfile" "api/"' in docker["Which images need rebuilding"]["run"]
    assert doc["jobs"]["test"]["runs-on"] == "ubuntu-24.04"
    text = (proj / CI_PATH).read_text(encoding="utf-8")
    assert "#   - ci:service-tests:" in text and "#       api/test_live.py: imports requests" in text

    with pytest.raises(RenderError, match="refusing to overwrite .github/workflows/ci.yml"):   # second run
        render_ci(ctx, contracts, plan_build(ctx, contracts), proj, writer=RepoWriter(proj))


def test_invalid_workflow_is_never_written():
    ok, problems, _ = validate_workflow("name: ci\non: push\njobs:\n  t:\n    runs-on: ubuntu-24.04\n"
                                        "    steps:\n      - run: echo ${{ github.nope }}\n")
    assert ok and any("nope" in p for p in problems)
    ok, problems, _ = validate_workflow("name: ci\n  group: x  cancel: true\n")
    assert not ok and problems


# --- the change check, run for real in bash ------------------------------------------------

def _check_function(proj, contracts):  # noqa: F811
    ctx = scan_repo(proj)
    render_ci(ctx, contracts, plan_build(ctx, contracts), proj, writer=RepoWriter(proj))
    doc = yaml.safe_load((proj / CI_PATH).read_text(encoding="utf-8"))
    script = next(s["run"] for s in doc["jobs"]["docker"]["steps"] if s.get("id") == "changes")
    return script[script.index("check() {"):script.index("}\n", script.index("check() {")) + 2], script


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
@pytest.mark.parametrize("changed,expected", [
    ("api/app.py", {"build_api": "true", "build_tools": "false"}),
    ("tools/Dockerfile\nREADME.md", {"build_api": "false", "build_tools": "true"}),
    ("README.md", {"build_api": "false", "build_tools": "false"}),
    ("", {"build_api": "false", "build_tools": "false"}),
    ("*", {"build_api": "true", "build_tools": "true"}),            # first push of a branch
    ("api-old/x.py", {"build_api": "false", "build_tools": "false"}),  # prefix lookalike
])
def test_rendered_change_check(proj, contracts, tmp_path, changed, expected):  # noqa: F811
    func, script = _check_function(proj, contracts)
    calls = "\n".join(line.strip() for line in script.splitlines() if line.strip().startswith("check build_"))
    out = tmp_path / "out.txt"
    script_file = tmp_path / "check.sh"         # a file, not `bash -c`: no Windows command-line quoting
    script_file.write_bytes(f'GITHUB_OUTPUT="{out.as_posix()}"\nchanged=$(printf "{changed}")\n{func}\n{calls}\n'
                            .encode())
    # full path: on Windows a bare "bash" can resolve to WSL's System32\bash.exe instead of Git Bash
    subprocess.run([shutil.which("bash"), script_file.as_posix()], check=True)
    got = dict(line.split("=") for line in out.read_text().splitlines())
    assert got == expected
