"""Config templates: Dockerfile, .dockerignore, Makefile, compose base file. Nothing is built or run."""

import json
from pathlib import Path

import pytest
import yaml

from builder_agent.decide import plan_build
from builder_agent.render import RenderError, render_adapters
from builder_agent.render.configs import _instructions, config_context, lint_configs, render_configs
from builder_agent.repo_writer import RepoWriter
from builder_agent.scan import scan_repo

from test_render import MINI_GOLD, TAXI, TAXI_GOLD, contracts, mini  # noqa: F401 (fixtures)


def render_all(ctx, contracts, slots, root):
    """One Builder run: adapters and configs share a writer (and its manifest)."""
    plan = plan_build(ctx, contracts)
    writer = RepoWriter(root)
    render_adapters(ctx, contracts, slots, root, writer=writer)
    result = render_configs(ctx, contracts, plan, slots, root, writer=writer)
    return plan, result


def read(root, name):
    return (root / name).read_text(encoding="utf-8")


# --- taxi --------------------------------------------------------------------

taxi_only = pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")


@pytest.fixture(scope="module")
def taxi_render(contracts, tmp_path_factory):
    root = tmp_path_factory.mktemp("taxi")
    ctx = scan_repo(TAXI)
    plan, result = render_all(ctx, contracts, TAXI_GOLD, root)
    return ctx, plan, result, root


@taxi_only
def test_taxi_configs_lint_clean(taxi_render):
    _, _, result, _ = taxi_render
    assert result.files == ["Dockerfile", ".dockerignore", "Makefile", "compose.base.yml"]
    assert result.ok, result.lint


@taxi_only
def test_taxi_dockerfile(taxi_render):
    *_, root = taxi_render
    ins = _instructions(read(root, "Dockerfile"))
    kinds = [k for k, _ in ins]
    assert ins[0] == ("FROM", "python:3.9-slim")
    deps = ins.index(("COPY", "Pipfile Pipfile.lock ./"))
    install = next(i for i, (k, a) in enumerate(ins) if k == "RUN" and "pipenv install --deploy --system" in a)
    code = ins.index(("COPY", ". ."))
    assert deps < install < code                                   # dependencies cached before code
    assert "pip install mlflow==2.17.2" in ins[install][1]          # pinned to the server image (fault-001)
    assert kinds.count("COPY") == 2                                # nothing else copied (no data)
    assert ("EXPOSE", "8000") in ins
    assert ins[-1] == ("CMD", '["gunicorn", "--bind", "0.0.0.0:8000", "pipeline.serve:app"]')
    assert any(k == "RUN" and "apt-get install -y --no-install-recommends make" in a for k, a in ins)


@taxi_only
def test_taxi_dockerignore(taxi_render):
    *_, root = taxi_render
    lines = read(root, ".dockerignore").splitlines()
    for entry in ("data/", "models/", "**/*.ipynb", ".git", "logs/", "reports/"):
        assert entry in lines
    assert "src/" not in lines and "pipeline/" not in lines


@taxi_only
def test_taxi_makefile(taxi_render):
    *_, root = taxi_render
    make = read(root, "Makefile")
    assert "SAMPLE ?= 0\nexport SAMPLE\n" in make
    assert ".PHONY: install data train evaluate serve test all\n" in make
    for target, cmd in [("data", "$(PYTHON) pipeline/data.py"), ("train", "$(PYTHON) pipeline/train.py"),
                        ("evaluate", "$(PYTHON) pipeline/evaluate.py"),
                        ("serve", "gunicorn --bind 0.0.0.0:8000 pipeline.serve:app"),
                        ("all", "$(MAKE) install data train evaluate test")]:
        assert f"\n{target}:\n\t{cmd}\n" in make, target
    assert ("\ninstall:\n\tpip install pipenv==2023.12.1\n\tpipenv install --deploy --system\n"
            "\tpip install mlflow==2.17.2\n") in make
    assert "\ntest:\n\t$(PYTHON) pipeline/smoke_test.py\n" in make


@taxi_only
def test_taxi_compose(taxi_render):
    *_, root = taxi_render
    compose = yaml.safe_load(read(root, "compose.base.yml"))
    mlflow, serving = compose["services"]["mlflow"], compose["services"]["serving-current"]
    assert mlflow["image"] == "ghcr.io/mlflow/mlflow:v2.17.2" and mlflow["ports"] == ["5000:5000"]
    assert serving["volumes"] == ["./data:/app/data:ro", "./models:/app/models", "./logs:/app/logs"]
    assert serving["environment"] == {"MLFLOW_TRACKING_URI": "http://mlflow:5000", "MODEL_URI": "${MODEL_URI:-}"}
    assert serving["ports"] == ["8000:8000"] and serving["depends_on"] == ["mlflow"]
    assert "http://localhost:8000/health" in serving["healthcheck"]["test"][-1]
    assert "router" not in compose["services"]                     # the Deployer's, not ours


@taxi_only
def test_manifest_covers_adapters_and_configs(taxi_render):
    *_, root = taxi_render
    files = json.loads(read(root, "pipeline/.builder-manifest.json"))["files"]
    assert set(files) == {"Dockerfile", ".dockerignore", "Makefile", "compose.base.yml", "pipeline/train.py",
                          "pipeline/evaluate.py", "pipeline/serve.py", "pipeline/data.py",
                          "pipeline/sample_request.json", "pipeline/smoke_test.py"}


# --- the linter catches broken files -------------------------------------------

@taxi_only
@pytest.mark.parametrize("file,old,new,problem", [
    ("Dockerfile", "COPY Pipfile Pipfile.lock ./\n", "", "must be copied and installed before `COPY . .`"),
    ("Dockerfile", "COPY . .\n", "COPY . .\nCOPY data/processed ./data/processed\n", "copies data (data/processed)"),
    ("Dockerfile", "EXPOSE 8000", "EXPOSE 9696", "EXPOSE 8000 missing"),
    (".dockerignore", "\ndata/\n", "\n", ".dockerignore: data dir data/ is not excluded"),
    ("Makefile", "\tpip install mlflow==2.17.2", "    pip install mlflow==2.17.2", "recipe line must start with a tab"),
    ("Makefile", "\nevaluate:\n", "\nevaluation:\n", "no rule for contract target evaluate"),
    ("Makefile", "export SAMPLE\n", "", "SAMPLE is not exported"),
    ("compose.base.yml", "./data:/app/data:ro", "./data:/app/data", "data dir data is not mounted read-only"),
    ("compose.base.yml", "  mlflow:\n", "  tracking:\n", "service mlflow missing"),
])
def test_lint_catches(taxi_render, contracts, tmp_path, file, old, new, problem):
    ctx, plan, _, root = taxi_render
    for name in ("Dockerfile", ".dockerignore", "Makefile", "compose.base.yml"):
        text = read(root, name)
        if name == file:
            assert old in text
            text = text.replace(old, new)
        (tmp_path / name).write_text(text, encoding="utf-8", newline="\n")
    tctx = config_context(ctx, contracts, plan, validate(ctx))
    problems = lint_configs(tmp_path, tctx)
    assert any(problem in p for p in problems), problems


def validate(ctx):
    from builder_agent.render import validate_slots
    return validate_slots(ctx, TAXI_GOLD).slots


# --- other repo shapes and refusals ---------------------------------------------

def test_mini_requirements_and_generated_service(mini, contracts, tmp_path):  # noqa: F811
    plan, result = render_all(mini, contracts, MINI_GOLD, tmp_path)
    assert plan.serving.mode == "generate"                          # mini has no web app
    assert result.ok, result.lint
    docker = read(tmp_path, "Dockerfile")
    assert "FROM python:3.11-slim" in docker
    assert "COPY requirements.txt ./\nRUN pip install -r requirements.txt \\\n && pip install " in docker
    install = next(line for line in docker.splitlines() if line.startswith(" && pip install"))
    assert {"mlflow==2.17.2", "gunicorn"} <= set(install.split()) and "flask" not in install.split()  # flask declared
    assert "\nserve:\n\tgunicorn --bind 0.0.0.0:8000 pipeline.serve:app\n" in read(tmp_path, "Makefile")


def test_refuses_undecided_python(mini, contracts, tmp_path):  # noqa: F811
    from builder_agent.scan.models import VersionHint
    clash = [VersionHint(value=v, source=".github/workflows/ci.yml", kind="ci") for v in ("3.10", "3.11")]
    plan = plan_build(mini.model_copy(update={"python_version_hints": clash}), contracts)
    with pytest.raises(RenderError, match="Python version not decided"):
        render_configs(mini, contracts, plan, MINI_GOLD, tmp_path)


def test_no_declared_python_renders_the_assumed_default(mini, contracts, tmp_path):  # noqa: F811
    plan = plan_build(mini.model_copy(update={"python_version_hints": []}), contracts)
    assert plan.python.assumed
    render_configs(mini, contracts, plan, MINI_GOLD, tmp_path)
    assert "FROM python:3.11-slim" in (tmp_path / "Dockerfile").read_text(encoding="utf-8")


def test_refuses_generated_service_without_slots(mini, contracts, tmp_path):  # noqa: F811
    with pytest.raises(RenderError, match="serve.py adapter needs slot answers"):
        render_configs(mini, contracts, plan_build(mini, contracts), None, tmp_path)


def test_never_overwrites_a_hand_written_dockerfile(mini, contracts, tmp_path):  # noqa: F811
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    with pytest.raises(RenderError, match="refusing to overwrite Dockerfile: it already exists in the target repo"):
        render_all(mini, contracts, MINI_GOLD, tmp_path)
    assert read(tmp_path, "Dockerfile") == "FROM scratch\n"
    assert not (tmp_path / "Makefile").exists()


# --- MLflow client pinned to the server image (tests/faults/mlflow_client_server_mismatch) ----

@pytest.mark.parametrize("image,spec", [
    ("ghcr.io/mlflow/mlflow:v2.17.2", "mlflow==2.17.2"),
    ("ghcr.io/mlflow/mlflow:3.1.4", "mlflow==3.1.4"),
    ("localhost:5000/mlflow:v2.16.2", "mlflow==2.16.2"),   # registry port is not the tag
])
def test_mlflow_client_spec_from_image_tag(image, spec):
    from builder_agent.render.configs import mlflow_client_spec
    assert mlflow_client_spec(image) == spec


@pytest.mark.parametrize("image", ["ghcr.io/mlflow/mlflow:latest", "ghcr.io/mlflow/mlflow"])
def test_mlflow_client_spec_needs_a_version_tag(image):
    from builder_agent.render.configs import mlflow_client_spec
    with pytest.raises(RenderError, match="use a vX.Y.Z tag"):
        mlflow_client_spec(image)


def test_mlflow_client_override_reproduces_fault_001(mini, contracts, tmp_path):  # noqa: F811
    plan = plan_build(mini, contracts)
    writer = RepoWriter(tmp_path)
    render_adapters(mini, contracts, MINI_GOLD, tmp_path, writer=writer)
    render_configs(mini, contracts, plan, MINI_GOLD, tmp_path, mlflow_client="mlflow", writer=writer)
    install = next(line for line in read(tmp_path, "Dockerfile").splitlines() if line.startswith(" && pip install"))
    assert "mlflow" in install.split() and "mlflow==2.17.2" not in install


def _apt_line(root):
    return next(line for line in (root / "Dockerfile").read_text(encoding="utf-8").splitlines()
                if "apt-get install" in line)


def test_known_library_adds_its_system_packages(mini, contracts, tmp_path):  # noqa: F811
    from builder_agent.render import validate_slots
    from builder_agent.render.configs import Overrides
    from builder_agent.scan.models import Requirement
    deps = [d.model_copy(update={"packages": d.packages + [Requirement(name="lightgbm")]})
            if d.kind == "requirements" else d for d in mini.dependency_files]
    ctx = mini.model_copy(update={"dependency_files": deps})
    plan = plan_build(ctx, contracts)
    render_configs(ctx, contracts, plan, MINI_GOLD, tmp_path)
    assert _apt_line(tmp_path).endswith("--no-install-recommends make libgomp1 \\")
    # a fix that adds the same package doesn't duplicate it
    tctx = config_context(ctx, contracts, plan, validate_slots(ctx, MINI_GOLD).slots,
                          overrides=Overrides(apt=["libgomp1", "libffi8"]))
    assert tctx["apt_packages"] == ["make", "libgomp1", "libffi8"]


def test_no_known_library_means_only_make(mini, contracts, tmp_path):  # noqa: F811
    render_configs(mini, contracts, plan_build(mini, contracts), MINI_GOLD, tmp_path)
    assert _apt_line(tmp_path).endswith("--no-install-recommends make \\")
