"""Fault generator: injections change only the Builder's files; a case is saved only if the fault breaks
the sandbox and the expected fix repairs it. Fake Docker runner (no Docker)."""

import json
from pathlib import Path

import pytest
import yaml

from builder_agent.decide import plan_build
from builder_agent.faults import CATALOG, FAMILIES, Variant, error_signature, generate_case
from builder_agent.fix import _installed
from builder_agent.fix.models import Diagnosis
from builder_agent.render import validate_slots
from builder_agent.render.configs import Overrides, config_context, render_configs

from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)
from test_repo_rule import snapshot

PANDAS = Variant(name="pandas_uninstalled", family="missing_dependency", title="pandas missing",
                 injection=Overrides(post_install_commands=["pip uninstall -y pandas"]),
                 expected_fix={"action": "add_dependency", "name": "pandas"}, cause="pandas uninstalled")


class InjectionRunner:
    """Fake docker: `make train` fails while the Dockerfile has `fault` and its last pip install lacks `cure`."""

    def __init__(self, fault: str, cure: str):
        self.fault, self.cure, self.dockerfiles = fault, cure, []

    def __call__(self, cmd, cwd, timeout):
        if "train" in cmd and "logs" not in cmd:
            docker = (Path(cwd) / "Dockerfile").read_text(encoding="utf-8")
            self.dockerfiles.append(docker)
            extras = [line for line in docker.splitlines() if "pip install" in line][-1]
            if self.fault in docker and self.cure not in extras.split():
                return 2, ("python pipeline/train.py\nTraceback (most recent call last):\n"
                           "  File \"/app/pipeline/train.py\", line 3, in <module>\n"
                           "ModuleNotFoundError: No module named 'pandas'\nmake: *** [Makefile:27: train] Error 1")
        return 0, "prediction: 98.5\nsmoke test passed" if "predict" in cmd else "ok"


def gen(mini, v=PANDAS, runner=None, **kw):  # noqa: F811
    runner = runner or InjectionRunner("pip uninstall -y pandas", "pandas")
    return generate_case(v, mini.root, MINI_GOLD, runner=runner, sandbox_log=None, **kw)


# --- the catalog -----------------------------------------------------------------------------------

def test_catalog_has_one_variant_per_family_with_a_menu_fix():
    assert {v.family for v in CATALOG.values()} == set(FAMILIES)
    for v in CATALOG.values():
        Diagnosis.model_validate({"fix": v.expected_fix, "reason": "x"})   # a menu action, strict
        assert v.injection != Overrides() or v.mlflow_client, v.name


# --- injection hooks in the rendered files ---------------------------------------------------------

def render(mini, contracts, o: Overrides, out: Path):  # noqa: F811
    plan = plan_build(mini, contracts)
    render_configs(mini, contracts, plan, MINI_GOLD, out, overrides=o)
    files = {n: (out / n).read_text(encoding="utf-8") for n in ["Dockerfile", "compose.base.yml"]}
    return config_context(mini, contracts, plan, validate_slots(mini, MINI_GOLD).slots, None, o), files


def test_hooks_land_before_apt_and_between_repo_install_and_extras(mini, contracts, tmp_path):  # noqa: F811
    o = Overrides(pre_apt_commands=["dpkg --remove --force-depends libffi8"],
                  post_install_commands=["pip uninstall -y pandas"])
    _, files = render(mini, contracts, o, tmp_path)
    docker = files["Dockerfile"]
    assert docker.index("RUN dpkg --remove") < docker.index("RUN apt-get update")
    assert docker.index("-r requirements.txt") < docker.index("pip uninstall -y pandas") < docker.index("mlflow==2.17.2")


def test_env_override_reaches_compose_and_replaces_the_default_uri(mini, contracts, tmp_path):  # noqa: F811
    _, files = render(mini, contracts, Overrides(env={"MLFLOW_TRACKING_URI": "http://mlflow-server:5000"}), tmp_path)
    env = yaml.safe_load(files["compose.base.yml"])["services"]
    model_env = next(s["environment"] for s in env.values() if "build" in s)
    assert model_env["MLFLOW_TRACKING_URI"] == "http://mlflow-server:5000" and "MODEL_URI" in model_env
    assert files["Dockerfile"].count("MLFLOW_TRACKING_URI") == 1          # the override, not the default too


def test_default_render_keeps_its_compose_env(mini, contracts, tmp_path):  # noqa: F811
    _, files = render(mini, contracts, Overrides(), tmp_path)
    model_env = next(s["environment"] for s in yaml.safe_load(files["compose.base.yml"])["services"].values()
                     if "build" in s)
    assert model_env == {"MLFLOW_TRACKING_URI": "http://mlflow:5000", "MODEL_URI": "${MODEL_URI:-}"}


def test_uninstalled_package_is_not_installed_for_the_fix_menu(mini, contracts, tmp_path):  # noqa: F811
    tctx, _ = render(mini, contracts, PANDAS.injection, tmp_path)
    assert "pandas" not in _installed(mini, tctx)
    assert "numpy" in _installed(mini, tctx)


# --- error signatures ------------------------------------------------------------------------------

@pytest.mark.parametrize("tail,sig", [
    (["Traceback (most recent call last):", "ModuleNotFoundError: No module named 'pandas'",
      "make: *** [Makefile:27: train] Error 1"], "ModuleNotFoundError: No module named 'pandas'"),
    (["#9 41.2 ImportError: libffi.so.8: cannot open shared object file", "#9 ERROR: process did not complete",
      "------", "failed to solve: process \"/bin/sh -c pip install\" did not complete successfully: exit code: 1"],
     "ImportError: libffi.so.8: cannot open shared object file"),
    (["#10 38.17 [pipenv.exceptions.InstallError]:       AttributeError: module 'pkgutil' has no attribute 'ImpImporter'",
      "#10 38.17 [pipenv.exceptions.InstallError]: error: subprocess-exited-with-error",
      "38.17 ERROR: Couldn't install package: {}"], "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"),
    (["no error words here", "last line"], "last line"),
])
def test_error_signature(tail, sig):
    assert error_signature(tail) == sig
    assert any(sig in line for line in tail)


# --- generating a case -----------------------------------------------------------------------------

def test_case_saved_when_fault_breaks_and_expected_fix_passes(mini, tmp_path):  # noqa: F811
    before = snapshot(Path(mini.root))
    r = gen(mini, out=tmp_path)
    assert snapshot(Path(mini.root)) == before                                   # the repo is untouched
    assert r.saved and not r.fault_attempt.ok and r.fix_attempt.ok
    case = json.loads(Path(r.saved).read_text(encoding="utf-8"))
    assert (case["id"], case["name"], case["family"], case["stage"]) == \
        ("gen-001", "pandas_uninstalled", "missing_dependency", "train")
    assert case["error_signature"] == "ModuleNotFoundError: No module named 'pandas'"
    assert case["expected_fix"] == {"action": "add_dependency", "name": "pandas"}
    assert case["injection"] == {"post_install_commands": ["pip uninstall -y pandas"]}
    assert case["split"] is None and case["fix"]["verified_by"]["ok"]


def test_ids_are_sequential_and_a_variant_is_generated_once(mini, tmp_path):  # noqa: F811
    gen(mini, out=tmp_path)
    other = Variant(**(PANDAS.model_dump() | {"name": "other"}))
    r = gen(mini, other, out=tmp_path)
    assert r.saved, r.reason
    assert json.loads(Path(r.saved).read_text(encoding="utf-8"))["id"] == "gen-002"
    with pytest.raises(FileExistsError):
        gen(mini, out=tmp_path)


def test_not_saved_when_the_fault_does_not_break(mini, tmp_path):  # noqa: F811
    r = gen(mini, runner=InjectionRunner("never in a Dockerfile", "x"), out=tmp_path)
    assert r.saved is None and "did not break" in r.reason and r.fix_attempt is None
    assert not list(tmp_path.iterdir())


def test_not_saved_when_the_expected_fix_does_not_pass(mini, tmp_path):  # noqa: F811
    r = gen(mini, runner=InjectionRunner("pip uninstall -y pandas", "never in a Dockerfile"), out=tmp_path)
    assert r.saved is None and "did not pass: failed at train" in r.reason
    assert not list(tmp_path.iterdir())


def test_expected_fix_the_loop_would_reject_is_refused(mini, tmp_path):  # noqa: F811
    bad = Variant(**(PANDAS.model_dump() | {"name": "bad", "expected_fix": {"action": "add_dependency",
                                                                              "name": "numpy"}}))
    with pytest.raises(ValueError, match="already installed"):
        gen(mini, bad, out=tmp_path)


def test_mlflow_client_variant_renders_unpinned_and_is_recorded(mini, tmp_path):  # noqa: F811
    v = Variant(name="unpinned", family="version_mismatch", title="t", injection=Overrides(), mlflow_client="mlflow",
                expected_fix={"action": "pin_package", "name": "mlflow", "version": "2.17.2"}, cause="c")
    runner = InjectionRunner("FROM", "mlflow==2.17.2")
    r = gen(mini, v, runner=runner, out=tmp_path)
    assert r.saved, r.reason
    assert "mlflow==" not in runner.dockerfiles[0] and "mlflow==2.17.2" in runner.dockerfiles[1]
    assert r.case["injection"] == {"mlflow_client": "mlflow"}
