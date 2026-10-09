"""Build-failure diagnosis loop, with a fake model and a fake Docker runner (no Ollama, no Docker)."""

import json
from pathlib import Path

import pytest

from builder_agent.decide import plan_build
from builder_agent.fix import FixLoopResult, run_fix_loop, score_fix
from builder_agent.fix.models import apply_fix, validate_fix
from builder_agent.llm import load_prompt
from builder_agent.render import render_adapters
from builder_agent.render.configs import Overrides, render_configs
from builder_agent.repo_writer import RepoWriter

from test_llm import FakeModel
from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)
from test_repo_rule import snapshot

FAULT_001 = json.loads((Path(__file__).parent / "faults/mlflow_client_server_mismatch/case.json")
                       .read_text(encoding="utf-8"))
PIN = {"fix": {"action": "pin_package", "name": "mlflow", "version": "2.17.2"},
       "reason": "the 3.x client calls /logged-models, which a 2.x server doesn't have"}


class Fault001Runner:
    """Fake docker: `make train` fails with fault-001's real output unless the Builder's Dockerfile pins mlflow."""

    def __init__(self):
        self.dockerfiles: list[str] = []

    def __call__(self, cmd, cwd, timeout):
        if "train" in cmd and "logs" not in cmd:
            docker = (Path(cwd) / "Dockerfile").read_text(encoding="utf-8")
            self.dockerfiles.append(docker)
            if "mlflow==2.17.2" not in docker:
                return 2, "\n".join(FAULT_001["error_tail"])
        return 0, "prediction: 98.5\nsmoke test passed" if "predict" in cmd else "ok"


def loop(mini, model, runner=None, prompt="diagnose_v1", **kw):  # noqa: F811
    return run_fix_loop(mini.root, MINI_GOLD, prompt, model, mlflow_client="mlflow",
                        runner=runner or Fault001Runner(), sandbox_log=None, **kw)


# --- the menu and its validation ---------------------------------------------------------------

@pytest.mark.parametrize("answer,reason", [
    ({"fix": {"action": "pin_package", "name": "mlflow", "version": "latest"}, "reason": "x"}, "must look like 2.17"),
    ({"fix": {"action": "pin_package", "name": "mlflow", "version": 2.17}, "reason": "x"}, "valid string"),
    ({"fix": {"action": "set_python_version", "version": "2.7"}, "reason": "x"}, "must be 3.N"),
    ({"fix": {"action": "set_env_var", "name": "PATH", "value": "/x"}, "reason": "x"}, "PATH can't be changed"),
    ({"fix": {"action": "set_env_var", "name": "lower", "value": "x"}, "reason": "x"}, "UPPER_CASE"),
    ({"fix": {"action": "add_dependency", "name": "pandas"}, "reason": "x"}, "already installed"),
    ({"fix": {"action": "add_system_package", "name": "Bad Name"}, "reason": "x"}, "not a valid Debian"),
    ({"fix": {"action": "rewrite_code", "path": "x"}, "reason": "x"}, "does not match any of the expected tags"),
    ({"fix": {"action": "give_up", "reason": "x"}, "reason": "x", "extra": 1}, "Extra inputs"),
])
def test_rejections(answer, reason):
    v = validate_fix(answer, Overrides(), installed={"pandas"})
    assert not v.ok and any(reason in r for r in v.reasons), v.reasons


def test_a_fix_already_tried_is_rejected():
    applied = apply_fix(Overrides(), validate_fix(PIN, Overrides(), set()).diagnosis.fix)
    assert applied.pins == {"mlflow": "2.17.2"}
    v = validate_fix(PIN, applied, set())
    assert v.reasons == ["pin_package mlflow==2.17.2 was already applied and the stage still failed"]


def test_overrides_change_only_the_builders_dockerfile(mini, contracts, tmp_path):  # noqa: F811
    o = Overrides(pins={"mlflow": "2.17.2", "numpy": "1.26.4"}, apt=["libgomp1"], python_version="3.12",
                  env={"GIT_PYTHON_REFRESH": "quiet"}, dependencies=["scipy"])
    w = RepoWriter(tmp_path)
    render_adapters(mini, contracts, MINI_GOLD, tmp_path, writer=w)
    render_configs(mini, contracts, plan_build(mini, contracts), MINI_GOLD, tmp_path, mlflow_client="mlflow",
                   writer=w, overrides=o)
    docker = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in docker
    assert "--no-install-recommends make libgomp1" in docker
    assert 'ENV GIT_PYTHON_REFRESH="quiet"' in docker
    install = next(line for line in docker.splitlines() if line.startswith(" && pip install"))
    assert install.split()[3:] == ["gunicorn", "scipy", "mlflow==2.17.2", "numpy==1.26.4"]   # pin beats "mlflow"
    assert sorted(w.created) == sorted(w.created)        # only new files: the repo's own files aren't written


# --- the loop -----------------------------------------------------------------------------------

def test_fault_001_fixed_by_the_expected_pin(mini, tmp_path):  # noqa: F811
    before = snapshot(Path(mini.root))
    runner = Fault001Runner()
    log = tmp_path / "fix.jsonl"
    r = loop(mini, FakeModel(PIN), runner, fix_log=log)
    assert r.initial == "failed at train" and r.final_ok and r.stopped_because == "passed"
    [a] = r.attempts
    assert (a.stage, a.action_text, a.outcome) == ("train", "pin_package mlflow==2.17.2", "passed")
    assert "logged-models failed with error code 404" in "\n".join(a.error_tail)
    assert "mlflow==2.17.2" not in runner.dockerfiles[0] and "mlflow==2.17.2" in runner.dockerfiles[1]
    assert score_fix(r, FAULT_001["expected_fix"]) == {"first_fix_passed": True, "fixes_to_pass": 1, "exact": True,
                                                        "loose": True, "first_action": PIN["fix"]}
    assert snapshot(Path(mini.root)) == before                          # the repo itself is never touched

    [entry] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entry["prompt_version"] == "diagnose_v1" and entry["model_digest"] == "sha256:fake"
    assert entry["action"] == PIN["fix"] and entry["outcome"] == "passed"
    user = entry["calls"][0]["messages"][1]["content"]
    assert "FAILED STAGE: train" in user and "/api/2.0/mlflow/logged-models" in user


def test_rejected_answer_is_retried_with_reasons(mini):  # noqa: F811
    model = FakeModel({"fix": {"action": "pin_package", "name": "mlflow", "version": "latest"}, "reason": "x"}, PIN)
    r = loop(mini, model)
    assert r.final_ok and len(r.attempts[0].calls) == 2
    retry = model.sent[1][-1]["content"]
    assert retry.startswith("Your previous answer was rejected") and "must look like 2.17" in retry


def test_loose_score_for_a_wrong_version(mini):  # noqa: F811
    wrong = {"fix": {"action": "pin_package", "name": "mlflow", "version": "2.9.2"}, "reason": "x"}
    r = loop(mini, FakeModel(wrong, wrong, PIN, PIN))            # 2.9.2 doesn't fix the fake; then the right pin
    assert r.final_ok and [a.action_text for a in r.attempts] == ["pin_package mlflow==2.9.2",
                                                                 "pin_package mlflow==2.17.2"]
    # second diagnosis: repeating 2.9.2 is rejected, and the history shows it didn't help
    assert r.attempts[1].calls[0].reasons == ["pin_package mlflow==2.9.2 was already applied and the stage "
                                              "still failed"]
    assert r.attempts[1].calls[1].valid
    assert "- pin_package mlflow==2.9.2: then failed at train" in r.attempts[1].calls[0].messages[1]["content"]
    score = score_fix(r, FAULT_001["expected_fix"])
    assert (score["first_fix_passed"], score["fixes_to_pass"], score["exact"], score["loose"]) == (False, 2, False, True)


def test_give_up_stops(mini):  # noqa: F811
    r = loop(mini, FakeModel({"fix": {"action": "give_up", "reason": "not a build problem"}, "reason": "x"}))
    assert not r.final_ok and r.stopped_because == "give_up" and r.attempts[0].outcome == "not applied: the LLM gave up"


def test_at_most_three_fixes(mini):  # noqa: F811
    envs = [{"fix": {"action": "set_env_var", "name": f"V{i}", "value": "1"}, "reason": "x"} for i in range(3)]
    r = loop(mini, FakeModel(*envs))
    assert not r.final_ok and r.stopped_because == "max fixes reached" and len(r.attempts) == 3
    assert r.applied.env == {"V0": "1", "V1": "1", "V2": "1"}


def test_invalid_diagnosis_after_three_calls_stops(mini):  # noqa: F811
    r = loop(mini, FakeModel("nope", "still nope", ["a list"]))
    assert r.stopped_because == "invalid diagnosis" and len(r.attempts[0].calls) == 3 and not r.final_ok


def test_llm_error_is_logged_and_stops(mini, tmp_path):  # noqa: F811
    class Broken(FakeModel):
        def chat(self, messages, schema=None):
            raise ConnectionError("Ollama is down")
    log = tmp_path / "fix.jsonl"
    r = loop(mini, Broken(), fix_log=log)
    assert r.stopped_because == "LLM call failed"
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["calls"][0]["error"] == "ConnectionError: Ollama is down"


def test_the_draft_prompt_renders(mini):  # noqa: F811
    p = load_prompt("diagnose_v1")
    assert set(p.sections) == {"system", "user", "retry"}
    r: FixLoopResult = loop(mini, FakeModel(PIN))
    msgs = r.attempts[0].calls[0].messages
    assert "pin_package {name, version}" in msgs[0]["content"]        # menu in the system section
    assert "FIXES ALREADY TRIED:\nnone" in msgs[1]["content"]
    assert "MLflow server image" not in msgs[1]["content"]            # draft uses the error alone


# --- Chain-of-Thought prompts: "analysis" first, only when the prompt asks for it ---------------

COT = {"analysis": "Step 1: 'API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404'. "
                   "Step 2: the server lacks the endpoint the client calls. Step 3: the mlflow package version. "
                   "Step 4: pin_package.",
       "fix": PIN["fix"], "reason": PIN["reason"]}


def test_v2_asks_for_analysis_first_and_logs_it(mini, tmp_path):  # noqa: F811
    model = FakeModel(COT)
    log = tmp_path / "fix.jsonl"
    r = loop(mini, model, prompt="diagnose_v2", fix_log=log)
    assert r.final_ok and r.prompt_version == "diagnose_v2"
    assert list(model.schema["properties"]) == ["analysis", "fix", "reason"]
    assert model.schema["required"][0] == "analysis"
    assert r.attempts[0].analysis == COT["analysis"]
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["analysis"] == COT["analysis"] and entry["action"] == PIN["fix"]
    system, user = (m["content"] for m in r.attempts[0].calls[0].messages)
    assert "think step by step in the \"analysis\" field" in system and "{{" not in system + user
    assert '{"analysis": "Step 1: ... Step 2:' in user


def test_v2_rejects_an_answer_without_analysis(mini):  # noqa: F811
    model = FakeModel(PIN, {**COT, "analysis": "  "}, COT)
    r = loop(mini, model, prompt="diagnose_v2")
    calls = r.attempts[0].calls
    assert any("analysis: Field required" in reason for reason in calls[0].reasons)
    assert calls[1].reasons == ["analysis is empty: write the steps the prompt asks for before the fix"]
    assert calls[2].valid and r.final_ok
    assert "analysis first" in model.sent[1][-1]["content"]          # v2's retry wording


def test_v1_schema_is_unchanged(mini):  # noqa: F811
    model = FakeModel(PIN)
    r = loop(mini, model)
    assert list(model.schema["properties"]) == ["fix", "reason"] and r.attempts[0].analysis is None
    assert not validate_fix(COT, Overrides(), set()).ok             # v1 doesn't accept extra fields


# --- check 2: a Python package is not a system package ----------------------------------------------

@pytest.mark.parametrize("name,known", [("mlflow", {"mlflow"}), ("scikit-learn", {"scikit-learn"}),
                                        ("Scikit_Learn", {"scikit-learn"})])
def test_add_system_package_rejects_python_packages(name, known):
    answer = {"fix": {"action": "add_system_package", "name": name.lower()}, "reason": "x"}
    v = validate_fix(answer, Overrides(), installed=set(), python_packages=known)
    assert not v.ok and any("is a Python package (installed with pip), not a Debian package" in r for r in v.reasons)


def test_add_system_package_still_allows_debian_packages():
    answer = {"fix": {"action": "add_system_package", "name": "libgomp1"}, "reason": "x"}
    assert validate_fix(answer, Overrides(), installed={"mlflow"}, python_packages={"xgboost"}).ok


def test_the_loop_rejects_the_v2_mistake(mini):  # noqa: F811
    apt_mlflow = {"fix": {"action": "add_system_package", "name": "mlflow"}, "reason": "x"}
    r = loop(mini, FakeModel(apt_mlflow, PIN))
    calls = r.attempts[0].calls
    assert "not a Debian package" in calls[0].reasons[0] and calls[1].valid and r.final_ok


def test_loop_starts_from_an_injected_fault_and_exports_its_logs(mini, tmp_path):  # noqa: F811
    from builder_agent.fix import export_run

    runner, fix_log, sbx_log = Fault001Runner(), tmp_path / "fix.jsonl", tmp_path / "sbx.jsonl"
    r = run_fix_loop(mini.root, MINI_GOLD, "diagnose_v1", FakeModel(PIN), runner=runner, sandbox_log=sbx_log,
                     fix_log=fix_log, injection=Overrides(pins={"mlflow": "3.1.4"}))
    assert "mlflow==3.1.4" in runner.dockerfiles[0] and "mlflow==2.17.2" in runner.dockerfiles[1]
    assert r.final_ok and r.injection.pins == {"mlflow": "3.1.4"} and r.applied.pins == {"mlflow": "2.17.2"}
    out = export_run(r, tmp_path / "run", r.text(), fix_log, sbx_log)
    assert len((out / "fix_attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((out / "sandbox_attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_score_counts_a_passing_alternative_fix_and_not_fixed_runs():
    from builder_agent.fix import score_attempts

    expected = {"action": "pin_package", "name": "flask", "version": "3.0.1"}
    werkzeug = {"action": "pin_package", "name": "werkzeug", "version": "2.3.1"}
    passed = score_attempts([{"n": 1, "action": werkzeug, "outcome": "passed"}], expected)
    assert (passed["first_fix_passed"], passed["fixes_to_pass"], passed["exact"], passed["loose"]) == (True, 1, False, False)
    failed = score_attempts([{"n": 1, "action": None, "outcome": "not applied: no valid fix after 3 call(s)"},
                             {"n": 2, "action": expected, "outcome": "failed at build"}], expected)
    assert (failed["first_fix_passed"], failed["fixes_to_pass"], failed["exact"]) == (False, None, True)


# --- apt pattern names and the regression guard (gen-011: "add_system_package clang++" broke apt) ---------

@pytest.mark.parametrize("name", ["clang++", "lib*", "libfoo?", "lib[ab]"])
def test_apt_pattern_names_are_rejected(name):
    v = validate_fix({"fix": {"action": "add_system_package", "name": name}, "reason": "x"}, Overrides(), set())
    assert not v.ok and "apt-get reads as a pattern" in v.reasons[0] and "build-essential" in v.reasons[0]


def test_exact_apt_name_is_accepted():
    assert validate_fix({"fix": {"action": "add_system_package", "name": "libffi8"}, "reason": "x"},
                        Overrides(), set()).ok


class GuardRunner(Fault001Runner):
    """The build fails if the Dockerfile installs `breakit`; otherwise like fault-001 (train fails until pinned)."""

    def __call__(self, cmd, cwd, timeout):
        if cmd[-1] == "build" or "build" in cmd[-2:]:
            if "breakit" in (Path(cwd) / "Dockerfile").read_text(encoding="utf-8"):
                return 1, "E: Unable to locate package breakit"
        return super().__call__(cmd, cwd, timeout)


def test_regression_is_reverted_and_counted_as_a_failed_attempt(mini, tmp_path):  # noqa: F811
    breakit = {"fix": {"action": "add_system_package", "name": "breakit"}, "reason": "x"}
    runner = GuardRunner()
    r = loop(mini, FakeModel(breakit, breakit, PIN), runner, fix_log=tmp_path / "fix.jsonl")
    first, second = r.attempts
    assert first.action_text == "add_system_package breakit"
    assert first.outcome == "reverted: regression (failed at build, before train)" and first.sandbox_attempt_id
    assert second.stage == "train"                                   # diagnosed on the error from before the fix
    assert second.calls[0].reasons == ["add_system_package breakit was tried and reverted: it made an earlier "
                                       "stage fail"]
    assert "- add_system_package breakit: then reverted: regression" in second.calls[0].messages[1]["content"]
    assert r.final_ok and "breakit" not in r.applied.apt
    assert "breakit" not in runner.dockerfiles[-1]
    assert score_fix(r, FAULT_001["expected_fix"])["fixes_to_pass"] == 2


def test_failing_at_the_same_or_a_later_stage_is_kept(mini):  # noqa: F811
    from builder_agent.fix import regressed
    from builder_agent.sandbox import SandboxResult

    def res(stage):
        return SandboxResult(attempt_id="a", repo="r", started_at="t", ok=stage is None, failed_stage=stage,
                             total_s=0, workdir="w", compose_project="p", stages=[])
    assert regressed(res("train"), res("build")) and regressed(res("evaluate"), res("render"))
    assert not regressed(res("train"), res("train")) and not regressed(res("train"), res("serve"))
    assert not regressed(res("train"), res(None))
