"""Leave-one-out evaluation (docs/EVAL_PROTOCOL.md): the memory, the statistics, the runner's rules, the analysis
and the blind label file. No Docker, no Ollama: fake fix-loop results."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "eval_loo"))

import analyze  # noqa: E402
import make_labels  # noqa: E402
import run  # noqa: E402
import stats  # noqa: E402

from builder_agent.memory import MemoryCaseError, all_cases, doc_from_case, loo_docs  # noqa: E402

from test_render import mini  # noqa: E402,F401 (fixture)


# --- the leave-one-out memory -------------------------------------------------------------------------

def test_fourteen_cases_and_their_memories():
    ids = [c["id"] for c in all_cases()]
    assert len(ids) == 14 and ids[0] == "fault-001" and "gen-011" in ids
    for case_id in ids:
        docs = {d.id for d in loo_docs(case_id)}
        assert case_id not in docs
        assert len(docs) == (12 if case_id in ("gen-008", "fault-001") else 13)
    assert "fault-001" not in {d.id for d in loo_docs("gen-008")}            # the twin goes too
    assert {"gen-001", "gen-010", "gen-013"} <= {d.id for d in loo_docs("gen-011")}   # test cases, for others
    with pytest.raises(MemoryCaseError, match="never indexed"):
        doc_from_case(next(c for c in all_cases() if c["id"] == "gen-011"))   # still refused outside loo
    with pytest.raises(MemoryCaseError):
        loo_docs("gen-099")


# --- statistics ---------------------------------------------------------------------------------------

def test_statistics():
    lo, hi = stats.wilson(5, 10)
    assert round(lo, 3) == 0.237 and round(hi, 3) == 0.763
    assert stats.sign_test([1, 1, 1, 1]) == (4, 0, 0.125)                    # 4 cases can't reach 0.05
    assert stats.sign_test([1, -1, 0]) == (1, 1, 1.0)
    assert stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    pc = {"x": [1, 1, 1, 1, 1], "y": [0, 0, 0, 0, 0], "z": [1, 0, 1, 0, 1]}
    lo, hi = stats.cluster_bootstrap(pc, reps=2000)
    assert 0 <= lo < stats.rate(pc)[2] < hi <= 1
    assert stats.cluster_bootstrap(pc, reps=500) == stats.cluster_bootstrap(pc, reps=500)   # fixed seed
    d, lo, hi = stats.paired_bootstrap({"x": [1, 1], "y": [1, 0]}, {"x": [0, 0], "y": [0, 0]}, reps=500)
    assert d == 0.75 and lo > 0
    assert stats.mcnemar([(1, 0)] * 6 + [(0, 1)] + [(1, 1)] * 3) == (6, 1, 0.125)


# --- the runner -------------------------------------------------------------------------------------

CASE = {"id": "gen-002", "variant": "xgboost_uninstalled", "family": "missing_dependency", "stage": "train",
        "error_signature": "boom",
        "injection": {"post_install_commands": ["pip uninstall -y xgboost"]}}


def fake_result(initial="failed at train", tail=("ModuleNotFoundError: No module named 'xgboost'",), error=None,
                outcome="passed"):
    from builder_agent.fix import Call, FixAttempt, FixLoopResult
    from builder_agent.render.configs import Overrides
    att = FixAttempt(n=1, stage="train", error_tail=list(tail), outcome=outcome, sandbox_attempt_id="s1",
                     calls=[Call(messages=[], response="{}", latency_s=1, valid=True, error=error)],
                     action={"action": "add_dependency", "name": "xgboost"}, action_text="add_dependency xgboost",
                     reason="xgboost is missing: No module named 'xgboost'", retrieved=[{"id": "gen-006"}])
    return FixLoopResult(repo="taxi", prompt_version="diagnose_v3", prompt_sha256="p", model="m", model_digest="d",
                         initial=initial, initial_attempt_id="s0", attempts=[att], final_ok=outcome == "passed",
                         stopped_because="passed", applied=Overrides(), python_rule=False)


def test_runner_rules():
    assert run.fault({"id": "fault-001"}) == (None, "mlflow")
    o, client = run.fault({"injection": {"python_version": "3.13"}})
    assert o.python_version == "3.13" and client is None
    p1 = run.plan("main", ["v1", "v3-advanced"], ["gen-001", "gen-002"], [101, 102])
    assert len(p1) == 8 and p1 == run.plan("main", ["v1", "v3-advanced"], ["gen-001", "gen-002"], [101, 102])
    assert run.invalid(fake_result(), CASE) is None
    assert "LLM call failed" in run.invalid(fake_result(error="URLError: refused"), CASE)
    assert "Docker error" in run.invalid(fake_result(tail=("failed to connect to the docker API at npipe",)), CASE)
    assert "didn't reproduce" in run.invalid(fake_result(initial="failed at build"), CASE)


def test_run_one_writes_last_and_deletes_an_invalid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "loo_docs", lambda case_id: [])
    d = run.run_dir(tmp_path, "main", "v1", "gen-002", 101)
    (d / "half").mkdir(parents=True)                                       # an earlier, incomplete run
    rec = run.run_one(d, "main", "v1", CASE, 101, {}, "m", run_fix_loop=lambda *a, **k: fake_result())
    assert run.is_done(d) and not (d / "half").exists() and rec["first_fix_passed"] and rec["python_rule"] is False
    assert json.loads((d / "result.json").read_text(encoding="utf-8"))["temperature"] == 0.7
    bad = run.run_dir(tmp_path, "main", "v1", "gen-002", 102)
    with pytest.raises(run.Stop, match="didn't reproduce"):
        run.run_one(bad, "main", "v1", CASE, 102, {}, "m",
                    run_fix_loop=lambda *a, **k: fake_result(initial="failed at build"))
    assert not bad.exists()


# --- analysis and labels ---------------------------------------------------------------------------

def result(config, case, seed, first, ok, reason, retrieved=()):
    return {"config": config, "case": case, "variant": "v", "family": "f", "seed": seed, "first_fix_passed": first,
            "final_ok": ok, "memory_ids": ["x"] if retrieved else [],
            "attempts": [{"n": 1, "source": "llm", "action": {"action": "pin_package"}, "sandbox_attempt_id": "s",
                          "action_text": "pin_package mlflow==2.17.2", "outcome": "passed" if first else "failed",
                          "reason": reason, "error_tail": [], "retrieved": [{"id": r} for r in retrieved]}]}


def test_analysis_and_blind_labels(tmp_path):
    good = "The client gets a 404 from /api/2.0/mlflow/logged-models: pin the client to the server."
    runs = [result(cfg, case, s, first, first, good, ["gen-007"] if cfg != "v1" else ())
            for cfg, first in (("v1", False), ("v3-advanced", True))
            for case in ("gen-001", "gen-007", "gen-008") for s in (101, 102)]
    res = analyze.analyze(runs)
    assert res["configs"]["v3-advanced"]["first_fix_passed"]["rate"] == 1.0
    assert res["configs"]["v1"]["first_fix_passed"]["rate"] == 0.0
    c = res["comparisons"]["first_fix_passed"]["v3-advanced - v1"]
    assert c["sign_test"]["better"] == 3 and c["verdict"] == "no difference shown"     # 3 cases: p = 0.25
    assert "First fix passes (main)" in analyze.markdown(res, "main")
    picked = make_labels.sample(runs, 6)
    assert len(picked) == 6 and {p["config"] for p in picked} == {"v1", "v3-advanced"}
    make_labels.write(picked, tmp_path / "labels")
    rows = list(csv.DictReader((tmp_path / "labels" / "to_label.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["item", "fault", "error_line", "reason", "label"] and rows[0]["label"] == ""
    sheet = (tmp_path / "labels" / "to_label.csv").read_text(encoding="utf-8")
    assert "v3-advanced" not in sheet and "refers to" not in sheet                    # blind: no config, no verdict
    assert analyze.label_agreement(tmp_path / "labels") is None                       # nothing labelled yet


# --- speed: runs grouped by case, the first sandbox run shared -------------------------------------

def test_plan_groups_runs_by_case():
    todo = run.plan("main", list(run.CONFIGS), ["gen-001", "gen-002", "gen-003"], [101, 102])
    cases_in_order = [c for _, c, _ in todo]
    blocks = [c for i, c in enumerate(cases_in_order) if i == 0 or c != cases_in_order[i - 1]]
    assert len(todo) == 18 and len(blocks) == 3                  # each case's 6 runs are consecutive
    assert todo == run.plan("main", list(run.CONFIGS), ["gen-001", "gen-002", "gen-003"], [101, 102])


def fake_sandbox_result(stage="train"):
    from builder_agent.sandbox import SandboxResult, StageResult
    return SandboxResult(repo="taxi", attempt_id="first-1", started_at="t", ok=False, failed_stage=stage, total_s=1,
                         workdir="w", compose_project="p", stages=[StageResult(name=stage, status="failed",
                                                                               output_tail=["boom"])], teardown="ok")


def test_initial_run_is_stored_and_reused_only_while_the_fingerprint_holds(tmp_path, monkeypatch):
    fp = {"builder_agent": "a", "contracts": "b", "gold": "c", "taxi_head": "d", "uncommitted": False}
    monkeypatch.setattr(run, "fingerprint", lambda: dict(fp))
    calls = []

    def sandbox(*a, **k):
        calls.append(1)
        return fake_sandbox_result()
    r, reused = run.initial_run(tmp_path, CASE, sandbox=sandbox)
    assert not reused and calls == [1] and (tmp_path / "_initial" / "gen-002.json").exists()
    r2, reused = run.initial_run(tmp_path, CASE, sandbox=sandbox)
    assert reused and calls == [1] and r2.attempt_id == r.attempt_id
    fp["builder_agent"] = "changed"                              # the Builder changed: run it again
    _, reused = run.initial_run(tmp_path, CASE, sandbox=sandbox)
    assert not reused and calls == [1, 1]
    fp["uncommitted"] = True                                     # uncommitted code: never reused, never stored
    (tmp_path / "_initial" / "gen-002.json").unlink()
    _, reused = run.initial_run(tmp_path, CASE, sandbox=sandbox)
    assert not reused and not (tmp_path / "_initial" / "gen-002.json").exists()
    with pytest.raises(run.Stop, match="didn't reproduce"):
        run.initial_run(tmp_path, CASE, sandbox=lambda *a, **k: fake_sandbox_result("build"))


def test_run_fix_loop_uses_a_given_first_run(mini):  # noqa: F811
    from test_fix import PIN, Fault001Runner
    from test_llm import FakeModel
    from test_render import MINI_GOLD
    from builder_agent.fix import run_fix_loop
    runner = Fault001Runner()
    first = run_fix_loop(mini.root, MINI_GOLD, "diagnose_v1", FakeModel(PIN), mlflow_client="mlflow", runner=runner,
                         sandbox_log=None, fix_log=None)
    builds = len(runner.dockerfiles)
    runner2 = Fault001Runner()
    given = fake_sandbox_result()
    given.stages[0].output_tail = first.attempts[0].error_tail
    again = run_fix_loop(mini.root, MINI_GOLD, "diagnose_v1", FakeModel(PIN), mlflow_client="mlflow", runner=runner2,
                         sandbox_log=None, fix_log=None, initial=given)
    assert again.initial_attempt_id == "first-1" and again.final_ok
    assert len(runner2.dockerfiles) == builds - 1                # the first sandbox run was not repeated


def test_action_accuracy():
    cases = {c["id"]: c for c in all_cases()}

    def first(action):
        return {"attempts": [{"n": 1, "source": "llm", "action": action}] if action else
                [{"n": 1, "source": "llm", "action": None}]}
    alts = {"gen-004": [{"action": "set_python_version", "version": "3.10"}]}
    m = lambda case, action: analyze.action_match(first(action), cases[case], alts)  # noqa: E731
    assert m("gen-001", {"action": "pin_package", "name": "mlflow", "version": "2.17.2"}) == "exact"
    assert m("gen-001", {"action": "pin_package", "name": "MLflow", "version": "2.10.0"}) == "same_target"
    assert m("gen-013", {"action": "set_env_var", "name": "MLFLOW_TRACKING_URI", "value": "http://x:1"}) == "same_target"
    assert m("gen-004", {"action": "set_python_version", "version": "3.10"}) == "alternative"
    assert m("gen-004", {"action": "set_python_version", "version": "3.11"}) == "wrong"     # no same target for Python
    assert m("gen-001", {"action": "give_up", "reason": "x"}) == "wrong"
    assert m("gen-001", None) == "none"
    real = analyze.verified_alternatives()                       # checks.json: only fixes that passed
    assert {"action": "set_python_version", "version": "3.10"} in real["gen-004"]
    assert all(f != {"action": "set_env_var", "name": "GIT_PYTHON_REFRESH", "value": "quiet"} for f in real.get("gen-007", []))
    runs = [result(cfg, "gen-001", s, True, True, "x") for cfg in ("v1", "v3-advanced") for s in (101, 102)]
    res = analyze.analyze(runs)
    assert res["configs"]["v1"]["action"]["rate"] == 0.0 and res["configs"]["v1"]["action_kinds"]["wrong"] == 2
    assert "action" not in res["comparisons"]                    # secondary: never compared
    assert "Action accuracy (secondary)" in analyze.markdown(res, "main")


# --- changes during the run (2026-10-14): stopped runs are kept; Docker errors with Docker healthy rerun once ----

DOCKER_TAIL = ("error during connect: Get http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.47/info: EOF",)


def stopped(tmp_path):
    return sorted(p for p in (tmp_path / "_stopped").rglob("seed-*") if p.is_dir())


def test_a_stopped_run_is_moved_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "loo_docs", lambda case_id: [])
    d = run.run_dir(tmp_path, "main", "v1", "gen-002", 101)
    (d / "partial").mkdir(parents=True)                                     # an interrupted earlier run
    with pytest.raises(run.Stop, match="moved to"):
        run.run_one(d, "main", "v1", CASE, 101, {}, "m", initial=fake_sandbox_result(), health=lambda: None,
                    run_fix_loop=lambda *a, **k: fake_result(error="URLError: refused"))
    assert not d.exists()
    kept = stopped(tmp_path)
    assert len(kept) == 2 and all(k.parent == tmp_path / "_stopped" / "main" / "v1" / "gen-002" for k in kept)
    reasons = sorted((k / "reason.txt").read_text(encoding="utf-8") for k in kept)
    assert any("LLM call failed" in r for r in reasons) and any("incomplete" in r for r in reasons)
    assert any((k / "sandbox_attempts.jsonl").exists() for k in kept)        # the evidence is still there


def test_a_docker_error_with_docker_healthy_is_rerun_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "loo_docs", lambda case_id: [])
    results = [fake_result(tail=DOCKER_TAIL), fake_result()]                 # clean the second time
    d = run.run_dir(tmp_path, "main", "v1", "gen-002", 103)
    rec = run.run_one(d, "main", "v1", CASE, 103, {}, "m", health=lambda: None,
                      run_fix_loop=lambda *a, **k: results.pop(0))
    assert run.is_done(d) and rec["docker_line_kept"] is None and len(rec["reruns"]) == 1
    assert "dockerDesktopLinuxEngine" in rec["reruns"][0]["line"] and len(stopped(tmp_path)) == 1


def test_the_same_docker_error_twice_is_kept_as_a_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "loo_docs", lambda case_id: [])
    d = run.run_dir(tmp_path, "main", "v1", "gen-002", 104)
    rec = run.run_one(d, "main", "v1", CASE, 104, {}, "m", health=lambda: None,
                      run_fix_loop=lambda *a, **k: fake_result(tail=DOCKER_TAIL, outcome="failed at build"))
    assert run.is_done(d) and "dockerDesktopLinuxEngine" in rec["docker_line_kept"]
    assert rec["final_ok"] is False and len(rec["reruns"]) == 1 and len(stopped(tmp_path)) == 1


def test_a_docker_error_with_docker_down_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "loo_docs", lambda case_id: [])
    d = run.run_dir(tmp_path, "main", "v1", "gen-002", 105)
    with pytest.raises(run.Stop, match="Docker error"):
        run.run_one(d, "main", "v1", CASE, 105, {}, "m", health=lambda: "Docker is not answering",
                    run_fix_loop=lambda *a, **k: fake_result(tail=DOCKER_TAIL))
    assert not d.exists() and len(stopped(tmp_path)) == 1
