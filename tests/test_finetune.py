"""Fine-tuning data: only split=memory cases, the case left out of its own retrieval, labels, faithful
reasons, and the datasets never containing a held-out test case. Fake embedder and model (no Ollama)."""

import json
from pathlib import Path

import pytest

from builder_agent.finetune import (OUT_DIR, CasePrompt, DataError, build_datasets, faithful, label_sample, match,
                                    render_case_prompt, sample_case, split_cases)
from builder_agent.fix.models import AddDependency, PinPackage, SetEnvVar

from test_llm import FakeModel
from test_memory import FakeEmbedder
from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)

MEMORY = {c["id"]: c for c in split_cases("memory")}
TEST = split_cases("test")


def prompt_for(mini, case_id):  # noqa: F811
    return render_case_prompt(MEMORY[case_id], mini.root, MINI_GOLD, FakeEmbedder())


# --- which cases, and what they retrieve ---------------------------------------------------------------

def test_only_memory_cases_make_prompts(mini):  # noqa: F811
    assert len(MEMORY) == 9 and len(TEST) == 4
    with pytest.raises(DataError, match="only split=memory"):
        render_case_prompt(TEST[0], mini.root, MINI_GOLD, FakeEmbedder())


@pytest.mark.parametrize("case_id", sorted(MEMORY))
def test_a_case_never_retrieves_itself(mini, case_id):  # noqa: F811
    cp = prompt_for(mini, case_id)
    ids = [r["id"] for r in cp.retrieved]
    assert case_id not in ids and len(ids) == 3
    assert not set(ids) & {c["id"] for c in TEST}
    user = cp.messages[1]["content"]
    assert MEMORY[case_id]["error_signature"][:60] in user                    # its own error, in the error output
    assert "FIXES ALREADY TRIED:\nnone" in user and "(fix attempt 1 of 3)" in user


def test_gen_008_also_leaves_out_fault_001_the_same_fault(mini):  # noqa: F811
    cp = prompt_for(mini, "gen-008")
    assert cp.excluded == ["fault-001", "gen-008"] and "fault-001" not in [r["id"] for r in cp.retrieved]


# --- sampling and labels --------------------------------------------------------------------------------------

def answer(fix, reason="x"):
    return json.dumps({"fix": fix, "reason": reason})


def test_samples_are_logged_with_their_seed(mini):  # noqa: F811
    cp = prompt_for(mini, "gen-002")
    rows = sample_case(cp, lambda seed: FakeModel(json.loads(answer({"action": "add_dependency", "name": "xgboost"}))),
                       n=3)
    assert [r["seed"] for r in rows] == [1, 2, 3] and all(r["model_digest"] == "sha256:fake" for r in rows)
    assert all(r["case_id"] == "gen-002" and r["prompt_version"] == "diagnose_v3" for r in rows)


def test_match():
    expected = {"action": "pin_package", "name": "mlflow", "version": "2.17.2"}
    assert match(PinPackage(action="pin_package", name="mlflow", version="2.17.2"), expected) == "exact"
    assert match(PinPackage(action="pin_package", name="MLflow", version="2.10.0"), expected) == "loose"
    assert match(PinPackage(action="pin_package", name="werkzeug", version="2.3.1"), expected) is None
    env = {"action": "set_env_var", "name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
    assert match(SetEnvVar(action="set_env_var", name="MLFLOW_TRACKING_URI", value="http://x:1"), env) is None
    assert match(AddDependency(action="add_dependency", name="xgboost"),
                 {"action": "add_dependency", "name": "xgboost"}) == "exact"


@pytest.mark.parametrize("fix,label", [
    ({"action": "add_dependency", "name": "xgboost"}, "correct"),
    ({"action": "pin_package", "name": "xgboost", "version": "2.0.0"}, "candidate"),   # xgboost is in the error
    ({"action": "add_system_package", "name": "libgomp1"}, "wrong"),                   # not named in the error
    ({"action": "add_system_package", "name": "clang++"}, "invalid"),                  # the validator rejects it
])
def test_labels(mini, fix, label):  # noqa: F811
    cp = prompt_for(mini, "gen-002")
    row = {"response": answer(fix), "error": None}
    assert label_sample(row, MEMORY["gen-002"], cp)["label"] == label


# --- faithful reasons ---------------------------------------------------------------------------------------------

def test_faithful_reason_refers_to_this_error_and_not_a_retrieved_one():
    case = MEMORY["gen-012"]                                 # wrong port 5001
    retrieved = [{"id": "gen-005", "error": MEMORY["gen-005"]["error_signature"], "cause": MEMORY["gen-005"]["cause"]
                  ["summary"]}]
    assert faithful("The client can't reach port 5001: nothing listens there.", case, retrieved)[0]
    ok, why = faithful("The host mlflow-server doesn't resolve.", case, retrieved)
    assert (ok, why) == (False, "names server from a retrieved case, not from this error")   # gen-005's host
    assert faithful("Something is wrong.", case, retrieved) == (False, "names no term of this case's error line")


def test_the_answers_own_fix_terms_are_not_counted_as_copied():
    case = MEMORY["gen-004"]                                 # Python 3.12 image; a retrieved case mentions 3.9
    retrieved = [{"id": "gen-009", "error": "x", "cause": "xgboost 3.x requires Python 3.10; the image is Python 3.9"}]
    reason = "pkgutil.ImpImporter is gone in Python 3.12; the lock file is for 3.9"
    assert not faithful(reason, case, retrieved)[0]
    assert faithful(reason, case, retrieved, "set_python_version 3.9")[0]


# --- datasets ---------------------------------------------------------------------------------------------------------

def test_build_datasets_dedupes_and_pairs():
    cp = {"variant": "v", "family": "f", "prompt_version": "diagnose_v3", "prompt_sha256": "s",
          "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]}
    fix_ok, fix_bad = {"action": "add_dependency", "name": "x"}, {"action": "add_dependency", "name": "y"}
    rows = [{"case_id": "c", "seed": 1, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "a", "faithful": True},
            {"case_id": "c", "seed": 2, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "a", "faithful": True},
            {"case_id": "c", "seed": 3, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "b", "faithful": False},
            {"case_id": "c", "seed": 4, "label": "wrong", "fix": fix_bad, "reason": "r"},
            {"case_id": "c", "seed": 5, "label": "invalid", "fix": None, "reason": None}]
    sft, dpo, stats = build_datasets({"c": cp}, rows)
    assert len(sft) == 1 and sft[0]["messages"][-1] == {"role": "assistant", "content": json.dumps(
        {"fix": fix_ok, "reason": "a"})}
    assert len(dpo) == 1 and json.loads(dpo[0]["rejected"])["fix"] == fix_bad
    assert stats["f"] | {} == {"cases": 1, "samples": 5, "correct": 3, "correct_dropped_unfaithful": 1, "wrong": 1,
                               "invalid": 1, "sft_examples": 1, "dpo_pairs": 1}


@pytest.mark.parametrize("name", ["sft.jsonl", "dpo.jsonl"])
def test_no_held_out_test_case_in_the_datasets(name):
    """Fails if a split=test case appears in a dataset: as the source case, or anywhere in a prompt or answer."""
    path = OUT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert row["case_id"] in MEMORY and row["variant"] == MEMORY[row["case_id"]]["variant"]
        text = json.dumps(row, ensure_ascii=False)
        for case in TEST:
            assert case["variant"] not in text and case["error_signature"][:60] not in text, (name, case["id"])
            assert f'"{case["id"]}"' not in text
