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
    case = MEMORY["gen-004"]                                 # Python 3.12 image; only a retrieved case says 3.10
    retrieved = [{"id": "gen-009", "error": "x", "cause": "xgboost 3.x requires Python 3.10 or newer"}]
    reason = "pkgutil.ImpImporter is gone in Python 3.12; Python 3.10 still has it"
    assert faithful(reason, case, retrieved) == (False, "names 3.10 from a retrieved case, not from this error")
    assert faithful(reason, case, retrieved, "set_python_version 3.10")[0]


def test_the_cases_own_cause_is_not_counted_as_copied():
    case = MEMORY["gen-007"]                                 # its cause names the server image v2.17.2
    retrieved = [{"id": "gen-008", "error": "x", "cause": "the tracking server is ghcr.io/mlflow/mlflow:v2.17.2"}]
    reason = "The 404 on /logged-models: the 3.x client calls an endpoint the v2.17.2 server doesn't have."
    assert faithful(reason, case, retrieved, "pin_package mlflow==2.17.2")[0]


# --- datasets ---------------------------------------------------------------------------------------------------------

def test_build_datasets_dedupes_and_pairs():
    cp = {"variant": "v", "family": "f", "prompt_version": "diagnose_v3", "prompt_sha256": "s",
          "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]}
    fix_ok, fix_bad = {"action": "add_dependency", "name": "x"}, {"action": "add_dependency", "name": "y"}
    rows = [{"case_id": "c", "seed": 1, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "a", "faithful": True},
            {"case_id": "c", "seed": 2, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "a", "faithful": True},
            {"case_id": "c", "seed": 3, "label": "correct", "match": "exact", "fix": fix_ok, "reason": "b", "faithful": False},
            {"case_id": "c", "seed": 4, "label": "wrong", "fix": fix_bad, "reason": "r"},
            {"case_id": "c", "seed": 5, "label": "invalid", "fix": fix_bad, "reason": "already applied"},
            {"case_id": "c", "seed": 6, "label": "invalid", "fix": None, "reason": None},
            {"case_id": "c", "seed": 1, "source": "rationalized", "label": "correct", "match": "exact", "fix": fix_ok,
             "reason": "c", "faithful": True}]
    sft, dpo, stats = build_datasets({"c": cp}, rows)
    assert [(r["source"], json.loads(r["messages"][-1]["content"])["reason"]) for r in sft] == [
        ("sampled", "a"), ("rationalized", "c")]
    # every sampled answer that isn't a faithful correct one is rejected once; chosen is the sampled answer
    assert [(d["rejected_kind"], d["chosen_source"]) for d in dpo] == [
        ("unfaithful", "sampled"), ("wrong", "sampled"), ("invalid", "sampled")]
    assert stats["f"] == {"cases": 1, "samples": 6, "correct": 3, "correct_dropped_unfaithful": 1, "wrong": 1,
                          "invalid": 2, "sft_examples": 2, "sft_sampled": 1, "sft_rationalized": 1, "sft_human": 0, "sft_written": 0,
                          "dpo_pairs": 3, "dpo_rejected_wrong": 1, "dpo_rejected_invalid": 1,
                          "dpo_rejected_unfaithful": 1}


def test_dpo_chosen_falls_back_to_rationalized_then_written():
    cp = {"variant": "v", "family": "f", "prompt_version": "diagnose_v3", "prompt_sha256": "s", "messages": []}
    wrong = {"case_id": "c", "seed": 1, "label": "wrong", "fix": {"action": "add_dependency", "name": "y"}, "reason": "r"}
    ok = {"case_id": "c", "label": "correct", "match": "exact", "fix": {"action": "add_dependency", "name": "x"},
          "faithful": True}
    written = ok | {"seed": None, "source": "written", "reason": "w"}
    rationalized = ok | {"seed": 2, "source": "rationalized", "reason": "z"}
    assert build_datasets({"c": cp}, [wrong, written])[1][0]["chosen_source"] == "written"
    assert build_datasets({"c": cp}, [wrong, written, rationalized])[1][0]["chosen_source"] == "rationalized"


def unique_lines(case):
    """Lines of a test case's error output that no memory case has (its own run names, ids, paths).
    Its error line alone can be shared: gen-001 has the same 404 as the memory cases gen-007 and gen-008."""
    memory_lines = {line for c in MEMORY.values() for line in c["error_tail"]}
    return [line for line in case["error_tail"] if len(line) > 40 and line not in memory_lines]


def test_every_test_case_has_lines_of_its_own():
    assert all(unique_lines(case) for case in TEST)


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
            assert case["variant"] not in text and f'"{case["id"]}"' not in text, (name, case["id"])
            for line in unique_lines(case):
                assert json.dumps(line)[1:-1] not in text, (name, case["id"], line)


def test_a_wrong_claim_about_the_image_python_version_is_unfaithful():
    case = MEMORY["gen-009"]                                 # xgboost 3.0.0 on a Python 3.9 image
    lie = "xgboost requires Python >= 3.10, and the image is currently using Python 3.12."
    assert faithful(lie, case, [], "set_python_version 3.10", "3.9") == (
        False, "says the image is Python 3.12; it is Python 3.9")
    requirement = "No matching distribution for xgboost==3.0.0: xgboost 3.x requires Python 3.10 or newer."
    assert faithful(requirement, case, [], "pin_package xgboost==2.0.0", "3.9")[0]   # a requirement, not the image
    ok = "xgboost 3.0.0 requires Python 3.10 or later; the image has Python 3.9."
    assert faithful(ok, case, [], "set_python_version 3.10", "3.9")[0]


def test_apt_get_is_not_a_copied_term():
    case = MEMORY["gen-006"]
    retrieved = [{"id": "gen-005", "error": MEMORY["gen-005"]["error_signature"], "cause": "x"}]
    assert faithful("libffi.so.8 is missing; install libffi8 with apt-get.", case, retrieved)[0]


# --- rationalization (STaR): hinted reasons, written fallback, sources, no hint in training data -----------

from builder_agent.finetune.rationalize import (HINT_MARK, hinted, judge, rationalize_case,  # noqa: E402
                                                training_rows, written_reason, written_row)


def test_hint_goes_at_the_end_of_the_user_message_only(mini):  # noqa: F811
    cp = prompt_for(mini, "gen-004")
    h = hinted(cp, MEMORY["gen-004"])
    assert h.messages[0] == cp.messages[0]
    assert h.messages[1]["content"].startswith(cp.messages[1]["content"])
    assert h.messages[1]["content"].endswith("not on the past failures from memory.")
    assert "`set_python_version 3.9`" in h.messages[1]["content"] and HINT_MARK not in cp.messages[1]["content"]


@pytest.mark.parametrize("fix,reason,kept,why", [
    ({"action": "set_python_version", "version": "3.9"},
     "pkgutil.ImpImporter was removed after Python 3.9, so the build fails on the 3.12 image.", True, "refers to"),
    ({"action": "set_python_version", "version": "3.10"}, "pkgutil is gone.", False, "changed the fix"),
    ({"action": "set_python_version", "version": "3.9"}, "Something is wrong with the build.", False, "names no term"),
])
def test_judge(mini, fix, reason, kept, why):  # noqa: F811
    cp = prompt_for(mini, "gen-004")
    row = judge({"response": answer(fix, reason)}, MEMORY["gen-004"], cp)
    assert row["kept"] is kept and why in row["why"] and row["source"] == "rationalized"


def test_rationalize_case_samples_the_hinted_prompt(mini):  # noqa: F811
    cp = prompt_for(mini, "gen-002")
    good = json.loads(answer({"action": "add_dependency", "name": "xgboost"}, "ModuleNotFoundError: no xgboost."))
    models = []

    def factory(seed):
        models.append(FakeModel(good))
        return models[-1]
    rows = rationalize_case(cp, MEMORY["gen-002"], factory, n=4)
    assert len(rows) == 4 and all(r["hinted"] and r["kept"] for r in rows)
    assert all(HINT_MARK in m.sent[0][1]["content"] for m in models)


@pytest.mark.parametrize("case_id", sorted(MEMORY))
def test_written_reasons_are_two_sentences_and_pass_the_filter(case_id):
    """On the real taxi prompts (the recorded causes are about the taxi repo, e.g. its Python 3.9)."""
    path = OUT_DIR / "prompts.json"
    if not path.exists():
        pytest.skip("prompts.json not generated yet")
    cp = CasePrompt(**json.loads(path.read_text(encoding="utf-8"))[case_id])
    row = written_row(MEMORY[case_id], cp)
    assert row["kept"], row["why"]
    assert row["reason"].startswith(f"The {MEMORY[case_id]['stage']} stage fails with: ")
    assert len(written_reason(MEMORY[case_id])) < 400


def test_training_rows_fall_back_to_a_written_reason(mini):  # noqa: F811
    prompts = {cid: prompt_for(mini, cid) for cid in ("gen-002", "gen-007")}
    kept = {"case_id": "gen-002", "seed": 1, "source": "rationalized", "kept": True, "why": "refers to xgboost",
            "fix": MEMORY["gen-002"]["expected_fix"], "fix_text": "add_dependency xgboost", "reason": "No xgboost."}
    lost = kept | {"case_id": "gen-007", "kept": False}
    rows = training_rows([kept, lost], {cid: MEMORY[cid] for cid in prompts}, prompts)
    assert [(r["case_id"], r["source"]) for r in rows] == [("gen-002", "rationalized"), ("gen-007", "written")]


def test_training_records_use_the_prompt_without_the_hint(mini):  # noqa: F811
    from builder_agent.finetune import build_datasets
    cp = prompt_for(mini, "gen-007")
    prompts = {"gen-007": cp.model_dump()}
    wrong = {"case_id": "gen-007", "seed": 1, "label": "wrong", "fix": {"action": "set_env_var",
             "name": "GIT_PYTHON_REFRESH", "value": "quiet"}, "reason": "git warning"}
    rows = [wrong] + training_rows([], {"gen-007": MEMORY["gen-007"]}, {"gen-007": cp})
    sft, dpo, stats = build_datasets(prompts, rows)
    assert [r["source"] for r in sft] == ["written"] and dpo[0]["chosen_source"] == "written"
    assert stats["version_mismatch"]["samples"] == 1 and stats["version_mismatch"]["sft_written"] == 1
    for text in [json.dumps(r) for r in sft + dpo]:
        assert HINT_MARK not in text


@pytest.mark.parametrize("name", ["sft.jsonl", "dpo.jsonl"])
def test_no_training_prompt_contains_the_hint(name):
    path = OUT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    for line in path.read_text(encoding="utf-8").splitlines():
        assert HINT_MARK not in line


def test_every_memory_case_has_an_sft_example():
    path = OUT_DIR / "sft.jsonl"
    if not (path.exists() and (OUT_DIR / "rationalized.jsonl").exists()):
        pytest.skip("sft.jsonl not built with rationalization yet")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["case_id"] for r in rows} == set(MEMORY)
    assert all(r["source"] in ("sampled", "rationalized", "written") for r in rows)


@pytest.mark.parametrize("reason", ["pkgutil.ImpImporter is gone in 3.12; switching to 3.9, as verified, fixes it.",
                                    "The hint says pkgutil fails on 3.12, so use 3.9."])
def test_a_reason_that_mentions_the_hint_is_not_kept(mini, reason):  # noqa: F811
    row = judge({"response": answer({"action": "set_python_version", "version": "3.9"}, reason)},
                MEMORY["gen-004"], prompt_for(mini, "gen-004"))
    assert not row["kept"] and "refers to the hint" in row["why"]


# --- human reasons and the per-case cap ---------------------------------------------------------------------

from builder_agent.finetune import SOURCE_PRIORITY, cap_per_case  # noqa: E402
from builder_agent.finetune.human import TEMPLATE, load_human  # noqa: E402


def test_human_reasons_are_checked_like_the_models(mini, tmp_path):  # noqa: F811
    fix = MEMORY["gen-004"]["expected_fix"]
    good = ("AttributeError: pkgutil has no attribute ImpImporter: the old setuptools in Pipfile.lock "
            "needs pkgutil.ImpImporter, which Python 3.12 removed; Python 3.9 still has it.")
    path = tmp_path / "human.json"
    path.write_text(json.dumps({"_how": "notes", "gen-004": [
        {"fix": fix, "reason": ""},                                                   # placeholder: skipped
        {"fix": fix, "reason": good},
        {"fix": {"action": "set_python_version", "version": "3.10"}, "reason": good},  # not the verified fix
        {"fix": fix, "reason": "Use the right Python."}],                             # names no term of the error
        "gen-001": [{"fix": {}, "reason": "x"}]}), encoding="utf-8")                  # a test case
    rows, problems = load_human(path, MEMORY, {"gen-004": prompt_for(mini, "gen-004")})
    assert [(r["source"], r["reason"]) for r in rows] == [("human", good)]
    assert [p.split(":")[0] for p in problems] == ["gen-004 #3", "gen-004 #4", "gen-001"]
    assert "not the verified fix" in problems[0] and "names no term" in problems[1] and "not a memory case" in problems[2]
    assert load_human(tmp_path / "missing.json", MEMORY, {}) == ([], [])


def test_human_template_is_a_memory_case_with_its_verified_fix():
    assert [k for k in TEMPLATE if not k.startswith("_")] == ["gen-004"]
    assert TEMPLATE["gen-004"] == [{"fix": MEMORY["gen-004"]["expected_fix"], "reason": ""}]


def test_dpo_prefers_human_over_written_but_not_over_the_models_answers():
    assert SOURCE_PRIORITY == ("sampled", "rationalized", "human", "written")
    cp = {"variant": "v", "family": "f", "prompt_version": "diagnose_v3", "prompt_sha256": "s", "messages": []}
    wrong = {"case_id": "c", "seed": 1, "label": "wrong", "fix": {"action": "add_dependency", "name": "y"}, "reason": "r"}
    ok = {"case_id": "c", "seed": None, "label": "correct", "match": "exact", "faithful": True,
          "fix": {"action": "add_dependency", "name": "x"}}
    sft, dpo, stats = build_datasets({"c": cp}, [wrong, ok | {"source": "written", "reason": "w"},
                                                 ok | {"source": "human", "reason": "h"}])
    assert dpo[0]["chosen_source"] == "human" and stats["f"]["sft_human"] == 1 and len(sft) == 2


def test_cap_keeps_at_most_four_per_case_human_first():
    rows = ([{"case_id": "a", "source": "rationalized", "n": i} for i in range(3)]
            + [{"case_id": "a", "source": "sampled", "n": i} for i in range(3)]
            + [{"case_id": "a", "source": "human", "n": 0}, {"case_id": "b", "source": "written", "n": 0}])
    capped = cap_per_case(rows)
    assert [(r["case_id"], r["source"], r["n"]) for r in capped] == [
        ("a", "human", 0), ("a", "sampled", 0), ("a", "sampled", 1), ("a", "sampled", 2), ("b", "written", 0)]


def test_human_reasons_file_holds_only_memory_cases():
    path = OUT_DIR / "human_reasons.json"
    if not path.exists():
        pytest.skip("human_reasons.json not created yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {k for k in data if not k.startswith("_")} <= set(MEMORY)


def test_qlora_notebook_parses_and_matches_the_cap_rule():
    import ast
    import re
    from builder_agent.finetune import MAX_PER_CASE, SFT_CAP_ORDER

    nb = json.loads((Path(__file__).parents[1] / "notebooks" / "qlora_sft.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse("\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")      # not run
    hyper = next(src for src in code if "LORA_R = " in src)
    settings = [line for line in hyper.splitlines() if re.match(r"^[A-Z_0-9]+ = ", line)]
    assert len(settings) >= 15 and all("  # " in line for line in settings)                 # one comment each
    assert f"MAX_PER_CASE = {MAX_PER_CASE} " in hyper
    cap_order = re.search(r"^CAP_ORDER = (\(.*\))$", "\n".join(code), re.M).group(1)
    assert ast.literal_eval(cap_order) == SFT_CAP_ORDER
    assert 'MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"' in hyper and 'GGUF_QUANT = "q4_k_m"' in hyper
    assert 'OLLAMA_NAME = "qwen-builder-sft"' in hyper and "train_on_responses_only" in "".join(code)
