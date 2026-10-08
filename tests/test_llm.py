"""Slot-filling loop with a scripted fake model (no Ollama)."""

import copy
import json
from pathlib import Path

import pytest

from builder_agent.decide import plan_build
from builder_agent.llm import PromptError, fill_slots, load_prompt
from builder_agent.llm.eval import SLOTS, compare, run_eval
from builder_agent.llm.ollama import ChatResponse, inline_refs
from builder_agent.llm.prompts import slot_variables
from builder_agent.render.slots import SlotAnswers

from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)
from test_scan import write

FIXTURES = Path(__file__).parent / "fixtures" / "prompts"


class FakeModel:
    """Returns scripted answers in order; records the messages it was sent."""
    model, options, digest = "fake:1b", {"temperature": 0, "seed": 42}, "sha256:fake"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sent: list[list[dict]] = []

    def chat(self, messages, schema=None):
        self.sent.append(copy.deepcopy(messages))
        self.schema = schema
        a = self.answers.pop(0)
        return ChatResponse(content=a if isinstance(a, str) else json.dumps(a), latency_s=0.5,
                            prompt_tokens=100, output_tokens=20)


@pytest.fixture
def prompt():
    return load_prompt("fixture_v1", FIXTURES)


# --- prompt files --------------------------------------------------------------

def test_prompt_sections_and_version(prompt):
    assert prompt.version == "fixture_v1" and len(prompt.sha256) == 64
    assert set(prompt.sections) == {"system", "user", "retry"}
    assert prompt.sections["system"].strip() == "Answer as JSON."


def test_prompt_missing_section(tmp_path):
    write(tmp_path, {"p_v1.md": "## system\nx\n## user\ny\n"})
    with pytest.raises(PromptError, match="missing section.*retry"):
        load_prompt("p_v1", tmp_path)


def test_unknown_placeholder_names_it(tmp_path, mini, contracts):  # noqa: F811
    write(tmp_path, {"p_v1.md": "## system\n{{ repo_name }}\n## user\nx\n## retry\ny\n"})
    p = load_prompt("p_v1", tmp_path)
    with pytest.raises(PromptError, match=r"\[system\]: 'repo_name' is undefined; available: .*repo"):
        p.render("system", slot_variables(mini, plan_build(mini, contracts)))


def test_missing_prompt_file(tmp_path):
    with pytest.raises(PromptError, match="prompt file not found"):
        load_prompt("nope_v9", tmp_path)


def test_variables_come_from_plan_and_candidates(mini, contracts):  # noqa: F811
    v = slot_variables(mini, plan_build(mini, contracts))
    assert v["candidates"]["target_column"][0] == "price"
    assert json.loads(v["candidates_json"]) == v["candidates"]
    assert "target:train" in v["needs_llm_text"] or v["needs_llm"] == []
    assert v["schema"]["title"] == "SlotAnswers"


def test_schema_is_inlined():
    schema = inline_refs(SlotAnswers.model_json_schema())
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["train"]["properties"]["arg_map"]["type"] == "object"


# --- the loop ------------------------------------------------------------------------

def test_valid_on_first_try(mini, contracts, prompt, tmp_path):  # noqa: F811
    model = FakeModel(MINI_GOLD)
    log = tmp_path / "calls.jsonl"
    r = fill_slots(mini, plan_build(mini, contracts), prompt, model, log_path=log)
    assert r.valid and r.valid_on_first_try and len(r.attempts) == 1 and r.answer == MINI_GOLD
    assert r.total_tokens == 120 and r.total_s == 0.5
    assert [m["role"] for m in model.sent[0]] == ["system", "user"]
    assert "Targets: price" in model.sent[0][1]["content"]
    assert model.schema["title"] == "SlotAnswers"
    [entry] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entry["prompt_version"] == "fixture_v1" and entry["model_digest"] == "sha256:fake"
    assert entry["valid"] and entry["latency_s"] == 0.5 and entry["prompt_tokens"] == 100
    assert entry["prompt_sha256"] == prompt.sha256 and entry["messages"][0]["role"] == "system"


def test_retry_sends_reasons_then_accepts(mini, contracts, prompt):  # noqa: F811
    wrong = dict(MINI_GOLD, target_column="duration")
    model = FakeModel(wrong, MINI_GOLD)
    r = fill_slots(mini, plan_build(mini, contracts), prompt, model, log_path=None)
    assert r.valid and not r.valid_on_first_try and len(r.attempts) == 2
    retry = model.sent[1]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert json.loads(retry[2]["content"]) == wrong                       # its own previous answer
    assert retry[3]["content"].startswith("Attempt 2. Rejected:")
    assert "target_column 'duration' is not a scanned target candidate" in retry[3]["content"]


def test_gives_up_after_three_attempts(mini, contracts, prompt):  # noqa: F811
    model = FakeModel("not json", ["a list"], dict(MINI_GOLD, model_flavor="lightgbm"))
    r = fill_slots(mini, plan_build(mini, contracts), prompt, model, log_path=None)
    assert not r.valid and len(r.attempts) == 3
    assert "not valid JSON" in r.attempts[0].reasons[0]
    assert r.attempts[1].reasons == ["the answer must be one JSON object"]
    assert any("lightgbm" in reason for reason in r.attempts[2].reasons)
    assert r.answer["model_flavor"] == "lightgbm"                          # last parsed answer kept


# --- eval ----------------------------------------------------------------------------

def test_compare_slot_by_slot():
    gold = MINI_GOLD
    answer = copy.deepcopy(gold)
    answer["evaluate"]["group_column"] = "a"
    del answer["data"]
    scores = {s.slot: s for s in compare(answer, gold)}
    assert len(scores) == len(SLOTS) == 11
    assert not scores["evaluate.group_column"].correct and not scores["data.data_step"].correct
    assert scores["data.data_step"].answer is None
    assert sum(s.correct for s in scores.values()) == 9
    assert sum(s.correct for s in compare(None, gold)) == 0


def test_run_eval_report(mini, contracts, tmp_path):  # noqa: F811
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(MINI_GOLD), encoding="utf-8")
    model = FakeModel(dict(MINI_GOLD, target_column="duration"), MINI_GOLD)
    report = run_eval("fixture_v1", "fake:1b", mini.root, gold, call_log=None, client=model,
                      prompts_dir=FIXTURES)
    assert (report.slots_correct, report.slots_total) == (11, 11)
    assert (report.valid_final, report.valid_on_first_try, report.attempts) == (True, False, 2)
    assert report.total_tokens == 240
    text = report.text()
    assert "slots correct      11/11" in text and "valid on 1st try   no" in text
    assert "attempt 1 rejected: target_column 'duration'" in text
    json.loads(report.model_dump_json())                                   # serializable


def test_lenient_score_accepts_alternatives():
    answer = dict(MINI_GOLD, train=dict(MINI_GOLD["train"], train_file="data/processed/other.csv"))
    scores = {s.slot: s for s in compare(answer, MINI_GOLD, {"train.train_file": ["data/processed/other.csv"]})}
    s = scores["train.train_file"]
    assert (s.correct, s.correct_lenient) == (False, True)
    assert sum(x.correct for x in scores.values()) == 10 and sum(x.correct_lenient for x in scores.values()) == 11


def test_report_label_sandbox_and_side_by_side(mini, contracts, tmp_path):  # noqa: F811
    from builder_agent.llm.eval import EvalReport, side_by_side

    class Runner:   # fake docker: every stage passes, the predict stage prints a prediction
        def __call__(self, cmd, cwd, timeout):
            return 0, "prediction: 98.5\nsmoke test passed" if "predict" in cmd else "ok"

    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(dict(MINI_GOLD, _alternatives={"evaluate.group_column": ["a"]})), encoding="utf-8")
    report = run_eval("fixture_v1", "fake:1b", mini.root, gold, call_log=None, client=FakeModel(MINI_GOLD),
                      prompts_dir=FIXTURES, label="v1, fixed system", sandbox=True, sandbox_runner=Runner())
    assert report.label == "v1, fixed system" and report.slots_correct_lenient == 11
    assert report.sandbox.ok and report.sandbox.prediction == 98.5
    assert "end to end         all stages ok, prediction 98 s" in report.text()
    again = EvalReport.model_validate_json(report.model_dump_json())
    table = side_by_side([again, report])
    assert "v1, fixed system" in table and "slots correct (lenient)" in table and "all stages ok" in table
