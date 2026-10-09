"""Transport retry, failed-call logging, clearer rejection reasons, eval/train row overlap, multi-run summary."""

import copy
import io
import json
import urllib.error
from pathlib import Path

import pytest

from builder_agent.decide import plan_build
from builder_agent.llm import fill_slots, load_prompt
from builder_agent.llm.eval import EvalReport, SLOTS, compare, summarize
from builder_agent.llm.ollama import ChatResponse, OllamaClient, OllamaError
from builder_agent.render.slots import validate_slots
from builder_agent.scan import rows, scan_repo

from test_llm import FIXTURES, FakeModel
from test_render import MINI_GOLD, TAXI, TAXI_GOLD, changed, contracts, mini, taxi, taxi_only  # noqa: F401
from test_scan import write


# --- (1) transport retry and failed-call logging -------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


OK_BODY = json.dumps({"message": {"content": "{}"}, "prompt_eval_count": 5, "eval_count": 2}).encode()


def _http_error(code):
    return urllib.error.HTTPError("http://x/api/chat", code, "err", {}, io.BytesIO(b'{"error":"llm server error"}'))


def _patch_urlopen(monkeypatch, outcomes):
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _Resp(out)
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return calls


def test_5xx_is_retried_once(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(500), OK_BODY])
    resp = OllamaClient("m", retry_wait_s=0).chat([{"role": "user", "content": "x"}])
    assert len(calls) == 2 and resp.content == "{}"
    assert resp.transport_errors == ['HTTP 500 from Ollama: {"error":"llm server error"}']


def test_second_5xx_raises_with_both_errors(monkeypatch):
    _patch_urlopen(monkeypatch, [_http_error(500), _http_error(503)])
    with pytest.raises(OllamaError) as e:
        OllamaClient("m", retry_wait_s=0).chat([])
    assert [m[:8] for m in e.value.errors] == ["HTTP 500", "HTTP 503"]


def test_4xx_is_not_retried(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(400), OK_BODY])
    with pytest.raises(OllamaError, match="HTTP 400"):
        OllamaClient("m", retry_wait_s=0).chat([])
    assert len(calls) == 1


def test_connection_refused_is_retried(monkeypatch):
    _patch_urlopen(monkeypatch, [urllib.error.URLError(ConnectionRefusedError("refused")), OK_BODY])
    assert OllamaClient("m", retry_wait_s=0).chat([]).transport_errors[0].startswith("can't reach Ollama")


class FailingModel(FakeModel):
    def chat(self, messages, schema=None):
        if not self.answers:
            raise OllamaError(["HTTP 500 from Ollama: crash", "HTTP 500 from Ollama: crash"])
        return super().chat(messages, schema)


def test_failed_call_is_logged_and_stops_the_loop(mini, contracts, tmp_path):  # noqa: F811
    log = tmp_path / "calls.jsonl"
    model = FailingModel(dict(MINI_GOLD, target_column="duration"))   # attempt 1 rejected, attempt 2 fails
    r = fill_slots(mini, plan_build(mini, contracts), load_prompt("fixture_v1", FIXTURES), model, log_path=log)
    assert not r.valid and r.error.startswith("OllamaError: HTTP 500") and len(r.attempts) == 1
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["attempt"] for e in entries] == [1, 2]
    assert entries[1]["error"].startswith("OllamaError") and entries[1]["response"] is None
    assert entries[1]["valid"] is False and entries[1]["messages"][-1]["role"] == "user"


# --- (2) clearer rejection reasons -------------------------------------------------------

def test_swapped_arg_map_is_named(mini):  # noqa: F811
    swapped = changed(MINI_GOLD, "train.arg_map", {"$train_path": "data/processed/train.csv",
                                                   "$target_column": "price"})
    reasons = validate_slots(mini, swapped).reasons
    swap = [r for r in reasons if "wrong way round" in r]
    assert swap == ["train.arg_map is the wrong way round: its KEYS must be train's parameter names "
                    "(train_path, target_col, rounds) and its VALUES tokens ($train_path, $train_df, $X, $y, "
                    "$target_column). You used $train_path, $target_column as keys. "
                    'Expected shape: {"train_path": <token>, "target_col": <token>}']
    assert not any("has no parameter" in r or "is a data file path" in r for r in reasons)   # no noise


def test_unknown_parameter_lists_the_real_ones(mini):  # noqa: F811
    answer = changed(MINI_GOLD, "train.arg_map", {"train_path": "$train_path", "target_column": "$target_column"})
    reasons = validate_slots(mini, answer).reasons
    assert "train.arg_map: train has no parameter target_column; its parameters are train_path, target_col, rounds" \
        in reasons


def test_transform_reason_names_the_evidence(mini):  # noqa: F811
    reasons = validate_slots(mini, changed(MINI_GOLD, "transform_inside_train_fn", False)).reasons
    assert any("the scanner saw log1p applied to the target (via target_col) in src/train_model.py, the file of "
               "train: if train applies it itself, set transform_inside_train_fn to true" in r for r in reasons)


# --- (3) eval_file must not share rows with train_file -------------------------------------

@pytest.fixture
def split_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(rows, "SAMPLE_MOD", 1)                      # sample every row
    monkeypatch.setattr(rows, "CACHE_FILE", tmp_path / "cache" / "rows.json")
    head = "a,b,price\n"
    train = "".join(f"{i},{i * 2},{i * 10}\n" for i in range(20))
    val = "".join(f"{i},{i * 2},{i * 10}\n" for i in range(20, 30))
    test = "".join(f"{i},{i * 2},{i * 10}\n" for i in range(30, 40))
    root = write(tmp_path / "repo", {
        "requirements.txt": "xgboost\npandas\nnumpy\n",
        ".python-version": "3.11\n",
        "data/processed/train.csv": head + train,
        "data/processed/validation.csv": head + val,
        "data/processed/train_large.csv": head + train + val,
        "data/processed/test.csv": head + test,
        "src/train_model.py": ("import numpy as np\nimport pandas as pd\nimport xgboost as xgb\n"
                               "def train(train_path, target_col):\n    df = pd.read_csv(train_path)\n"
                               "    y = np.log1p(df[target_col].values)\n"
                               "    return xgb.train({}, xgb.DMatrix(df.drop(target_col, axis=1).values, label=y))\n"),
        "nb.ipynb": json.dumps({"cells": [{"cell_type": "code", "source":
                                "from src.train_model import train\ntrain('data/processed/train.csv', 'price')"}]}),
    })
    return scan_repo(root)


SPLIT_ANSWER = {
    "target_column": "price", "target_transform": "log1p", "transform_inside_train_fn": True,
    "model_flavor": "xgboost", "model_input": "dmatrix",
    "train": {"train_function": "src.train_model:train", "train_file": "data/processed/train_large.csv",
              "arg_map": {"train_path": "$train_path", "target_col": "$target_column"}},
    "evaluate": {"eval_file": "data/processed/test.csv", "group_column": "a"},
    "data": {"data_step": "existing"},
}


def test_scanner_finds_containment(split_repo):
    rels = {(o.a, o.b): o for o in split_repo.data_overlaps}
    o = rels[("data/processed/train_large.csv", "data/processed/validation.csv")]
    assert (o.b_in_a, o.shared_in_sample) == (1.0, 10)              # train_large contains validation
    assert ("data/processed/test.csv", "data/processed/train_large.csv") not in rels
    assert "train_large.csv contains data/processed/validation.csv (10/10 sampled rows)" in split_repo.summary()
    assert {d.path: d.n_rows for d in split_repo.data_columns}["data/processed/train_large.csv"] == 30


def test_eval_file_inside_train_file_is_rejected(split_repo):
    v = validate_slots(split_repo, changed(SPLIT_ANSWER, "evaluate.eval_file", "data/processed/validation.csv"))
    assert any("evaluate.eval_file data/processed/validation.csv shares rows with train.train_file "
               "data/processed/train_large.csv: about 100% of its rows are in the training file" in r
               and r.endswith("Files with no shared rows: data/processed/test.csv") for r in v.reasons), v.reasons


def test_eval_file_equal_to_train_file_is_rejected(split_repo):
    v = validate_slots(split_repo, changed(SPLIT_ANSWER, "evaluate.eval_file", "data/processed/train_large.csv"))
    assert any("is the training file" in r for r in v.reasons)


def test_disjoint_split_is_accepted(split_repo):
    answer = changed(changed(SPLIT_ANSWER, "train.train_file", "data/processed/train.csv"),
                     "evaluate.eval_file", "data/processed/validation.csv")
    assert validate_slots(split_repo, answer).ok
    assert validate_slots(split_repo, SPLIT_ANSWER).ok


def test_row_samples_are_cached(split_repo, monkeypatch):
    monkeypatch.setattr(rows, "_sample_file", lambda p: pytest.fail("should come from the cache"))
    scan_repo(split_repo.root)


@taxi_only
def test_taxi_validation_inside_train_large(taxi):  # noqa: F811
    bad = changed(TAXI_GOLD, "evaluate.eval_file", "data/processed/validation.csv")
    assert any("about 100% of its rows are in the training file" in r for r in validate_slots(taxi, bad).reasons)
    ok = changed(changed(TAXI_GOLD, "train.train_file", "data/processed/train.csv"),
                 "evaluate.eval_file", "data/processed/validation.csv")
    assert validate_slots(taxi, ok).ok                              # train/validation is a clean split


# --- multi-run summary -----------------------------------------------------------------------

def _report(seed, strict, tokens, valid, answer):
    scores = compare(answer, MINI_GOLD)
    return EvalReport(time="2026-10-08T00:00:00+00:00", repo="r", prompt_version="p", prompt_sha256="h",
                      model="m", model_digest="d", label="L", seed=seed, slots_correct=strict,
                      slots_correct_lenient=strict, slots_total=11, per_slot=scores, final_answer=answer,
                      valid_final=valid, valid_on_first_try=False, attempts=2, rejection_reasons=[],
                      total_tokens=tokens, llm_s=7.0, wall_s=7.2)


def test_summary_mean_spread_and_counts():
    other = copy.deepcopy(MINI_GOLD)
    other["evaluate"]["group_column"] = "b"
    reports = [_report(1, 11, 6000, True, MINI_GOLD), _report(2, 10, 5000, True, other),
               _report(3, 11, 7000, False, MINI_GOLD)]
    s = summarize(reports, ["a.json", "b.json", "c.json"])
    assert (s.runs, s.seeds, s.valid_final, s.distinct_final_answers) == (3, [1, 2, 3], 2, 2)
    assert (s.slots_correct.mean, s.slots_correct.min, s.slots_correct.max) == (10.667, 10, 11)
    assert s.total_tokens.std == 1000.0 and s.per_slot_correct["evaluate.group_column"] == 2
    assert set(s.per_slot_correct) == set(SLOTS)
    text = s.text()
    assert "valid at the end         2/3" in text and "distinct final answers   2" in text


def test_ollama_unloads_the_model_after_each_call_by_default(monkeypatch):
    sent = []

    def fake_post(self, data, errors):
        sent.append(json.loads(data))
        return ChatResponse(content="{}", latency_s=0.0)
    monkeypatch.setattr(OllamaClient, "_post", fake_post)
    OllamaClient("m").chat([])
    OllamaClient("m", keep_alive="5m").chat([])
    assert [b["keep_alive"] for b in sent] == [0, "5m"]
