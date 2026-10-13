"""Runtime slot retries (ROADMAP B6): the rule that says whether a sandbox failure comes from a slot answer,
and the create command sending such an error back to the slot LLM. Fake LLM, fake Docker."""

import json
from pathlib import Path

from builder_agent.create import create
from builder_agent.fix.origin import changed_slots, classify, outside_group, tracebacks
from builder_agent.llm import load_prompt

from test_llm import FakeModel
from test_repo_rule import git
from test_sandbox import FakeRunner
from test_script_mode import BASE, MLFLOW_RUN, MLFLOW_SCRIPT, script_repo

GIT_WARNING = ["WARNING mlflow.utils.git_utils: Failed to import Git. Error: Failed to initialize: Bad git executable."]
ARGV_TRACE = ["Traceback (most recent call last):", '  File "/app/scripts/train.py", line 10, in <module>',
              "    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0", "            ^^^^^^^^^^^^^^^^^^",
              "ValueError: could not convert string to float: 'price'"]
ADAPTER_TRACE = ["Traceback (most recent call last):", '  File "/app/pipeline/train.py", line 72, in main',
                 "    subprocess.run([sys.executable, str(ROOT / SCRIPT), *args], cwd=ROOT, env=env, check=True)",
                 '  File "/usr/local/lib/python3.11/subprocess.py", line 571, in run',
                 "subprocess.CalledProcessError: Command '[...]' returned non-zero exit status 1."]
FILES = {"pipeline/train.py", "pipeline/evaluate.py", "scripts/train.py", "src/train_model.py"}


def trace(path, exc, code="x = 1"):
    return ["Traceback (most recent call last):", f'  File "/app/{path}", line 5, in f', f"    {code}", exc]


# --- the rule -------------------------------------------------------------------------------------

def test_the_section_8_failure_is_slot_caused():
    o = classify("train", GIT_WARNING + ARGV_TRACE + ADAPTER_TRACE, FILES)
    assert (o.kind, o.group, o.frame) == ("slot", "script_args", "scripts/train.py line 10")
    assert o.code.startswith("alpha = float(sys.argv[2])")
    assert [exc for _, exc in tracebacks(ARGV_TRACE + ADAPTER_TRACE)][0].startswith("ValueError")


def test_slot_groups():
    cases = {
        "columns": trace("pipeline/evaluate.py", "KeyError: 'trip_duraton'"),
        "function_args": trace("pipeline/train.py", "TypeError: train() got an unexpected keyword argument 'y'"),
        "data_files": trace("pipeline/train.py", "FileNotFoundError: [Errno 2] No such file: 'data/x.csv'"),
        "function": trace("pipeline/train.py", "AttributeError: module 'src.train_model' has no attribute 'fit'"),
        "model_output": trace("pipeline/train.py", "OSError: No such file or directory: '/tmp/model-x/model.pkl'"),
    }
    for group, tail in cases.items():
        o = classify("train", tail, FILES)
        assert (o.kind, o.group) == ("slot", group), (group, o)
    usage = ["usage: train.py [-h] --data DATA --out OUT", "train.py: error: the following arguments are required: --out"]
    assert classify("train", usage, FILES).group == "script_args"


def test_environment_and_unknown_go_to_the_fix_loop():
    env = {
        "build": ("build", ["ERROR: No matching distribution found for xgboost==3.0.0"]),
        "module": ("train", trace("pipeline/train.py", "ModuleNotFoundError: No module named 'xgboost'")),
        "shared lib": ("train", trace("src/train_model.py", "OSError: libffi.so.8: cannot open shared object file")),
        "mlflow server": ("train", trace("pipeline/train.py", "mlflow.exceptions.RestException: API request to "
                                                              "http://mlflow:5000/api/2.0/mlflow/logged-models failed")),
        "outside our files": ("train", trace("usr/lib/python3.11/site-packages/xgboost/core.py", "KeyError: 'x'")),
        "unknown error": ("train", trace("pipeline/train.py", "ZeroDivisionError: division by zero")),
        "git warning only": ("train", GIT_WARNING),
    }
    for name, (stage, tail) in env.items():
        assert classify(stage, tail, FILES).kind == "environment", name


def test_changed_slots_and_the_outside_group_flag():
    before = BASE | {"train_script": MLFLOW_RUN | {"args": ["$train_path", "$target_column"]}}
    after = BASE | {"train_script": MLFLOW_RUN}
    assert changed_slots(before, after) == ["train_script.args"]
    assert outside_group(["train_script.args"], "script_args") == []
    assert outside_group(["train_script.args", "evaluate.group_column"], "script_args") == ["evaluate.group_column"]


def test_the_prompt_is_generic_and_loads_with_only_a_user_section():
    p = load_prompt("slot_runtime_retry_v1", required=("user",))
    assert set(p.sections) == {"user"}
    assert "argv" not in p.sections["user"] and "script" not in p.sections["user"]       # no case-specific hint


# --- the create command ---------------------------------------------------------------------------

class ArgvRunner(FakeRunner):
    """Fake docker: `make train` fails like section 8 while the adapter passes the target column as argv[2]."""

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        if "train" in cmd and "logs" not in cmd:
            if 'ARGS = ["$train_path", "$target_column"]' in (Path(cwd) / "pipeline" / "train.py").read_text(encoding="utf-8"):
                return 2, "\n".join(GIT_WARNING + ARGV_TRACE + ADAPTER_TRACE)
        return 0, "ok"


def git_script_repo(tmp_path):
    root = script_repo(tmp_path / "repo", MLFLOW_SCRIPT)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_create_corrects_a_slot_answer_from_the_runtime_error(tmp_path):
    root = git_script_repo(tmp_path)
    bad = BASE | {"train_script": MLFLOW_RUN | {"args": ["$train_path", "$target_column"]}}
    slots = tmp_path / "bad.json"
    slots.write_text(json.dumps(bad), encoding="utf-8")
    model = FakeModel(BASE | {"train_script": MLFLOW_RUN})                    # the corrected answer
    res = create(root, model, slots_file=slots, runner=ArgvRunner(), sandbox_log=None, fix_log=None, call_log=None,
                 report_dir=None, retriever=lambda s, t: [], retriever_name="none")
    assert res.status == "written", res.text()
    [r] = res.runtime_retries
    assert r["origin"]["group"] == "script_args" and r["changed"] == ["train_script.args"] and r["outside_group"] == []
    assert r["sandbox_after"] == "passed" and res.slots["train_script"]["args"] == ["$train_path", "0.5"]
    sent = model.sent[0][-1]["content"]                                       # the runtime retry message
    assert "failed at the train stage" in sent and "could not convert string to float" in sent
    assert "scripts/train.py line 10" in sent and "train_script.args" in sent
    assert model.sent[0][-2]["role"] == "assistant"                           # same conversation, previous answer
    assert "Runtime slot retries:" in git(root, "log", "-1", "--format=%B", res.branch)


def test_a_retry_that_changes_other_slots_is_flagged(tmp_path):
    root = git_script_repo(tmp_path)
    bad = BASE | {"train_script": MLFLOW_RUN | {"args": ["$train_path", "$target_column"]}}
    slots = tmp_path / "bad.json"
    slots.write_text(json.dumps(bad), encoding="utf-8")
    other = BASE | {"evaluate": {"eval_file": "data/processed/test.csv", "group_column": "a"},
                    "train_script": MLFLOW_RUN}
    res = create(root, FakeModel(other), slots_file=slots, runner=ArgvRunner(), sandbox_log=None, fix_log=None,
                 call_log=None, report_dir=None, retriever=lambda s, t: [], retriever_name="none", dry_run=True)
    [r] = res.runtime_retries
    assert r["outside_group"] == ["evaluate.group_column"] and "CHANGED OUTSIDE THE BLAMED GROUP" in res.text()


def test_an_environment_failure_skips_the_runtime_retry(tmp_path):
    root = git_script_repo(tmp_path)
    slots = tmp_path / "ok.json"
    slots.write_text(json.dumps(BASE | {"train_script": MLFLOW_RUN}), encoding="utf-8")
    runner = FakeRunner(fail_on="train", output="ok")
    give_up = {"fix": {"action": "give_up", "reason": "x"}, "reason": "x"}
    res = create(root, FakeModel(give_up), slots_file=slots, runner=runner, sandbox_log=None, fix_log=None,
                 call_log=None, report_dir=None, retriever=lambda s, t: [], retriever_name="none")
    assert res.runtime_retries == [] and res.status == "failed"              # straight to the fix loop
