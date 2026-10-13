"""Create-mode validator checks (validate_slots strict=True): only the create command uses them; the sandbox,
the fix loop and the evaluations keep the plain validator."""

import shutil

from builder_agent.render import validate_slots
from builder_agent.scan import scan_repo

from test_render import MINI_GOLD, mini  # noqa: F401 (fixture)
from test_scan import write

VAL_ROWS = "a,b,flag,price\n7,1.5,True,70\n8,2.5,False,80\n"


def repo(mini, tmp_path, files):  # noqa: F811
    root = tmp_path / "r"
    shutil.copytree(mini.root, root)
    write(root, files)
    return scan_repo(root)


# --- step 1: evaluate on the test file ---------------------------------------------------------------

def test_a_validation_file_is_rejected_when_a_test_file_exists(mini, tmp_path):  # noqa: F811
    ctx = repo(mini, tmp_path, {"data/processed/validation.csv": VAL_ROWS})
    answer = MINI_GOLD | {"evaluate": {"eval_file": "data/processed/validation.csv", "group_column": "flag"}}
    assert validate_slots(ctx, answer).ok                                     # the plain validator: unchanged
    v = validate_slots(ctx, answer, strict=True)
    assert not v.ok and "evaluate.eval_file 'data/processed/validation.csv' is not a test file" in v.reasons[0]
    assert "data/processed/test.csv" in v.reasons[0]
    assert validate_slots(ctx, MINI_GOLD, strict=True).ok                     # test.csv: accepted


def test_no_test_file_with_the_target_means_no_rule(mini, tmp_path):  # noqa: F811
    ctx = repo(mini, tmp_path, {"data/processed/validation.csv": VAL_ROWS,
                                "data/processed/test.csv": "a,b,flag\n3,0.5,False\n"})   # a test file without labels
    answer = MINI_GOLD | {"evaluate": {"eval_file": "data/processed/validation.csv", "group_column": "flag"}}
    assert validate_slots(ctx, answer, strict=True).ok


def test_create_gets_the_slot_llm_to_switch_to_the_test_file(mini, tmp_path):  # noqa: F811
    from test_create import run as create_run
    from test_llm import FakeModel
    from test_repo_rule import git
    root = tmp_path / "target"
    shutil.copytree(mini.root, root)
    write(root, {"data/processed/validation.csv": VAL_ROWS})
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    first = MINI_GOLD | {"task": "regression",
                         "evaluate": {"eval_file": "data/processed/validation.csv", "group_column": "flag"}}
    model = FakeModel(first, MINI_GOLD | {"task": "regression"})
    res = create_run(root, tmp_path, model=model, dry_run=True)
    assert res.status == "dry-run", res.text()
    assert not res.slot_attempts[0]["valid"] and "is not a test file" in res.slot_attempts[0]["reasons"][0]
    assert res.slot_attempts[1]["valid"] and res.slots["evaluate"]["eval_file"] == "data/processed/test.csv"
