"""Script-training mode: scanner facts, the mode rule, the slot validator, rendering (step 3b)."""

import ast

import pytest

from builder_agent.decide import load_contracts, plan_build
from builder_agent.decide.rules import choose_train_mode
from builder_agent.llm.prompts import load_prompt, script_facts, slot_variables, slots_model
from builder_agent.render import lint_files, render_adapters, validate_slots
from builder_agent.render.configs import render_configs
from builder_agent.render.script_slots import ScriptSlotAnswers
from builder_agent.scan import scan_repo
from builder_agent.scan.scripts import argv_positions, cli_args, model_outputs, returns_value, trains

from test_render import CONTRACTS
from test_scan import write

ARGPARSE_SCRIPT = '''import argparse
import joblib
import pandas as pd
from sklearn.linear_model import Ridge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="training CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    joblib.dump(Ridge(alpha=args.alpha).fit(df.drop(columns=["price"]), df["price"]), args.out)


if __name__ == "__main__":
    main()
'''
MLFLOW_SCRIPT = '''import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    with mlflow.start_run():
        mlflow.sklearn.log_model(Ridge(alpha=alpha).fit(df.drop(columns=["price"]), df["price"]), "ridge")
'''


def script_repo(root, script=ARGPARSE_SCRIPT):
    rows = "".join(f"{i % 7},{i * 0.37:.2f},{i % 3},{10 + i * 1.3:.2f}\n" for i in range(60))
    write(root, {"requirements.txt": "scikit-learn\npandas\njoblib\n", ".python-version": "3.11\n",
                 "data/processed/train.csv": "a,b,g,price\n" + rows,
                 "data/processed/test.csv": "a,b,g,price\n" + rows.replace(",1", ",2"),
                 "scripts/train.py": script})
    return root


BASE = {"task": "regression", "target_column": "price", "target_transform": "none", "model_flavor": "sklearn",
        "model_input": "dataframe", "evaluate": {"eval_file": "data/processed/test.csv", "group_column": "g"},
        "data": {"data_step": "existing"}}
FILE_RUN = {"script": "scripts/train.py", "args": ["--data", "$train_path", "--out", "$model_dir/model.joblib"],
            "train_file": "data/processed/train.csv", "model_output": "file", "mlflow_artifact_path": None,
            "model_file": "$model_dir/model.joblib"}
MLFLOW_RUN = {"script": "scripts/train.py", "args": ["$train_path", "0.5"], "train_file": "data/processed/train.csv",
              "model_output": "mlflow", "mlflow_artifact_path": "ridge", "model_file": None}


def test_scanner_facts():
    tree = ast.parse(ARGPARSE_SCRIPT)
    args = {a.flags[0]: a for a in cli_args(tree)}
    assert args["--data"].required and args["--out"].required and not args["--alpha"].required
    assert args["--alpha"].default == "1.0" and args["--data"].help == "training CSV"
    assert [(o.kind, o.target) for o in model_outputs(tree)] == [("joblib", "args.out")]
    mtree = ast.parse(MLFLOW_SCRIPT)
    assert argv_positions(mtree) == [1, 2]
    assert [(o.kind, o.target) for o in model_outputs(mtree)] == [("mlflow.sklearn", "'ridge'")]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assert trains(main) and not returns_value(main)          # trains, but hands nothing back: not a function mode candidate
    fn = ast.parse("def fit(X, y):\n    return Ridge().fit(X, y)\n").body[0]
    assert trains(fn) and returns_value(fn)


def test_mode_rule(tmp_path):
    choice, _ = choose_train_mode(scan_repo(script_repo(tmp_path / "s")))
    assert (choice.status, choice.mode) == ("decided", "script") and choice.candidates[0].startswith("scripts/train.py")
    both = script_repo(tmp_path / "f")
    write(both, {"src/train_model.py": "def fit(X, y):\n    return Ridge().fit(X, y)\n"})
    choice, _ = choose_train_mode(scan_repo(both))
    assert choice.mode == "function"                        # a function that returns the model wins
    write(tmp_path / "n", {"src/x.py": "def f():\n    return 1\n"})
    choice, need = choose_train_mode(scan_repo(tmp_path / "n"))
    assert choice.status == "needs_llm" and need.item == "train_mode"


def test_validator(tmp_path):
    ctx = scan_repo(script_repo(tmp_path))
    assert validate_slots(ctx, BASE | {"train_script": FILE_RUN}).ok
    reasons = lambda run: validate_slots(ctx, BASE | {"train_script": FILE_RUN | run}).reasons  # noqa: E731
    assert any("requires --out" in r for r in reasons({"args": ["--data", "$train_path"]}))
    assert any("unknown token '$model'" in r for r in reasons({"args": ["--data", "$train_path", "--out", "$model"]}))
    assert any("is a data file path" in r for r in reasons({"args": ["--data", "data/processed/train.csv",
                                                                     "--out", "$model_dir/m.joblib"]}))
    assert any("model_file is required" in r for r in reasons({"model_file": None}))
    assert any("not a training script" in r for r in reasons({"script": "scripts/other.py"}))


def test_validator_mlflow_output(tmp_path):
    ctx = scan_repo(script_repo(tmp_path, MLFLOW_SCRIPT))
    assert validate_slots(ctx, BASE | {"train_script": MLFLOW_RUN}).ok
    r = validate_slots(ctx, BASE | {"train_script": MLFLOW_RUN | {"mlflow_artifact_path": "model"}})
    assert any("logs the model under 'ridge'" in x for x in r.reasons)


@pytest.mark.parametrize("script,run", [(ARGPARSE_SCRIPT, FILE_RUN), (MLFLOW_SCRIPT, MLFLOW_RUN)])
def test_render_script_mode(tmp_path, script, run):
    ctx = scan_repo(script_repo(tmp_path / "repo", script))
    c = load_contracts(CONTRACTS)
    out = tmp_path / "out"
    answer = BASE | {"train_script": run}
    result = render_adapters(ctx, c, answer, out)
    assert result.ok, result.lint
    train = (out / "pipeline" / "train.py").read_text(encoding="utf-8")
    assert "MLFLOW_RUN_ID=run.info.run_id" in train and 'SCRIPT = "scripts/train.py"' in train
    assert ("load_model_file" in train) == (run["model_output"] == "file")
    assert not lint_files(out, ["pipeline/train.py"])
    render_configs(ctx, c, plan_build(ctx, c), answer, out / "cfg")
    makefile = (out / "cfg" / "Makefile").read_text(encoding="utf-8")
    assert "$(PYTHON) pipeline/train.py" in makefile and "python scripts/train.py" not in makefile


def test_prompt_uses_the_script_schema_and_shows_the_code(tmp_path):
    ctx = scan_repo(script_repo(tmp_path))
    prompt = load_prompt("train_script_v1")
    assert slots_model(prompt) is ScriptSlotAnswers
    v = slot_variables(ctx, plan_build(ctx, load_contracts(CONTRACTS)))
    user = prompt.render("user", v)
    assert "option --data: required" in script_facts(ctx) and "joblib.dump" in user and "$model_dir" in user


def test_plan_says_script_mode_in_its_needs_llm_text(tmp_path):
    for script in (ARGPARSE_SCRIPT, MLFLOW_SCRIPT):
        ctx = scan_repo(script_repo(tmp_path / str(len(script)), script))
        plan = plan_build(ctx, load_contracts(CONTRACTS))
        item = next(n for n in plan.needs_llm if n.item == "target:train")
        assert item.reason.startswith("script mode: scripts/train.py trains the model")
        assert plan.target("train").status == "needs_llm" and plan.target("train").command is None
