"""Create mode: the whole Builder on a repo, files committed on builder/<date> only if the sandbox passes.
Fake LLM and fake Docker; every repo is a throwaway git repo built from the mini fixture."""

import json
import shutil
from pathlib import Path

import pytest

from builder_agent.create import create
from builder_agent.repo_writer import RepoWriter, current_branch

from test_llm import FakeModel
from test_render import MINI_GOLD, mini  # noqa: F401 (fixtures)
from test_repo_rule import git, snapshot
from test_sandbox import FakeRunner

GIVE_UP = {"fix": {"action": "give_up", "reason": "nothing on the menu fits"}, "reason": "x"}


@pytest.fixture
def repo(mini, tmp_path):  # noqa: F811
    root = tmp_path / "target"
    shutil.copytree(mini.root, root)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "teammates' work")
    return root


@pytest.fixture
def slots_file(tmp_path):
    path = tmp_path / "slots.json"                     # outside the target repo
    path.write_text(json.dumps(MINI_GOLD), encoding="utf-8")
    return path


def run(repo, tmp_path, model=None, slots=None, runner=None, **kw):
    return create(repo, model or FakeModel(), slots_file=slots, runner=runner or FakeRunner(), sandbox_log=None,
                  fix_log=None, call_log=None, report_dir=tmp_path / "reports", retriever=lambda stage, tail: [],
                  retriever_name="none", **kw)


def branches(repo) -> list[str]:
    return git(repo, "branch", "--format=%(refname:short)").split()


def test_written_on_a_builder_branch_and_back_on_main(repo, slots_file, tmp_path):
    before = snapshot(repo)
    res = run(repo, tmp_path, slots=slots_file)
    assert res.status == "written", res.text()
    assert current_branch(repo) == "main" and snapshot(repo) == before and res.worktree_unchanged
    assert res.branch.startswith("builder/") and res.branch in branches(repo)
    committed = git(repo, "show", "--name-only", "--format=", res.branch).split()
    assert sorted(committed) == sorted(res.files) and "Dockerfile" in committed and "pipeline/train.py" in committed
    log = git(repo, "log", "-1", "--format=%an <%ae>%n%B", res.branch)
    assert log.startswith("Builder agent <builder@localhost>")              # contracts.yaml integration
    for part in ("Files (", "- python: 3.11", "- task:", "- training mode: function", "Slot answers (from",
                 "Fixes applied:", "- none: the first sandbox run passed"):
        assert part in log, part
    assert git(repo, "rev-parse", f"{res.branch}~1") == git(repo, "rev-parse", "main")
    assert Path(res.report_path).exists()


def test_slots_from_the_llm_with_the_approved_prompt(repo, tmp_path):
    res = run(repo, tmp_path, model=FakeModel(MINI_GOLD | {"task": "regression"}))
    assert res.status == "written", res.text()
    assert res.slots_source == "llm" and res.slot_attempts == [{"attempt": 1, "valid": True, "reasons": []}]
    assert "slots_v3 x fake:1b, valid on try 1" in git(repo, "log", "-1", "--format=%B", res.branch)


def test_sandbox_failure_writes_nothing(repo, slots_file, tmp_path):
    before = snapshot(repo)
    res = run(repo, tmp_path, model=FakeModel(GIVE_UP), slots=slots_file, runner=FakeRunner(fail_on="train"))
    assert res.status == "failed" and "still fails at train" in res.reason
    assert branches(repo) == ["main"] and snapshot(repo) == before
    text = res.text()
    assert "give_up" in text and "still failing at train" in text and "Traceback: boom" in text


def test_dry_run_touches_nothing(repo, slots_file, tmp_path):
    before = snapshot(repo)
    res = run(repo, tmp_path, slots=slots_file, dry_run=True)
    assert res.status == "dry-run" and res.files and "Builder: pipeline files for" in res.commit_message
    assert branches(repo) == ["main"] and snapshot(repo) == before
    assert "would commit on a new builder/<date> branch" in res.text()


def test_existing_pipeline_files_mean_verify_mode(repo, slots_file, tmp_path):
    (repo / "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")
    git(repo, "add", "Makefile")
    git(repo, "commit", "-q", "-m", "a Makefile")
    res = run(repo, tmp_path, slots=slots_file)
    assert res.status == "refused" and "Makefile" in res.reason and "builder_agent.verify" in res.reason


def test_untracked_file_at_a_generated_path_is_refused(repo, slots_file, tmp_path):
    (repo / ".dockerignore").write_text("mine, not committed\n", encoding="utf-8")     # untracked
    before = snapshot(repo)
    res = run(repo, tmp_path, slots=slots_file)
    assert res.status == "refused" and "already exists" in res.reason and ".dockerignore" in res.reason
    assert branches(repo) == ["main"] and snapshot(repo) == before


def test_gitignored_generated_path_is_refused(repo, slots_file, tmp_path):
    (repo / ".gitignore").write_text("pipeline/\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore pipeline")
    res = run(repo, tmp_path, slots=slots_file)
    assert res.status == "refused" and ".gitignore excludes" in res.reason and branches(repo) == ["main"]


def test_uncommitted_tracked_changes_are_refused(repo, slots_file, tmp_path):
    (repo / "requirements.txt").write_text("changed\n", encoding="utf-8")
    res = run(repo, tmp_path, slots=slots_file)
    assert res.status == "refused" and "uncommitted changes" in res.reason


def test_a_crash_in_the_commit_still_restores_the_branch(repo, slots_file, tmp_path, monkeypatch):
    before = snapshot(repo)

    def boom(self, message, author):
        raise RuntimeError("disk full")
    monkeypatch.setattr(RepoWriter, "commit", boom)
    with pytest.raises(RuntimeError):
        run(repo, tmp_path, slots=slots_file)
    assert current_branch(repo) == "main" and branches(repo) == ["main"] and snapshot(repo) == before
    assert git(repo, "status", "--porcelain") == ""
