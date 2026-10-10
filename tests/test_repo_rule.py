"""Hard rule: the Builder never modifies an existing file in a target repo (CLAUDE.md).

New files only, on its own branch builder/<date>, never on main. These tests fail if any
Builder entry point changes a file that existed before it ran.
"""

import hashlib
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

from builder_agent.decide import plan_build
from builder_agent.render import RenderError, render_adapters
from builder_agent.render.configs import render_configs
from builder_agent.repo_writer import RepoWriter, WriteRefused, current_branch
from builder_agent.sandbox import run_sandbox
from builder_agent.scan import scan_repo
from builder_agent.verify import verify_repo

from test_render import MINI_GOLD, contracts, mini  # noqa: F401 (fixtures)
from test_sandbox import FakeRunner
from test_scan import write

TODAY = date(2026, 10, 8)
PACKAGE = Path(__file__).resolve().parents[1] / "builder_agent"


def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def git_repo(root: Path, files: dict[str, str]) -> Path:
    """A git repo whose files hold exactly these bytes (no Windows newline translation, no autocrlf)."""
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "teammates' work")
    return root


def snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}


def assert_untouched(before: dict[str, str], root: Path):
    after = snapshot(root)
    changed = [f for f in before if after.get(f) != before[f]]
    assert not changed, f"the Builder changed existing files: {changed}"


# --- the writer ------------------------------------------------------------------------

def test_writer_refuses_existing_files_and_writes_nothing(tmp_path):
    write(tmp_path, {"Dockerfile": "FROM teammate\n"})
    w = RepoWriter(tmp_path)
    with pytest.raises(WriteRefused, match="refusing to overwrite Dockerfile: it already exists"):
        w.write_new({"Makefile": "x\n", "Dockerfile": "FROM builder\n"})
    assert (tmp_path / "Dockerfile").read_text() == "FROM teammate\n"
    assert not (tmp_path / "Makefile").exists()                 # all or nothing


def test_writer_may_update_only_what_it_created_in_this_run(tmp_path):
    w = RepoWriter(tmp_path)
    w.write_new({"pipeline/a.txt": "1\n"})
    w.write_new({"pipeline/a.txt": "2\n"})                      # same run: allowed
    with pytest.raises(WriteRefused):
        RepoWriter(tmp_path).write_new({"pipeline/a.txt": "3\n"})   # next run: it exists now
    assert (tmp_path / "pipeline/a.txt").read_text() == "2\n"


@pytest.mark.parametrize("rel", ["../outside.txt", ".git/config"])
def test_writer_stays_inside_the_repo(tmp_path, rel):
    with pytest.raises(WriteRefused):
        RepoWriter(tmp_path).write_new({rel: "x"})


def test_writer_works_on_its_own_branch_never_main(tmp_path):
    repo = git_repo(tmp_path / "r", {"README.md": "hi\n"})
    main_head = git(repo, "rev-parse", "main")
    w = RepoWriter(repo, today=TODAY)
    assert w.branch == current_branch(repo) == "builder/2026-10-08"
    w.write_new({"agents.yaml": "x: 1\n"})
    assert git(repo, "rev-parse", "main") == main_head          # main untouched
    git(repo, "switch", "-q", "main")
    assert RepoWriter(repo, today=TODAY).branch == "builder/2026-10-08-2"   # never reuses an existing branch


def test_writer_refuses_when_someone_switched_branch(tmp_path):
    repo = git_repo(tmp_path / "r", {"README.md": "hi\n"})
    w = RepoWriter(repo, today=TODAY)
    git(repo, "switch", "-q", "main")
    with pytest.raises(WriteRefused, match="not the Builder's branch"):
        w.write_new({"agents.yaml": "x: 1\n"})


# --- every Builder entry point on a real git repo ------------------------------------------

@pytest.fixture
def target(mini, tmp_path):  # noqa: F811
    """The mini repo as a git repo with teammates' pipeline files already in it."""
    files = {p.relative_to(mini.root).as_posix(): p.read_text(encoding="utf-8")
             for p in Path(mini.root).rglob("*") if p.is_file()}
    files |= {"Dockerfile": "FROM python:3.11\n", "Makefile": "all:\n\techo teammate\n",
              "pipeline/serve.py": "# teammate's service\n", "docker-compose.yaml": "services:\n  web:\n"
              "    image: postgres:latest\n    ports: ['8000:8000']\n", "run.sh": "#!/bin/bash\r\necho hi\r\n"}
    return git_repo(tmp_path / "target", files)


def test_no_builder_run_changes_an_existing_file(target, contracts, tmp_path):  # noqa: F811
    before = snapshot(target)
    main_head = git(target, "rev-parse", "main")
    ctx = scan_repo(target)
    plan = plan_build(ctx, contracts)

    # adapters and configs collide with teammates' files: refused, nothing written
    writer = RepoWriter(target, today=TODAY)
    with pytest.raises(RenderError, match="pipeline/serve.py: it already exists"):
        render_adapters(ctx, contracts, MINI_GOLD, target, writer=writer)
    with pytest.raises(RenderError, match="already exists"):
        render_configs(ctx, contracts, plan, MINI_GOLD, target, writer=writer)

    # the sandbox works on a temp copy; the real repo is only read
    run_sandbox(target, MINI_GOLD, log_path=None, runner=FakeRunner())

    # verify reads a commit and writes its report elsewhere
    report = verify_repo(target, ref="HEAD", out_dir=tmp_path / "report")
    assert report.findings and (tmp_path / "report" / "fixes.patch").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="outside the target repo"):
        verify_repo(target, ref="HEAD", out_dir=target / "builder-report")

    assert_untouched(before, target)
    assert git(target, "rev-parse", "main") == main_head
    assert current_branch(target) == "builder/2026-10-08"
    assert git(target, "status", "--porcelain") == ""             # nothing new either: all refused


def test_new_files_land_on_the_builder_branch_only(mini, contracts, tmp_path):  # noqa: F811
    files = {p.relative_to(mini.root).as_posix(): p.read_text(encoding="utf-8")
             for p in Path(mini.root).rglob("*") if p.is_file()}
    repo = git_repo(tmp_path / "clean", files)
    before = snapshot(repo)
    ctx = scan_repo(repo)
    writer = RepoWriter(repo, today=TODAY)
    render_adapters(ctx, contracts, MINI_GOLD, repo, writer=writer)
    render_configs(ctx, contracts, plan_build(ctx, contracts), MINI_GOLD, repo, writer=writer)
    assert_untouched(before, repo)
    status = git(repo, "status", "--porcelain").splitlines()
    assert status and all(line.startswith("??") for line in status)   # only new, untracked files
    assert current_branch(repo) == "builder/2026-10-08"
    assert git(repo, "log", "--oneline", "main..HEAD") == ""       # the Builder doesn't commit


# --- only repo_writer.py writes into target repos ------------------------------------------------

WRITE_CALL = re.compile(r"\.write_text\(|\.write_bytes\(|\bopen\([^)]*[\"'][wax]b?\+?[\"']|\.open\([\"'][wax]"
                        r"|shutil\.(copy|copytree|move)\(|\.extractall\(")
ALLOWED_WRITERS = {
    "repo_writer.py": "the only writer into target repos (new files, builder branch)",
    "render/actionlint.py": "downloads the pinned actionlint binary into builder-agent's .tools/",
    "llm/__init__.py": "LLM call log in builder-agent's logs/",
    "fix/__init__.py": "fix-attempt log in builder-agent's logs/",
    "llm/eval.py": "eval reports in builder-agent's experiments/",
    "sandbox/__init__.py": "copies the target to a temp folder (read-only on the original); attempts log",
    "scan/rows.py": "row-sample cache in builder-agent's .cache/",
    "verify/__init__.py": "report outside the target (checked); temp export to check the patch",
    "finetune/__init__.py": "fine-tuning data in builder-agent's experiments/finetune/",
    "faults/__init__.py": "generated fault cases in builder-agent's tests/faults/generated/",
    "memory/learned.py": "fixes written back from real runs in builder-agent's memory/learned/",
}


def test_only_known_modules_write_files():
    writers = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*.py")
               if WRITE_CALL.search(p.read_text(encoding="utf-8"))}
    assert writers == set(ALLOWED_WRITERS), (
        f"new file writers: {sorted(writers - set(ALLOWED_WRITERS))}; "
        f"no longer writing: {sorted(set(ALLOWED_WRITERS) - writers)}. "
        "Target-repo writes must go through repo_writer.RepoWriter.")


def test_the_tests_never_write_to_the_real_logs():
    """tests/conftest.py sends every attempt and call log to a temp folder (test runs once appended
    545 fake-model lines to logs/; they are in logs/_test_pollution/)."""
    import os

    from builder_agent.fix import DEFAULT_FIX_LOG
    from builder_agent.llm import DEFAULT_CALL_LOG
    from builder_agent.sandbox import DEFAULT_LOG
    real = Path(__file__).resolve().parents[1] / "logs"
    for log in (DEFAULT_FIX_LOG, DEFAULT_CALL_LOG, DEFAULT_LOG):
        assert log.parent == Path(os.environ["BUILDER_LOG_DIR"]) and real not in log.parents, log
