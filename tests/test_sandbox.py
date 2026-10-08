"""Sandbox runner logic with a fake command runner (no Docker). The real run is done by hand."""

import hashlib
import json
from pathlib import Path

import yaml

from builder_agent.sandbox import STAGES, copy_repo, run_sandbox, write_override

from test_render import MINI_GOLD, mini  # noqa: F401 (fixture)
from test_scan import write


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


class FakeRunner:
    """Records docker commands; fails the one whose arguments contain `fail_on`."""

    def __init__(self, fail_on: str | None = None, output: str = "ok"):
        self.calls: list[list[str]] = []
        self.fail_on, self.output = fail_on, output

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in cmd and "logs" not in cmd:
            return 2, "\n".join(f"line {i}" for i in range(80)) + "\nTraceback: boom"
        return 0, self.output


def test_copy_skips_git_and_data_and_leaves_source_alone(tmp_path):
    src = write(tmp_path / "src", {"a.py": "x = 1\n", ".git/HEAD": "ref\n", "data/raw/big.csv": "a\n1\n",
                                   "src/data/loader.py": "y = 2\n", "pkg/__pycache__/m.pyc": "bin"})
    before = tree_hash(src)
    copy_repo(src, tmp_path / "dst", skip_top={"data"})
    copied = sorted(p.relative_to(tmp_path / "dst").as_posix() for p in (tmp_path / "dst").rglob("*") if p.is_file())
    assert copied == ["a.py", "src/data/loader.py"]       # nested src/data is code, kept
    assert tree_hash(src) == before


def test_override_mounts_real_data_read_only_and_drops_ports(tmp_path):
    write(tmp_path, {"compose.base.yml": yaml.safe_dump({"services": {
        "mlflow": {"image": "x", "ports": ["5000:5000"]},
        "serving-current": {"ports": ["8000:8000"],
                            "volumes": ["./data:/app/data:ro", "./models:/app/models"]}}})})
    real = tmp_path / "real-repo"
    text = write_override(tmp_path, "compose.base.yml", "serving-current", real).read_text(encoding="utf-8")
    assert "ports: !reset []" in text and "volumes: !override" in text
    doc = yaml.load(text.replace("!reset", "").replace("!override", ""), Loader=yaml.SafeLoader)
    vols = doc["services"]["serving-current"]["volumes"]
    assert vols[0] == {"type": "bind", "source": (real / "data").resolve().as_posix(),
                       "target": "/app/data", "read_only": True}
    assert vols[1] == "./models:/app/models"


def test_all_stages_in_order(mini, tmp_path):  # noqa: F811
    fake = FakeRunner()
    log = tmp_path / "attempts.jsonl"
    before = tree_hash(Path(mini.root))
    result = run_sandbox(mini.root, MINI_GOLD, expected={"prediction": 100, "rel_tolerance": 0.5},
                         log_path=log, runner=fake)
    assert result.ok, result.summary()
    assert [s.name for s in result.stages] == STAGES and all(s.status == "ok" for s in result.stages)
    assert tree_hash(Path(mini.root)) == before                     # real repo untouched

    joined = [" ".join(c) for c in fake.calls]
    assert any(c.endswith("build") and "--progress plain" in c for c in joined)
    for target in ("data", "train", "evaluate"):
        assert any(f"run --rm -T --no-deps serving-current make {target} SAMPLE=1" in c for c in joined)
    assert any("pipeline/smoke_test.py --url http://localhost:8000 --check predict" in c for c in joined)
    assert joined[-1].endswith("down -v --remove-orphans")
    assert all(c.startswith("docker compose -p builder-sbx-") and "-f compose.sandbox.yml" in c for c in joined)
    assert not Path(result.workdir).exists()                        # temp copy removed

    [line] = log.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["attempt_id"] == result.attempt_id


def test_stops_at_first_failure_and_keeps_the_tail(mini, tmp_path):  # noqa: F811
    fake = FakeRunner(fail_on="train")
    result = run_sandbox(mini.root, MINI_GOLD, log_path=tmp_path / "a.jsonl", runner=fake)
    assert not result.ok and result.failed_stage == "train"
    status = {s.name: s.status for s in result.stages}
    assert status == {"render": "ok", "build": "ok", "mlflow": "ok", "data": "ok", "train": "failed",
                      "evaluate": "skipped", "serve": "skipped", "health": "skipped", "predict": "skipped"}
    train = next(s for s in result.stages if s.name == "train")
    assert len(train.output_tail) == 50 and train.output_tail[-1] == "Traceback: boom"
    assert train.exit_code == 2 and "make train SAMPLE=1" in train.command
    assert not any("evaluate" in c for c in fake.calls)              # nothing ran after the failure
    assert fake.calls[-1][-3:] == ["down", "-v", "--remove-orphans"]
    assert "FAILED at train" in result.summary()


def test_health_failure_includes_service_logs(mini, tmp_path):  # noqa: F811
    fake = FakeRunner(fail_on="health")
    result = run_sandbox(mini.root, MINI_GOLD, log_path=None, runner=fake)
    health = next(s for s in result.stages if s.name == "health")
    assert health.status == "failed"
    assert any("[sandbox] logs of serving-current:" in line for line in health.output_tail)


def test_render_failure_stops_before_docker(mini, tmp_path):  # noqa: F811
    fake = FakeRunner()
    bad = dict(MINI_GOLD, target_column="duration")
    result = run_sandbox(mini.root, bad, log_path=None, runner=fake)
    assert result.failed_stage == "render" and fake.calls == []
    assert "invalid slot answers: target_column 'duration'" in result.stages[0].output_tail[0]
