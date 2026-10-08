"""Mode per artifact, and the static verify rules (one per fault case 002-006)."""

import json
import subprocess
from pathlib import Path

import pytest

from builder_agent.decide.modes import select_modes
from builder_agent.scan import scan_repo
from builder_agent.verify import RepoSource, check_patch, file_diff, verify_repo
from builder_agent.verify.rules import RULES, split_image

from test_render import TAXI, contracts  # noqa: F401 (fixture)
from test_repo_rule import git, git_repo, snapshot
from test_scan import write

MAIN = Path("C:/dev/mlops-zoomcamp-project")

COMPOSE = """\
x-common: &common
  healthcheck:
    test: ["CMD", "true"]
services:
  localstack:
    image: localstack/localstack:stable
    ports:
      - "4566:4566"
    healthcheck:
      test: ["CMD", "awslocal", "s3", "ls"]
  db:
    image: postgres:latest
    volumes:
      - db-data:/var/lib/postgresql/data
  newdb:
    image: postgres:18
    volumes:
      - type: volume
        source: d2
        target: /var/lib/postgresql/data
  okdb:
    image: postgres:16
    volumes:
      - d3:/var/lib/postgresql/data
  mlflow:
    build:
      context: ./mlflow
    ports:
      - ${MLFLOW_PORT}:5000
    environment:
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/tmp/artifacts
  api:
    build:
      context: ./api
    ports:
      - "8000:8000"
  worker:
    <<: *common
    image: redis:7.2
    ports:
      - "6379:6379"
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
"""
MLFLOW_DOCKERFILE = """\
FROM ubuntu:22.04
RUN apt-get install -y python3 python3-pip
CMD ["sh", "-c", "mlflow server --backend-store-uri $DB --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT --port 5000"]
"""
API_DOCKERFILE = "FROM python:3.11-slim AS base\nRUN apt-get install -y curl\nFROM base\nCMD [\"gunicorn\", \"app:app\"]\n"


@pytest.fixture
def repo(tmp_path):
    return git_repo(tmp_path / "proj", {
        "docker-compose.yaml": COMPOSE, "mlflow/Dockerfile": MLFLOW_DOCKERFILE, "api/Dockerfile": API_DOCKERFILE,
        "init.sh": "#!/bin/bash\necho ok\n", "win.sh": "#!/bin/bash\r\necho hi\r\n",
    })


def by_rule(report):
    out = {}
    for f in report.findings:
        out.setdefault(f.rule, []).append(f)
    return out


def test_every_rule_maps_to_a_recorded_fault_case():
    faults = {json.loads(p.read_text(encoding="utf-8"))["id"]
              for p in (Path(__file__).parent / "faults").glob("*/case.json")}
    assert set(RULES.values()) == {"fault-002", "fault-003", "fault-004", "fault-005", "fault-006"} <= faults


@pytest.mark.parametrize("ref,expected", [
    ("postgres:latest", ("postgres", "latest", False)),
    ("localstack/localstack:4.12", ("localstack/localstack", "4.12", False)),
    ("grafana/grafana", ("grafana/grafana", None, False)),
    ("localhost:5000/img", ("localhost:5000/img", None, False)),
    ("img@sha256:abc", ("img", None, True)),
    ("${IMG:-apache/airflow:2.9.3}", ("apache/airflow", "2.9.3", False)),
    ("${IMG}", None),
])
def test_split_image(ref, expected):
    assert split_image(ref) == expected


def test_floating_tags_and_postgres_mounts(repo, tmp_path):
    r = by_rule(verify_repo(repo, ref="HEAD"))
    floating = {(f.file, f.line, f.severity) for f in r["floating-image-tag"]}
    assert floating == {("docker-compose.yaml", 6, "error"),       # localstack:stable, known pin
                        ("docker-compose.yaml", 12, "error"),      # postgres:latest, known pin
                        ("docker-compose.yaml", 43, "warning")}    # grafana/grafana, no known pin
    grafana = next(f for f in r["floating-image-tag"] if f.line == 43)
    assert grafana.fix_diff is None and "no diff is proposed" in grafana.fix
    pg = {(f.line, f.severity) for f in r["postgres18-mount-path"]}
    assert pg == {(12, "error"), (16, "error")}                     # latest and 18; 16 is fine
    assert all(f.fault_id == "fault-003" for f in r["postgres18-mount-path"])
    assert "-    image: postgres:latest\n+    image: postgres:16" in r["postgres18-mount-path"][0].fix_diff


def test_crlf_scripts_and_gitattributes(repo):
    r = by_rule(verify_repo(repo, ref=None))                        # working tree: win.sh has CRLF
    crlf = {(f.file, f.severity) for f in r["crlf-shell-script"]}
    assert crlf == {("win.sh", "error"), (".gitattributes", "warning")}
    attrs = next(f for f in r["crlf-shell-script"] if f.file == ".gitattributes")
    assert "new file mode 100644" in attrs.fix_diff and "+*.sh text eol=lf" in attrs.fix_diff


def test_gitattributes_rule_silences_the_warning(tmp_path):
    repo = git_repo(tmp_path / "p", {"run.sh": "echo\n", ".gitattributes": "*.sh text eol=lf\n"})
    assert not verify_repo(repo, ref="HEAD").findings


def test_mlflow_local_artifact_root(repo):
    [f] = by_rule(verify_repo(repo, ref="HEAD"))["mlflow-local-artifact-root"]
    assert (f.file, f.line, f.fault_id, f.severity) == ("mlflow/Dockerfile", 3, "fault-005", "error")
    assert "$MLFLOW_DEFAULT_ARTIFACT_ROOT = /tmp/artifacts, set by service mlflow" in f.message
    assert "--serve-artifacts --artifacts-destination $MLFLOW_DEFAULT_ARTIFACT_ROOT" in f.fix_diff


def test_remote_artifact_root_is_fine(tmp_path):
    repo = git_repo(tmp_path / "p", {"Dockerfile": "FROM python:3.11\nCMD mlflow server --default-artifact-root "
                                                   "s3://bucket/artifacts\n"})
    assert not verify_repo(repo, ref="HEAD").findings


def test_services_without_healthcheck(repo):
    found = {f.evidence.split(":")[0]: f for f in by_rule(verify_repo(repo, ref="HEAD"))["service-without-healthcheck"]}
    assert set(found) == {"mlflow", "api", "grafana"}               # localstack has one, worker inherits one
    assert "http://localhost:5000/health" in found["mlflow"].fix_diff and "python3" in found["mlflow"].fix_diff
    assert '"curl", "-f", "http://localhost:8000/"' in found["api"].fix_diff
    assert found["grafana"].fix_diff is None


def test_patch_applies_and_target_is_untouched(repo, tmp_path):
    before, head = snapshot(repo), git(repo, "rev-parse", "HEAD")
    report = verify_repo(repo, ref="HEAD", out_dir=tmp_path / "out")
    assert report.patch_check.startswith("applies cleanly")
    for name in ("findings.json", "findings.md", "fixes.patch"):
        assert (tmp_path / "out" / name).exists()
    data = json.loads((tmp_path / "out" / "findings.json").read_text(encoding="utf-8"))
    assert {"file", "line", "rule", "fault_id", "severity", "fix_diff"} <= set(data["findings"][0])
    assert snapshot(repo) == before and git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""


def test_verify_reads_the_commit_not_the_working_tree(repo):
    (repo / "docker-compose.yaml").write_text(COMPOSE.replace("localstack:stable", "localstack:4.12"), encoding="utf-8")
    at_head = [f for f in verify_repo(repo, ref="HEAD").findings if f.rule == "floating-image-tag"]
    assert any("localstack:stable" in f.evidence for f in at_head)   # the local edit is ignored


def test_broken_patch_is_reported(repo):
    src = RepoSource(repo, "HEAD")
    bad = "diff --git a/init.sh b/init.sh\n--- a/init.sh\n+++ b/init.sh\n@@ -1,1 +1,1 @@\n-nope\n+x\n"
    assert check_patch(src, bad).startswith("DOES NOT APPLY")
    assert file_diff(src, []) == ""


# --- modes -------------------------------------------------------------------------------

def test_modes_on_a_repo_with_pipeline_files(repo, contracts):  # noqa: F811
    modes = {m.artifact: m for m in select_modes(scan_repo(repo), contracts)}
    assert modes["Dockerfile"].mode == "verify" and modes["Dockerfile"].existing == ["api/Dockerfile", "mlflow/Dockerfile"]
    assert modes["docker-compose"].existing == ["docker-compose.yaml"]
    assert {a for a, m in modes.items() if m.mode == "create"} == {
        "Makefile", "CI workflow", "agents.yaml", "schema draft", "contract adapter"}


def test_adapter_path_comes_from_the_contract(tmp_path, contracts):  # noqa: F811
    write(tmp_path, {"pipeline/serve.py": "# adapter\n", ".github/workflows/ci.yml": "on: push\n"})
    modes = {m.artifact: m.mode for m in select_modes(scan_repo(tmp_path), contracts)}
    assert modes["contract adapter"] == "verify" and modes["CI workflow"] == "verify"


@pytest.mark.skipif(not TAXI.is_dir(), reason="../taxi-trip-regression not checked out")
def test_taxi_creates_everything(contracts):  # noqa: F811
    assert {m.mode for m in select_modes(scan_repo(TAXI), contracts)} == {"create"}   # docs/Makefile is Sphinx's


# --- the main project (held-out: read-only) ----------------------------------------------------

@pytest.mark.skipif(not (MAIN / ".git").exists(), reason="C:/dev/mlops-zoomcamp-project not cloned")
def test_main_project_at_its_commit(tmp_path):
    status = subprocess.run(["git", "-C", str(MAIN), "status", "--porcelain"], capture_output=True, text=True).stdout
    report = verify_repo(MAIN, ref="e157717", out_dir=tmp_path)
    got = {(f.rule, f.file, f.line) for f in report.findings}
    assert got == {
        ("floating-image-tag", "docker-compose.yaml", 263), ("floating-image-tag", "docker-compose.yaml", 309),
        ("floating-image-tag", "docker-compose.yaml", 359), ("floating-image-tag", "docker-compose.yaml", 394),
        ("floating-image-tag", "docker-compose.yaml", 409),
        ("postgres18-mount-path", "docker-compose.yaml", 309), ("postgres18-mount-path", "docker-compose.yaml", 359),
        ("postgres18-mount-path", "docker-compose.yaml", 394),
        ("mlflow-local-artifact-root", "mlflow/Dockerfile", 35),
        ("crlf-shell-script", ".gitattributes", 0),
        ("service-without-healthcheck", "docker-compose.yaml", 329),
        ("service-without-healthcheck", "docker-compose.yaml", 374),
        ("service-without-healthcheck", "docker-compose.yaml", 408),
    }
    assert report.patch_check.startswith("applies cleanly")
    assert subprocess.run(["git", "-C", str(MAIN), "status", "--porcelain"],
                          capture_output=True, text=True).stdout == status   # read-only
