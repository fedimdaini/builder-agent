"""Python version and port hints from non-dependency files (Dockerfile, CI, compose, ...)."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .layout import pyc_version_tags
from .models import PortHint, VersionHint

DOCKER_FROM_RE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(?:\S+/)?python:(\d+(?:\.\d+){0,2})", re.I | re.M)
CI_PY_RE = re.compile(r"python-version:\s*\[?\s*[\"']?(\d+\.\d+(?:\.\d+)?)")
RUNTIME_RE = re.compile(r"python-(\d+\.\d+(?:\.\d+)?)")
EXPOSE_RE = re.compile(r"^\s*EXPOSE\s+(.+)$", re.I | re.M)
BIND_RE = re.compile(r"(?:--bind|-b)[\s=,\"']+[\w.\-]*:(\d{2,5})")
PORT_FLAG_RE = re.compile(r"--port[\s=,\"']+(\d{2,5})")
COMPOSE_PORT_RE = re.compile(r"^\s*-\s*[\"']?(?:[\d.]+:)?(\d{2,5}):(\d{2,5})", re.M)


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def _is_dockerfile(name: str) -> bool:
    return name == "Dockerfile" or name.startswith("Dockerfile.") or name.endswith(".Dockerfile")


def version_hints(root: Path, files: dict[str, int], pyc_names: list[str]) -> list[VersionHint]:
    hints: list[VersionHint] = []
    for rel in files:
        name = PurePosixPath(rel).name
        if name == ".python-version":
            for line in _read(root, rel).split():
                hints.append(VersionHint(value=line.strip(), source=rel, kind="python-version-file"))
        elif name == "runtime.txt":
            if m := RUNTIME_RE.search(_read(root, rel)):
                hints.append(VersionHint(value=m.group(1), source=rel, kind="runtime.txt"))
        elif _is_dockerfile(name):
            for m in DOCKER_FROM_RE.finditer(_read(root, rel)):
                hints.append(VersionHint(value=m.group(1), source=rel, kind="dockerfile"))
        elif rel.startswith(".github/workflows/") or name == ".gitlab-ci.yml":
            for v in sorted(set(CI_PY_RE.findall(_read(root, rel)))):
                hints.append(VersionHint(value=v, source=rel, kind="ci"))
    for version, path in pyc_version_tags(pyc_names).items():
        hints.append(VersionHint(value=version, source=path, kind="pyc-cache"))
    return hints


def infra_port_hints(root: Path, files: dict[str, int]) -> list[PortHint]:
    out: list[PortHint] = []
    for rel in files:
        name = PurePosixPath(rel).name
        if _is_dockerfile(name):
            text = _read(root, rel)
            for m in EXPOSE_RE.finditer(text):
                for tok in m.group(1).split():
                    if (p := tok.split("/")[0]).isdigit():
                        out.append(PortHint(port=int(p), source=rel, kind="dockerfile-expose"))
            out += _flag_ports(text, rel)
        elif name.startswith(("docker-compose", "compose.")) and name.endswith((".yml", ".yaml")):
            for m in COMPOSE_PORT_RE.finditer(_read(root, rel)):
                out.append(PortHint(port=int(m.group(2)), source=rel, kind="compose"))
        elif name == "Procfile":
            out += [h.model_copy(update={"kind": "procfile"}) for h in _flag_ports(_read(root, rel), rel)]
    return out


def _flag_ports(text: str, rel: str) -> list[PortHint]:
    ports = [PortHint(port=int(p), source=rel, kind="bind-flag") for p in BIND_RE.findall(text)]
    ports += [PortHint(port=int(p), source=rel, kind="port-flag") for p in PORT_FLAG_RE.findall(text)]
    return ports
