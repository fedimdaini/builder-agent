"""Static verify rules, one per fault case (tests/faults/). They read files; they never run anything.

    floating-image-tag        fault-002  image/FROM with latest, stable or no tag
    postgres18-mount-path     fault-003  postgres floating or >= 18 with a volume at /var/lib/postgresql/data
    crlf-shell-script         fault-004  .sh with CRLF; or no .gitattributes keeping *.sh LF
    mlflow-local-artifact-root fault-005 mlflow server --default-artifact-root <plain local path>
    service-without-healthcheck fault-006 a compose service with ports and no healthcheck
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass

import yaml

from .models import FileFix, Finding, LineEdit
from .source import RepoSource

RULES = {
    "floating-image-tag": "fault-002",
    "postgres18-mount-path": "fault-003",
    "crlf-shell-script": "fault-004",
    "mlflow-local-artifact-root": "fault-005",
    "service-without-healthcheck": "fault-006",
}
FLOATING_TAGS = {"latest", "stable", "edge", "nightly", "main", "master", "dev"}
# versions proven to work in a fault case: image -> (pin, fault id)
KNOWN_PINS = {"localstack/localstack": ("4.12", "fault-002"), "postgres": ("16", "fault-003")}
POSTGRES_NAMES = {"postgres", "library/postgres", "docker.io/library/postgres"}
POSTGRES_OLD_DATA = "/var/lib/postgresql/data"
COMPOSE_PATTERNS = ["docker-compose*.yml", "docker-compose*.yaml", "compose.yml", "compose.yaml", "compose.*.yml"]


# --- helpers ------------------------------------------------------------------------

def compose_files(src: RepoSource) -> list[str]:
    return [f for f in src.files() if any(fnmatch.fnmatchcase(posixpath.basename(f), p) for p in COMPOSE_PATTERNS)]


def dockerfiles(src: RepoSource) -> list[str]:
    out = []
    for f in src.files():
        name = posixpath.basename(f)
        if name == "Dockerfile" or name.startswith("Dockerfile.") or name.lower().endswith(".dockerfile"):
            out.append(f)
    return out


def _line(node: yaml.Node) -> int:
    return node.start_mark.line + 1


def mapping_get(node: yaml.Node | None, key: str) -> yaml.Node | None:
    """Value node for key, following YAML merge keys (<<: *anchor)."""
    if not isinstance(node, yaml.MappingNode):
        return None
    merges = []
    for k, v in node.value:
        if k.value == key:
            return v
        if k.tag == "tag:yaml.org,2002:merge" or k.value == "<<":
            merges += v.value if isinstance(v, yaml.SequenceNode) else [v]
    for m in merges:
        found = mapping_get(m, key)
        if found is not None:
            return found
    return None


@dataclass
class Service:
    file: str
    name: str
    key: yaml.Node
    body: yaml.Node

    def get(self, key: str) -> yaml.Node | None:
        return mapping_get(self.body, key)

    def build_context(self) -> str | None:
        b = self.get("build")
        ctx = b.value if isinstance(b, yaml.ScalarNode) else (mapping_get(b, "context").value if mapping_get(b, "context") else None)
        if ctx is None:
            return None
        return posixpath.normpath(posixpath.join(posixpath.dirname(self.file), ctx))

    def build_dockerfile(self) -> str | None:
        ctx = self.build_context()
        if ctx is None:
            return None
        b = self.get("build")
        df = mapping_get(b, "dockerfile")
        return posixpath.normpath(posixpath.join(ctx, df.value if df is not None else "Dockerfile"))

    def env(self) -> dict[str, str]:
        e = self.get("environment")
        out = {}
        if isinstance(e, yaml.SequenceNode):
            for item in e.value:
                k, _, v = str(item.value).partition("=")
                out[k.strip()] = v.strip()
        elif isinstance(e, yaml.MappingNode):
            out = {k.value: str(v.value) for k, v in e.value}
        return out


def services(src: RepoSource, path: str) -> list[Service]:
    try:
        root = yaml.compose(src.read_text(path))
    except yaml.YAMLError:
        return []
    svcs = mapping_get(root, "services")
    if not isinstance(svcs, yaml.MappingNode):
        return []
    return [Service(path, k.value, k, v) for k, v in svcs.value if isinstance(v, yaml.MappingNode)]


def split_image(ref: str) -> tuple[str, str | None, bool] | None:
    """'postgres:latest' -> ('postgres', 'latest', False). None when a variable can't be resolved."""
    ref = ref.strip().strip("'\"")
    m = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?-([^}]*)\}", ref)
    if m:
        ref = m.group(1)
    if "$" in ref:
        return None
    digest = "@" in ref
    name, tag = ref.split("@")[0], None
    if ":" in name.rsplit("/", 1)[-1]:
        name, tag = name.rsplit(":", 1)
    return name, tag, digest


def is_floating(tag: str | None, digest: bool) -> bool:
    return not digest and (tag is None or tag.lower() in FLOATING_TAGS)


def _repin(line: str, name: str, tag: str | None, pin: str) -> str:
    old = f"{name}:{tag}" if tag else name
    return re.sub(re.escape(old) + r"(?![\w.:/-])", f"{name}:{pin}", line, count=1)


def _lines(src: RepoSource, path: str) -> list[str]:
    return src.read_text(path).splitlines()


# --- fault-002 / fault-003: images ----------------------------------------------------------

def _image_refs(src: RepoSource) -> list[tuple[str, int, str, Service | None]]:
    """(file, line, image ref, compose service or None) for compose image: and Dockerfile FROM."""
    out = []
    for f in compose_files(src):
        for s in services(src, f):
            img = s.get("image")
            if isinstance(img, yaml.ScalarNode):
                out.append((f, _line(img), img.value, s))
    for f in dockerfiles(src):
        stages = set()
        for i, line in enumerate(_lines(src, f), 1):
            m = re.match(r"\s*FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?", line, re.I)
            if m:
                if m.group(1).lower() != "scratch" and m.group(1) not in stages:
                    out.append((f, i, m.group(1), None))
                if m.group(2):
                    stages.add(m.group(2))
    return out


def rule_images(src: RepoSource) -> list[Finding]:
    findings = []
    for f, line_no, ref, svc in _image_refs(src):
        parsed = split_image(ref)
        if parsed is None:
            continue
        name, tag, digest = parsed
        line = _lines(src, f)[line_no - 1]
        pin = KNOWN_PINS.get(name)
        fix_edit = [FileFix(path=f, edits=[LineEdit(line=line_no, replace=[_repin(line, name, tag, pin[0])])])] \
            if pin else []
        if is_floating(tag, digest):
            findings.append(Finding(
                rule="floating-image-tag", fault_id="fault-002", severity="error" if pin else "warning",
                file=f, line=line_no, evidence=line.strip(),
                message=f"{ref} has a floating tag ({tag or 'no tag'}): it changes under you "
                        "(fault-002: localstack:stable started requiring an account)",
                fix=(f"pin it to {name}:{pin[0]} (known to work, {pin[1]})" if pin else
                     f"pin {name} to the explicit version you run today; no known-good version is recorded, "
                     "so no diff is proposed"),
                fixes=fix_edit))
        if svc is not None and name in POSTGRES_NAMES and (is_floating(tag, digest) or _major(tag) >= 18):
            vol = _postgres_old_mount(svc)
            if vol is not None:
                findings.append(Finding(
                    rule="postgres18-mount-path", fault_id="fault-003", severity="error",
                    file=f, line=line_no, evidence=f"{line.strip()}  +  line {_line(vol)}: {vol.value}",
                    message=f"{ref} can resolve to PostgreSQL 18+, which exits when a volume is mounted at "
                            f"{POSTGRES_OLD_DATA} (line {_line(vol)})",
                    fix="pin it to postgres:16 (keeps the data path), or move the mount to /var/lib/postgresql "
                        "before going to 18",
                    fixes=[FileFix(path=f, edits=[LineEdit(line=line_no, replace=[_repin(line, name, tag, "16")])])]))
    return findings


def _major(tag: str | None) -> int:
    m = re.match(r"(\d+)", tag or "")
    return int(m.group(1)) if m else 0


def _postgres_old_mount(svc: Service) -> yaml.Node | None:
    vols = svc.get("volumes")
    for item in vols.value if isinstance(vols, yaml.SequenceNode) else []:
        if isinstance(item, yaml.ScalarNode):
            parts = str(item.value).split(":")
            if len(parts) >= 2 and parts[1].rstrip("/") == POSTGRES_OLD_DATA:
                return item
        elif isinstance(item, yaml.MappingNode):
            target = mapping_get(item, "target")
            if target is not None and str(target.value).rstrip("/") == POSTGRES_OLD_DATA:
                return target
    return None


# --- fault-004: CRLF in shell scripts ------------------------------------------------------

def _gitattributes_keeps_sh_lf(src: RepoSource) -> bool:
    for f in src.files():
        if posixpath.basename(f) != ".gitattributes":
            continue
        for line in _lines(src, f):
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#") and "eol=lf" in parts[1:] \
                    and fnmatch.fnmatchcase("x.sh", posixpath.basename(parts[0])):
                return True
    return False


def rule_crlf(src: RepoSource) -> list[Finding]:
    findings = []
    scripts = [f for f in src.files() if f.endswith(".sh")]
    for f in scripts:
        data = src.read_bytes(f)
        if b"\r\n" in data:
            lines = data.decode("utf-8", errors="replace").split("\n")
            first = next(i for i, l in enumerate(lines, 1) if l.endswith("\r"))
            edits = [LineEdit(line=i, replace=[l[:-1]]) for i, l in enumerate(lines, 1) if l.endswith("\r")]
            findings.append(Finding(
                rule="crlf-shell-script", fault_id="fault-004", severity="error", file=f, line=first,
                evidence=f"{len(edits)} line(s) end in CRLF",
                message="shell script has CRLF line endings: bash reads every \\r (fault-004)",
                fix="convert to LF", fixes=[FileFix(path=f, edits=edits)]))
    if scripts and not _gitattributes_keeps_sh_lf(src):
        existing = ".gitattributes" in src.files()
        rule_line = "*.sh text eol=lf"
        fix = (FileFix(path=".gitattributes", edits=[LineEdit(line=len(_lines(src, ".gitattributes")),
                                                              insert_after=[rule_line])])
               if existing else
               FileFix(path=".gitattributes", new_content="# Shell scripts run in Linux containers: keep LF on "
                                                          "every checkout (fault-004)\n" + rule_line + "\n"))
        findings.append(Finding(
            rule="crlf-shell-script", fault_id="fault-004", severity="warning", file=".gitattributes",
            line=0, evidence=f"{len(scripts)} shell script(s) ({', '.join(scripts)}); no .gitattributes rule "
                             "with eol=lf for *.sh",
            message="nothing keeps the shell scripts LF: a Windows checkout with core.autocrlf=true turns "
                    "them into CRLF and they fail in Linux containers (fault-004)",
            fix=f"{'add' if existing else 'create .gitattributes with'} `{rule_line}` so every checkout keeps "
                "them LF, whatever each teammate's git settings",
            fixes=[fix]))
    return findings


# --- fault-005: MLflow local artifact root ---------------------------------------------------

ARTIFACT_ROOT_RE = re.compile(r"--default-artifact-root[=\s]+(['\"]?)(\S+?)\1(?=[\s\"',\]]|$)")


def _resolve(value: str, env: dict[str, str]) -> str | None:
    m = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", value)
    if not m:
        return value
    v = env.get(m.group(1))
    return v if v and not v.startswith("$") else None


def _is_local_path(value: str) -> bool:
    return not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or value.startswith("file:")


def rule_mlflow_artifacts(src: RepoSource) -> list[Finding]:
    findings = []
    svcs = [s for f in compose_files(src) for s in services(src, f)]
    # Dockerfiles that start an MLflow server; their env comes from the services that build them
    for f in dockerfiles(src):
        users = [s for s in svcs if s.build_dockerfile() == f]
        env = {}
        for line in _lines(src, f):
            m = re.match(r"\s*ENV\s+([A-Za-z_][A-Za-z0-9_]*)[= ](.*)", line)
            if m:
                env[m.group(1)] = m.group(2).strip()
        for s in users:
            env |= s.env()
        for i, line in enumerate(_lines(src, f), 1):
            if "mlflow server" in line and re.match(r"\s*(CMD|ENTRYPOINT|RUN)\b", line):
                findings += _check_mlflow_cmd(f, i, line, env, [s for s in users])
    for s in svcs:
        cmd = s.get("command")
        text = " ".join(str(n.value) for n in cmd.value) if isinstance(cmd, yaml.SequenceNode) else \
            (str(cmd.value) if isinstance(cmd, yaml.ScalarNode) else "")
        if "mlflow server" in text:
            line = _lines(src, s.file)[_line(cmd) - 1]
            findings += _check_mlflow_cmd(s.file, _line(cmd), line if "mlflow server" in line else text,
                                          s.env(), [s])
    return findings


def _check_mlflow_cmd(f: str, line_no: int, line: str, env: dict[str, str], users: list[Service]) -> list[Finding]:
    m = ARTIFACT_ROOT_RE.search(line)
    if not m:
        return []
    raw = m.group(2)
    value = _resolve(raw, env)
    if value is None or not _is_local_path(value):
        return []
    where = f" ({raw} = {value}" + (f", set by service {users[0].name} in {users[0].file})" if users else ")")
    new = line[:m.start()] + f"--serve-artifacts --artifacts-destination {raw}" + line[m.end():]
    return [Finding(
        rule="mlflow-local-artifact-root", fault_id="fault-005", severity="error", file=f, line=line_no,
        evidence=line.strip(),
        message=f"mlflow server uses a plain local path as --default-artifact-root{where}: every client "
                "writes artifacts to its own disk, so other containers can't load the models (fault-005). "
                "--serve-artifacts alone doesn't help while the root is a local path",
        fix="let the server store and serve the artifacts: --serve-artifacts --artifacts-destination <path or "
            "s3://...> instead of --default-artifact-root; new experiments then get mlflow-artifacts:/ URIs "
            "(existing experiments keep their old location)",
        fixes=[FileFix(path=f, edits=[LineEdit(line=line_no, replace=[new])])])]


# --- fault-006: services with ports and no healthcheck ----------------------------------------

def _container_port(ports: yaml.Node) -> str | None:
    first = ports.value[0] if isinstance(ports, yaml.SequenceNode) and ports.value else None
    if isinstance(first, yaml.ScalarNode):
        return str(first.value).split(":")[-1].split("/")[0].split("-")[0]
    if isinstance(first, yaml.MappingNode):
        t = mapping_get(first, "target")
        return str(t.value) if t is not None else None
    return None


def _healthcheck_blocks(src: RepoSource, svcs: list[Service]) -> dict[str, list[str]]:
    """build context -> healthcheck lines (without indentation) from a service that has one."""
    out: dict[str, list[str]] = {}
    for s in svcs:
        hc, ctx = s.get("healthcheck"), s.build_context()
        if hc is None or ctx is None or ctx in out:
            continue
        lines = _lines(src, s.file)
        key_line = hc.start_mark.line - 1                       # the "healthcheck:" line (0-based)
        indent = len(lines[key_line]) - len(lines[key_line].lstrip())
        block = [lines[key_line].strip()]
        for line in lines[key_line + 1:]:
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            if line.strip():
                block.append(line[indent:].rstrip())
        out[ctx] = block
    return out


def _installs(src: RepoSource, dockerfile: str | None, tool: str) -> bool:
    if not dockerfile or not src.exists(dockerfile):
        return False
    text = src.read_text(dockerfile)
    return bool(re.search(rf"\b{tool}\b", text))


def rule_healthchecks(src: RepoSource) -> list[Finding]:
    findings = []
    svcs = [s for f in compose_files(src) for s in services(src, f)]
    known = _healthcheck_blocks(src, svcs)
    for s in svcs:
        ports = s.get("ports")
        if ports is None or s.get("healthcheck") is not None:
            continue
        lines = _lines(src, s.file)
        port = _container_port(ports)
        is_mlflow = "mlflow" in s.name or (s.build_dockerfile() and src.exists(s.build_dockerfile())
                                           and "mlflow server" in src.read_text(s.build_dockerfile()))
        path = "/health" if is_mlflow else "/"
        block, how = None, ""
        if s.build_context() in known:
            block, how = known[s.build_context()], "the healthcheck another compose file already uses for this image"
        elif port and _installs(src, s.build_dockerfile(), "curl"):
            block = ["healthcheck:", f'  test: ["CMD", "curl", "-f", "http://localhost:{port}{path}"]',
                     "  interval: 30s", "  timeout: 10s", "  retries: 5"]
            how = "curl, which its Dockerfile installs"
        elif port and _installs(src, s.build_dockerfile(), "python3"):
            block = ["healthcheck:",
                     f'  test: ["CMD", "python3", "-c", "import urllib.request; '
                     f"urllib.request.urlopen('http://localhost:{port}{path}', timeout=5)\"]",
                     "  interval: 30s", "  timeout: 10s", "  retries: 5"]
            how = "python3, which its Dockerfile installs"
        fixes = []
        if block:
            key_indent = " " * mapping_key_column(s.body, "ports")
            last = max(_line(i) for i in ports.value) if isinstance(ports, yaml.SequenceNode) and ports.value \
                else _line(ports)
            fixes = [FileFix(path=s.file, edits=[LineEdit(line=last, insert_after=[key_indent + b for b in block])])]
        findings.append(Finding(
            rule="service-without-healthcheck", fault_id="fault-006", severity="warning",
            file=s.file, line=_line(s.key), evidence=f"{lines[_line(s.key) - 1].strip()} publishes "
                                                    f"{', '.join(str(p.value) for p in ports.value)} "
                                                    "and has no healthcheck",
            message=f"service {s.name} has ports but no healthcheck: nothing can wait for it "
                    "(depends_on: service_healthy) and a broken service still shows as Up (fault-006)",
            fix=(f"add a healthcheck on http://localhost:{port}{path} using {how}" if block else
                 "add a healthcheck; its image has neither curl nor python3 visible in this repo, so no "
                 "command is proposed"),
            fixes=fixes))
    return findings


def mapping_key_column(node: yaml.MappingNode, key: str) -> int:
    for k, _ in node.value:
        if k.value == key:
            return k.start_mark.column
    return node.start_mark.column


ALL_RULES = [rule_images, rule_crlf, rule_mlflow_artifacts, rule_healthchecks]


def run_rules(src: RepoSource) -> list[Finding]:
    findings = [f for rule in ALL_RULES for f in rule(src)]
    order = {"error": 0, "warning": 1}
    return sorted(findings, key=lambda f: (order[f.severity], f.file, f.line, f.rule))
