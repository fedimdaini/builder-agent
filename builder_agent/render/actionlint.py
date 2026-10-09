"""actionlint for rendered GitHub Actions workflows: found on PATH, or downloaded once (pinned, checksum-verified)
into builder-agent's own .tools/ folder."""

from __future__ import annotations

import hashlib
import io
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.7.7"
# from https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_checksums.txt
SHA256 = {
    "windows_amd64.zip": "7f12f1801bca3d480d67aaf7774f4c2a6359a3ca8eebe382c95c10c9704aa731",
    "linux_amd64.tar.gz": "023070a287cd8cccd71515fedc843f1985bf96c436b7effaecce67290e7e0757",
}
TOOLS = Path(__file__).resolve().parents[2] / ".tools" / f"actionlint-{VERSION}"


def ensure_actionlint() -> Path:
    found = shutil.which("actionlint")
    if found:
        return Path(found)
    exe = TOOLS / ("actionlint.exe" if platform.system() == "Windows" else "actionlint")
    if exe.exists():
        return exe
    asset = "windows_amd64.zip" if platform.system() == "Windows" else "linux_amd64.tar.gz"
    url = f"https://github.com/rhysd/actionlint/releases/download/v{VERSION}/actionlint_{VERSION}_{asset}"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[asset]:
        raise RuntimeError(f"actionlint download checksum mismatch: {digest} != {SHA256[asset]}")
    TOOLS.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            exe.write_bytes(z.read("actionlint.exe"))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            exe.write_bytes(t.extractfile("actionlint").read())
        exe.chmod(0o755)
    return exe


def run_actionlint(text: str, name: str = ".github/workflows/ci.yml") -> tuple[list[str], str]:
    """Lint workflow text (read from stdin, so nothing has to be written first).
    Returns (problems, version). Empty problems: the workflow passed."""
    exe = ensure_actionlint()
    version = subprocess.run([str(exe), "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    r = subprocess.run([str(exe), "-no-color", "-oneline", "-stdin-filename", name, "-"], input=text,
                       capture_output=True, text=True)
    problems = [line for line in (r.stdout + r.stderr).splitlines() if line.strip()]
    if r.returncode not in (0, 1):            # 1 = problems found; anything else = actionlint itself failed
        problems.append(f"actionlint exited {r.returncode}")
    return problems, version
