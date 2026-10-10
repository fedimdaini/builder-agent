"""BuildPlan: what the Builder will generate, decided by rules from RepoContext + contracts.yaml.

Anything the rules can't settle is marked needs_llm with the reason and the few
facts an LLM would need, so the planner call stays small.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["decided", "needs_llm"]


class NeedsLLM(BaseModel):
    item: str                      # "python", "install", "serving", "target:train", ...
    reason: str
    context: list[str] = Field(default_factory=list)  # relevant scan facts, one per line


class PythonChoice(BaseModel):
    status: Status
    version: str | None = None     # "3.9"
    reason: str
    evidence: list[str] = Field(default_factory=list)
    assumed: bool = False          # no version in the repo: the Builder's default, reported as an assumption


class TaskChoice(BaseModel):
    """Regression or classification, decided by a rule from scan signals (decide/rules.py choose_task)."""
    status: Status
    task: Literal["regression", "classification"] | None = None
    reason: str
    evidence: list[str] = Field(default_factory=list)


class InstallPlan(BaseModel):
    status: Status
    tool: Literal["pipenv", "pip", "poetry"] | None = None
    source: str | None = None      # dependency file the install is based on
    commands: list[str] = Field(default_factory=list)
    extra_packages: list[str] = Field(default_factory=list)  # unpinned, added on top
    reason: str


class ServingPlan(BaseModel):
    status: Status
    mode: Literal["repo", "generate"] | None = None
    reason: str
    framework: Literal["flask", "fastapi"] | None = None
    app_path: str | None = None
    app_target: str | None = None  # "pkg.module:app"
    repo_port: int | None = None   # port the repo uses on its own (ignored: we bind contract port)
    port: int                      # contract serving.port
    command: str | None = None
    contract_gaps: list[str] = Field(default_factory=list)   # seen in the code
    unverified: list[str] = Field(default_factory=list)      # left to the smoke test


class MakeTarget(BaseModel):
    name: str
    description: str               # from contracts.yaml
    status: Status
    command: str | None = None     # final if decided; a candidate (or None) if needs_llm
    reason: str
    depends_on: list[str] = Field(default_factory=list)


class SamplePlan(BaseModel):
    variable: str
    fraction: float


class BuildPlan(BaseModel):
    repo: str
    contract_version: int
    python: PythonChoice
    install: InstallPlan
    serving: ServingPlan
    make_targets: list[MakeTarget]
    sample: SamplePlan
    task: TaskChoice | None = None      # None only in plans built before the task rule existed
    needs_llm: list[NeedsLLM] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.needs_llm

    def target(self, name: str) -> MakeTarget:
        return next(t for t in self.make_targets if t.name == name)

    def summary(self) -> str:
        """Compact text for the LLM planner and for humans."""
        out = [f"PLAN {self.repo} | contract v{self.contract_version} | "
               f"{'complete' if self.is_complete else f'{len(self.needs_llm)} item(s) need the LLM'}"]
        mark = {"decided": "ok ", "needs_llm": "LLM"}
        if self.task:
            out.append(f"[{mark[self.task.status]}] task: {self.task.task or '-'} — {self.task.reason}")
        assumed = " (ASSUMED)" if self.python.assumed else ""
        out.append(f"[{mark[self.python.status]}] python: {self.python.version or '-'}{assumed} — {self.python.reason}")
        extras = f" + pip install {' '.join(self.install.extra_packages)}" if self.install.extra_packages else ""
        out.append(f"[{mark[self.install.status]}] install: {self.install.tool or '-'} "
                   f"({self.install.source or '-'}){extras} — {self.install.reason}")
        s = self.serving
        out.append(f"[{mark[s.status]}] serving: mode={s.mode or '-'} {s.app_target or ''} port {s.port}"
                   + (f" (repo uses {s.repo_port})" if s.repo_port and s.repo_port != s.port else "")
                   + f" — {s.reason}")
        for g in s.contract_gaps:
            out.append(f"      gap: {g}")
        if s.unverified:
            out.append(f"      smoke test must verify: {'; '.join(s.unverified)}")
        out.append("MAKE TARGETS:")
        for t in self.make_targets:
            cmd = f"`{t.command}`" if t.command else "-"
            out.append(f"  [{mark[t.status]}] {t.name}: {cmd} — {t.reason}")
        return "\n".join(out)
