"""RepoContext: the facts the scanner extracts from a repository.

Facts only, each with the evidence it came from. Choosing between conflicting
facts (e.g. which Python version to use) is the decide layer's job.
All paths are repo-relative and use forward slashes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DepKind = Literal[
    "requirements", "pipfile", "pipfile_lock", "pyproject", "setup_py",
    "setup_cfg", "conda", "poetry_lock", "uv_lock",
]
EntryRole = Literal["train", "predict", "serve", "evaluate", "data", "features", "main", "other"]
FrameworkCategory = Literal["ml", "serving", "tracking", "data"]


class Requirement(BaseModel):
    name: str                      # as written in the file
    spec: str = ""                 # raw version spec, e.g. "==1.3", "*", ">=2,<3"
    dev: bool = False


class DependencyFile(BaseModel):
    path: str
    kind: DepKind
    packages: list[Requirement] = Field(default_factory=list)
    note: str | None = None        # e.g. "install_requires is dynamic", "42 pinned packages"


class VersionHint(BaseModel):
    value: str                     # as written: "3.9", ">=3.10", "3.9.18"
    source: str                    # file path
    kind: str                      # pipfile, requires-python, dockerfile, pyc-cache, ...


class ThirdPartyImport(BaseModel):
    module: str                    # top-level import name, e.g. "sklearn"
    distribution: str              # likely pip name, e.g. "scikit-learn"
    declared: bool                 # found in some dependency file
    files: list[str]               # where it is imported (.py and .ipynb)

    @property
    def notebooks_only(self) -> bool:
        """Imported only from notebooks, never from pipeline code (.py)."""
        return all(f.endswith(".ipynb") for f in self.files)


class Framework(BaseModel):
    name: str                      # pip name
    category: FrameworkCategory
    via: list[Literal["import", "dependency"]]


class FunctionSig(BaseModel):
    name: str
    params: list[str] = Field(default_factory=list)  # "target_col", "num_rounds=371", "*args", "**kw"
    doc: str | None = None                           # first docstring line

    def render(self) -> str:
        return f"{self.name}({', '.join(self.params)})"


class EntryPoint(BaseModel):
    path: str
    role: EntryRole
    has_main_guard: bool
    cli: str | None = None         # argparse, click, typer, fire, sys.argv
    functions: list[str] = Field(default_factory=list)  # top-level defs (capped)
    signatures: list[FunctionSig] = Field(default_factory=list)  # same defs with parameters


class DataColumns(BaseModel):
    path: str
    columns: list[str]             # header only (capped)
    n_columns: int


class TargetCandidate(BaseModel):
    column: str
    score: int                     # weighted evidence count (drop=1, label/fit/y-assign/target-*=3)
    in_data: bool                  # appears in a data file header
    evidence: list[str]            # "label: notebooks/x.ipynb", "drop: src/train.py", ...


class TargetTransform(BaseModel):
    forward: str                   # "log1p"
    inverse: str                   # "expm1"
    applied_to: list[str]          # target columns, or "via <variable>" if not resolvable
    forward_files: list[str]
    inverse_files: list[str]       # empty: inverse never applied (e.g. serving returns log values)


class Route(BaseModel):
    methods: list[str]
    path: str


class WebApp(BaseModel):
    path: str
    framework: Literal["flask", "fastapi"]
    app_var: str
    target: str                    # "pkg.module:app", for gunicorn/uvicorn
    routes: list[Route] = Field(default_factory=list)
    port: int | None = None        # only if set explicitly in code
    port_env_var: str | None = None


class PortHint(BaseModel):
    port: int
    source: str
    kind: str                      # app.run, uvicorn.run, dockerfile-expose, bind-flag, compose, procfile


class PathReference(BaseModel):
    literal: str                   # string literal from code, e.g. "../data/train.csv"
    files: list[str]
    exists: bool
    resolved: str | None = None    # repo-relative path it points to, if it exists


class ArtifactDir(BaseModel):
    path: str
    kind: Literal["data", "model"]
    subdirs: list[str] = Field(default_factory=list)
    n_files: int
    total_bytes: int
    extensions: dict[str, int] = Field(default_factory=dict)


class FileInfo(BaseModel):
    path: str
    size_bytes: int


class Notebook(BaseModel):
    path: str
    size_bytes: int
    n_code_cells: int = 0
    parse_ok: bool = True


class LfsPointer(BaseModel):
    path: str
    object_size: int | None = None  # size of the real file on the LFS server


class LfsInfo(BaseModel):
    patterns: list[str] = Field(default_factory=list)       # "path/.gitattributes: pattern"
    tracked_files: list[str] = Field(default_factory=list)  # files matching an LFS pattern
    pointer_files: list[LfsPointer] = Field(default_factory=list)  # content not pulled


class ParseError(BaseModel):
    path: str
    error: str


class RepoContext(BaseModel):
    root: str
    name: str
    is_git_repo: bool
    n_python_files: int = 0

    dependency_files: list[DependencyFile] = Field(default_factory=list)
    python_version_hints: list[VersionHint] = Field(default_factory=list)

    third_party_imports: list[ThirdPartyImport] = Field(default_factory=list)
    stdlib_modules: list[str] = Field(default_factory=list)
    local_modules: list[str] = Field(default_factory=list)
    frameworks: list[Framework] = Field(default_factory=list)

    entry_points: list[EntryPoint] = Field(default_factory=list)
    web_apps: list[WebApp] = Field(default_factory=list)
    port_hints: list[PortHint] = Field(default_factory=list)
    path_references: list[PathReference] = Field(default_factory=list)

    data_dirs: list[ArtifactDir] = Field(default_factory=list)
    model_dirs: list[ArtifactDir] = Field(default_factory=list)
    model_files: list[FileInfo] = Field(default_factory=list)
    loose_data_files: list[FileInfo] = Field(default_factory=list)  # data files outside data dirs
    notebooks: list[Notebook] = Field(default_factory=list)
    lfs: LfsInfo = Field(default_factory=LfsInfo)

    # slot candidates for the LLM
    data_columns: list[DataColumns] = Field(default_factory=list)   # CSV/TSV headers in data dirs
    target_candidates: list[TargetCandidate] = Field(default_factory=list)  # best first
    target_transforms: list[TargetTransform] = Field(default_factory=list)

    existing_pipeline_files: list[str] = Field(default_factory=list)  # Dockerfile, Makefile, CI, ...
    test_files: list[str] = Field(default_factory=list)
    has_pytest_config: bool = False
    parse_errors: list[ParseError] = Field(default_factory=list)

    @property
    def undeclared_imports(self) -> list[ThirdPartyImport]:
        return [i for i in self.third_party_imports if not i.declared]

    def summary(self, max_items: int = 8) -> str:
        """Compact, deterministic text for the LLM planner (small context)."""
        from .summary import render_summary
        return render_summary(self, max_items)
