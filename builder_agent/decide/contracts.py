"""Typed view of contracts.yaml. Only the sections the Builder uses are typed;
everything else is kept as-is (extra="allow")."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class Project(_Section):
    python_version: str = "auto"

    @field_validator("python_version", mode="before")
    @classmethod
    def _must_be_string(cls, v: Any) -> Any:
        if isinstance(v, float):
            # YAML reads 3.10 as the float 3.1
            raise ValueError(f'python_version must be a quoted string, e.g. "3.10" (got {v!r})')
        return v


class Endpoint(_Section):
    method: str
    path: str


class Serving(_Section):
    mode: Literal["auto", "repo", "generate"]
    port: int
    endpoints: dict[str, Endpoint]
    model_uri_env: str


class SampleMode(_Section):
    variable: str
    fraction: float


class Contracts(_Section):
    contract_version: int
    project: Project
    serving: Serving
    make_targets: dict[str, str]
    sample_mode: SampleMode
    paths: dict[str, str]
    smoke_test: dict[str, Any]


def load_contracts(path: str | Path) -> Contracts:
    with open(path, encoding="utf-8") as f:
        return Contracts.model_validate(yaml.safe_load(f))
