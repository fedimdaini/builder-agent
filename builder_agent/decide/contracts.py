"""Typed view of contracts.yaml. Only the sections the Builder uses are typed;
everything else is kept as-is (extra="allow").

The integration section is the only source of every value another agent or the team must confirm
(fields {value, default, confirmed}, grouped by who confirms them). load_contracts checks its shape and
fills the views the code reads (serving, paths, router, compose, formats, smoke_test) from the values,
so no consumer reads a team-dependent value from anywhere else. A moved key that still appears outside
integration is an error: there is one source per value."""

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


class IntegrationField(BaseModel):
    group: str                        # who confirms it: monitor, trainer, inspector, deployer, board, reviewer, team
    name: str
    value: Any
    default: Any
    confirmed: bool

    @property
    def at_default(self) -> bool:
        return self.value == self.default


class Contracts(_Section):
    contract_version: int
    project: Project
    serving: Serving
    make_targets: dict[str, str]
    sample_mode: SampleMode
    paths: dict[str, str]
    smoke_test: dict[str, Any]
    integration: dict[str, dict[str, dict[str, Any]]]

    def i(self, group: str, name: str) -> Any:
        """The current value of integration.<group>.<name>."""
        return self.integration[group][name]["value"]

    def integration_fields(self) -> list[IntegrationField]:
        return [IntegrationField(group=g, name=n, **f) for g, fields in self.integration.items() for n, f in fields.items()]


class ContractError(ValueError):
    """contracts.yaml is malformed: a bad integration field, or a value in two places."""


# integration.<group>.<name> -> where load_contracts puts its value for the code that reads it
INTEGRATION_VIEWS = {
    ("monitor", "predictions_log"): ("paths", "predictions_log"),
    ("trainer", "eval_report"): ("paths", "eval_report"),
    ("trainer", "latest_model_uri"): ("paths", "latest_model_uri"),
    ("trainer", "mlflow_experiment"): ("paths", "mlflow_experiment"),
    ("inspector", "raw_data"): ("paths", "raw_data"),
    ("inspector", "processed_data"): ("paths", "processed_data"),
    ("team", "mlflow_uri"): ("paths", "mlflow_uri"),
    ("deployer", "serving_port"): ("serving", "port"),
    ("deployer", "endpoints"): ("serving", "endpoints"),
    ("deployer", "model_uri_env"): ("serving", "model_uri_env"),
    ("deployer", "deploy_file"): ("compose", "deploy_file"),
}
# keys that moved into integration: they may no longer appear in their old place
MOVED = {("serving", k) for k in ("port", "endpoints", "request", "response", "errors", "model_uri_env")} | {
    ("paths", k) for k in ("raw_data", "processed_data", "latest_model_uri", "eval_report", "predictions_log",
                           "mlflow_uri", "mlflow_experiment")} | {
    ("compose", "deploy_file"), ("smoke_test", "predict_body_has"), ("router", None), ("formats", None)}


def apply_integration(raw: dict) -> dict:
    """Check the integration section and fill the code's views from its values (returns a new dict)."""
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    integ = raw.get("integration")
    if not isinstance(integ, dict) or not integ:
        raise ContractError("contracts.yaml has no integration section")
    for g, fields in integ.items():
        for n, f in (fields or {}).items():
            if not isinstance(f, dict) or set(f) != {"value", "default", "confirmed"} or not isinstance(f["confirmed"], bool):
                raise ContractError(f"integration.{g}.{n} must be {{value, default, confirmed}} with confirmed true/false")
    for section, key in MOVED:
        if section in raw and (key is None or key in (raw.get(section) or {})):
            raise ContractError(f"contracts.yaml {section}{'.' + key if key else ''} moved to the integration section; "
                                "remove it here (one source per value)")

    def v(g: str, n: str):
        try:
            return integ[g][n]["value"]
        except KeyError:
            raise ContractError(f"contracts.yaml integration.{g}.{n} is missing") from None
    for (g, n), (section, key) in INTEGRATION_VIEWS.items():
        raw.setdefault(section, {})[key] = v(g, n)
    raw["serving"].update(request={"format": v("deployer", "request_format")},
                          response={"prediction_field": v("deployer", "prediction_field"),
                                    "model_version_field": v("deployer", "model_version_field")},
                          errors={"status": v("deployer", "error_status")})
    raw["router"] = {"config_file": v("deployer", "router_config"),
                     "services": {"current": v("deployer", "current_service"),
                                  "candidate": v("deployer", "candidate_service")}}
    raw["formats"] = {"eval_report": {"metrics": {m: "float" for m in v("trainer", "eval_metrics")},
                                      "classification_metrics": {m: "float" for m in
                                                                 v("trainer", "eval_classification_metrics")}}}
    raw.setdefault("smoke_test", {})["predict_body_has"] = v("deployer", "prediction_field")
    for g, n in (("monitor", "predictions_log_fields"), ("trainer", "eval_report_fields"), ("board", "url"),
                 ("board", "image"), ("board", "agents_file"), ("board", "subscriptions"),
                 ("reviewer", "builder_commit_author"), ("reviewer", "ci_workflow"), ("team", "mlflow_server_image"),
                 ("inspector", "schema_path"), ("trainer", "train_command"), ("trainer", "evaluate_command"),
                 ("deployer", "serve_command"), ("deployer", "candidate_and_router")):
        v(g, n)                                       # read directly by the code (Contracts.i): must exist
    for g, n in (("trainer", "train_command"), ("trainer", "evaluate_command"), ("deployer", "serve_command")):
        words = str(v(g, n)).split()
        if len(words) != 2 or words[0] != "make" or words[1] not in (raw.get("make_targets") or {}):
            raise ContractError(f"integration.{g}.{n} {v(g, n)!r} must be `make <target>` with a target of "
                                "make_targets (renaming a target is a contract change)")
    return raw


def load_contracts(path: str | Path) -> Contracts:
    with open(path, encoding="utf-8") as f:
        return Contracts.model_validate(apply_integration(yaml.safe_load(f)))
