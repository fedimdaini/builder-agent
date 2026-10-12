"""Integration rehearsal (ROADMAP A4b): a copy of contracts.yaml with every integration field changed to a
different, valid value. The real contracts.yaml is only read.

    python experiments/integration/make_rehearsal_contracts.py   # writes contracts.rehearsal.yaml next to it

Fields left at their default, and why, are in NOT_CHANGED (also printed).
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("contracts.rehearsal.yaml")

NEW = {
    ("monitor", "predictions_log"): "monitoring/predictions.log.jsonl",
    ("monitor", "predictions_log_fields"): {"time": "ts", "model_version": "model", "features": "inputs",
                                            "prediction": "output", "request_id": "trace_id"},
    ("trainer", "eval_report"): "metrics_out/evaluation.json",
    ("trainer", "eval_report_fields"): {"model_version": "model", "task": "kind", "metrics": "scores",
                                        "per_group": "by_group"},
    ("trainer", "eval_metrics"): ["rmse", "mae"],
    ("trainer", "eval_classification_metrics"): ["accuracy", "f1_macro"],
    ("trainer", "latest_model_uri"): "state/latest_model.txt",
    ("trainer", "mlflow_experiment"): "rehearsal-experiment",
    ("inspector", "raw_data"): "data/external",
    ("inspector", "schema_path"): "quality/schema.py",
    ("deployer", "serving_port"): 8081,
    ("deployer", "endpoints"): {"predict": {"method": "POST", "path": "/v1/predict"},
                                "health": {"method": "GET", "path": "/v1/healthz"}},
    ("deployer", "prediction_field"): "y_pred",
    ("deployer", "model_version_field"): "model_tag",
    ("deployer", "model_uri_env"): "SERVING_MODEL_URI",
    ("deployer", "current_service"): "model-live",
    ("deployer", "candidate_service"): "model-next",
    ("deployer", "deploy_file"): "deploy/compose.deploy.yaml",
    ("deployer", "router_config"): "deploy/router.yaml",
    ("board", "url"): "http://board-svc:5678",
    ("board", "image"): "hashicorp/http-echo:1.0.0",          # a small public HTTP server (listens on 5678)
    ("board", "agents_file"): "team/agents.yaml",
    ("board", "subscriptions"): {
        "builder": ["pipeline:*", "type:question"],
        "reviewer": ["code:*", "feature:*", "pipeline:*"],
        "inspector": ["dataset:*", "type:question"],
        "trainer": ["dataset:*", "model:*", "type:alert"],
        "deployer": ["deployment:*"],
        "monitor": ["feature:*", "model:*", "expected_change:*", "type:alert"]},
    ("reviewer", "builder_commit_author"): "Rehearsal Bot <rehearsal@example.com>",
    ("reviewer", "ci_workflow"): ".github/workflows/builder-ci.yml",
    ("team", "mlflow_uri"): "http://tracking:5050",
    ("team", "mlflow_server_image"): "ghcr.io/mlflow/mlflow:v2.17.1",
}
NOT_CHANGED = {
    ("inspector", "processed_data"): "names where the repo keeps its processed data: data/processed in both test repos",
    ("trainer", "train_command"): "must be `make <target>`; the only training target is train",
    ("trainer", "evaluate_command"): "must be `make <target>`; the only evaluation target is evaluate",
    ("deployer", "serve_command"): "must be `make <target>`; the only serving target is serve",
    ("deployer", "request_format"): "json is the only format the generated service reads",
    ("deployer", "error_status"): "4xx_or_5xx is the only rule the service implements",
    ("deployer", "candidate_and_router"): "left off on purpose; true is checked separately (refused)",
}


def main() -> None:
    data = yaml.safe_load((ROOT / "contracts.yaml").read_text(encoding="utf-8"))
    fields = {(g, n) for g, group in data["integration"].items() for n in group}
    assert set(NEW) | set(NOT_CHANGED) == fields, sorted(fields ^ (set(NEW) | set(NOT_CHANGED)))
    for (g, n), value in NEW.items():
        assert value != data["integration"][g][n]["default"], (g, n)
        data["integration"][g][n]["value"] = value
    OUT.write_text("# Integration rehearsal copy of contracts.yaml (made by make_rehearsal_contracts.py): every\n"
                   "# integration field that has another valid value is changed. Never used for real runs.\n"
                   + yaml.safe_dump(data, sort_keys=False, width=110), encoding="utf-8", newline="\n")
    print(f"{len(NEW)} changed, {len(NOT_CHANGED)} not changed -> {OUT.relative_to(ROOT)}")
    for (g, n), why in NOT_CHANGED.items():
        print(f"  not changed: {g}.{n}: {why}")


if __name__ == "__main__":
    main()
