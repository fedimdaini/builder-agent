"""Signals of the ML task (regression or classification), for the decide rule. Facts only.

Three kinds, each with the task it points to:
- model: a model class the code instantiates (XGBClassifier(...), LinearRegression(...));
- metric: a metric function the code calls (accuracy_score(...), mean_squared_error(...));
- objective: an "objective" setting (xgb.train({"objective": "reg:squarederror"}), objective="binary");
- target_values: what a sample of the target column looks like (from the data, not the code).
Nothing here reads summary() text, so the prompts that show the scan summary don't change.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from .models import TaskSignal, TargetValues

CLASSIFIERS = {"LogisticRegression", "LogisticRegressionCV", "SVC", "LinearSVC", "NuSVC", "GaussianNB",
               "MultinomialNB", "BernoulliNB", "ComplementNB", "RidgeClassifier", "SGDClassifier",
               "Perceptron", "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"}
REGRESSORS = {"LinearRegression", "Ridge", "RidgeCV", "Lasso", "LassoCV", "ElasticNet", "ElasticNetCV",
              "SVR", "LinearSVR", "NuSVR", "BayesianRidge", "ARDRegression", "Lars", "LassoLars",
              "TheilSenRegressor", "PoissonRegressor", "GammaRegressor", "TweedieRegressor"}
CLASSIFICATION_METRICS = {"accuracy_score", "balanced_accuracy_score", "f1_score", "precision_score",
                          "recall_score", "roc_auc_score", "log_loss", "confusion_matrix",
                          "classification_report", "average_precision_score", "precision_recall_curve",
                          "roc_curve", "matthews_corrcoef"}
REGRESSION_METRICS = {"mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "r2_score",
                      "mean_absolute_percentage_error", "median_absolute_error", "mean_squared_log_error",
                      "explained_variance_score", "max_error"}
# xgboost "reg:squarederror" / "binary:logistic" / "multi:softprob"; lightgbm "regression", "binary", ...
REGRESSION_OBJECTIVE = re.compile(r"^(reg|count|survival|regression|regression_l[12]|huber|fair|poisson|"
                                  r"quantile|mape|gamma|tweedie|rmse|mae|mse|l[12])(:|$)", re.I)
CLASSIFICATION_OBJECTIVE = re.compile(r"^(binary|multi|multiclass|multiclassova|cross_?entropy|"
                                      r"logloss|multiclass_ova)(:|$)", re.I)

MAX_ROWS = 2000            # rows of the target column read from one data file
MAX_CLASSES = 20           # an integer target with more distinct values is read as a quantity
MIN_ROWS = 30              # fewer values than this settle nothing (a toy file, a header-only sample)


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _objective_task(value: str) -> str | None:
    if CLASSIFICATION_OBJECTIVE.match(value):
        return "classification"
    if REGRESSION_OBJECTIVE.match(value):
        return "regression"
    return None


def code_signals(tree: ast.Module, path: str) -> list[TaskSignal]:
    out: list[TaskSignal] = []

    def add(kind: str, value: str, task: str) -> None:
        out.append(TaskSignal(kind=kind, value=value, task=task, files=[path]))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _name(node.func)
            if name:
                if name in CLASSIFIERS or (name.endswith("Classifier") and name != "Classifier"):
                    add("model", name, "classification")
                elif name in REGRESSORS or (name.endswith("Regressor") and name != "Regressor"):
                    add("model", name, "regression")
                elif name in CLASSIFICATION_METRICS:
                    add("metric", name, "classification")
                elif name in REGRESSION_METRICS:
                    add("metric", name, "regression")
            for kw in node.keywords:
                if kw.arg == "objective" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    if task := _objective_task(kw.value.value):
                        add("objective", kw.value.value, task)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "objective"
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    if task := _objective_task(v.value):
                        add("objective", v.value, task)
    return out


def merge(signals: list[TaskSignal]) -> list[TaskSignal]:
    """One signal per (kind, value, task), with every file it appears in."""
    merged: dict[tuple[str, str, str], TaskSignal] = {}
    for s in signals:
        key = (s.kind, s.value, s.task)
        if key in merged:
            merged[key].files = sorted(set(merged[key].files) | set(s.files))
        else:
            merged[key] = s.model_copy(deep=True)
    return sorted(merged.values(), key=lambda s: (s.kind, s.value))


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def target_values(root: Path, path: str, column: str, delimiter: str = ",") -> TargetValues | None:
    """The first MAX_ROWS values of `column` in a data file, described; None if unreadable."""
    try:
        with (root / path).open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, [])
            if column not in header:
                return None
            i = header.index(column)
            values = [row[i].strip() for _, row in zip(range(MAX_ROWS), reader) if len(row) > i and row[i].strip()]
    except OSError:
        return None
    if not values:
        return None
    numbers = [_number(v) for v in values]
    numeric = all(n is not None for n in numbers)
    integer = numeric and all(float(n).is_integer() for n in numbers)
    distinct = len(set(values if not numeric else numbers))
    if len(values) < MIN_ROWS:
        task = None
    elif not numeric:
        task = "classification"                      # labels such as "yes"/"no", "True"/"False", "cat"
    elif integer and distinct <= MAX_CLASSES:
        task = "classification"                      # 0/1, or a few integer classes
    elif not integer and distinct > MAX_CLASSES:
        task = "regression"                          # a continuous quantity
    elif integer:
        task = "regression"                          # many distinct integers: a count or a duration
    else:
        task = None                                  # few distinct decimals (e.g. ratings 1.5, 2.0): unclear
    return TargetValues(file=path, column=column, n=len(values), distinct=distinct, numeric=numeric,
                        integer=integer, examples=sorted(set(values), key=values.index)[:5], task=task)
