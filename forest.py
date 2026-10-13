from __future__ import annotations

import json
import os
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_COL = "Cover_Type"
SAMPLE_SIZE = 50_000
SAMPLE_PATH = "forest_sample.csv"
EXPERIMENT_NAME = "forest-cover"
MODEL_PATH = "forest_model.joblib"
METADATA_PATH = "forest_model.metadata.json"


# --------------------------------------------------------------------------- #
# Column groups (Partner A)
# --------------------------------------------------------------------------- #
def get_column_groups() -> tuple[list[str], list[str], list[str]]:
    """10 numeric cartographic columns, 4 wilderness-area binaries,
    40 soil-type binaries — exactly the fetch_covtype(as_frame=True) schema."""
    numeric_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    wilderness_cols = [f"Wilderness_Area_{i}" for i in range(4)]
    soil_cols = [f"Soil_Type_{i}" for i in range(40)]
    return numeric_cols, wilderness_cols, soil_cols


# --------------------------------------------------------------------------- #
# Data (Partner A)
# --------------------------------------------------------------------------- #
def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    train_size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Pull a class-proportional subsample of `train_size` rows out of `df`.

    Pure function (no I/O, no network) so it's directly unit-testable —
    the real 581k-row fetch is done separately in load_and_sample().
    """
    sampled, _ = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return sampled.reset_index(drop=True)


def load_and_sample(
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    out_path: str = SAMPLE_PATH,
) -> pd.DataFrame:
    """Fetch the full 581k-row dataset, print shape + class distribution,
    stratified-sample down to `sample_size` rows, and cache to CSV so all
    three partners train on byte-identical data."""
    bunch = fetch_covtype(as_frame=True)
    full = bunch.frame
    print(f"Original shape: {full.shape}")
    print("Original target distribution:")
    print(full[TARGET_COL].value_counts().sort_index())

    sample = stratified_sample(full, TARGET_COL, sample_size, random_state)
    sample.to_csv(out_path, index=False)

    print(f"\nSampled to {sample.shape[0]} rows -> {out_path}")
    print("Sampled class counts:")
    print(sample[TARGET_COL].value_counts().sort_index())
    return sample


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Preprocessing (shared)
# --------------------------------------------------------------------------- #
def build_preprocessor(
    numeric_cols: list[str], wilderness_cols: list[str], soil_cols: list[str]
) -> ColumnTransformer:
    """Scale the 10 numeric cartographic columns; pass the 44 already-binary
    columns through untouched — standardizing a 0/1 column just adds noise."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", wilderness_cols + soil_cols),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation (shared helper — used by both baseline and tuning runs)
# --------------------------------------------------------------------------- #
def evaluate_pipeline(
    pipe: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: KFold,
) -> dict:
    """Cross-validate on train, fit on full train, score on held-out test.
    Returns the four metrics every MLflow run logs."""
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


# --------------------------------------------------------------------------- #
# Multi-model comparison (Partner B)
# --------------------------------------------------------------------------- #
def get_baseline_models() -> dict:
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gb": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def log_baseline_run(
    name: str,
    model,
    pre: ColumnTransformer,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: KFold,
) -> tuple[str, dict]:
    pipe = Pipeline([("pre", pre), ("model", model)])

    with mlflow.start_run(run_name=name) as run:
        metrics = evaluate_pipeline(pipe, X_train, y_train, X_test, y_test, cv)

        mlflow.log_param("model_type", name)
        mlflow.log_param("preprocess", "scale_numeric_passthrough_binary")
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(pipe, name="model")

        run_id = run.info.run_id
        print(f"AUDIT MARKER: run_id={run_id}")

    print(
        f"[{name}] cv_acc={metrics['cv_mean_accuracy']:.4f} "
        f"(+/-{metrics['cv_std_accuracy']:.4f})  "
        f"test_acc={metrics['test_accuracy']:.4f}  "
        f"test_f1={metrics['test_macro_f1']:.4f}"
    )
    return run_id, metrics


def run_baseline_comparison(pre, X_train, y_train, X_test, y_test) -> dict:
    """Fits + logs all three baseline model families. Returns
    {model_name: {"run_id":..., "metrics":...}} so Ex5/6-style downstream
    steps (winner selection, RUNS.md) can consume it directly."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in get_baseline_models().items():
        run_id, metrics = log_baseline_run(
            name, model, pre, X_train, y_train, X_test, y_test, cv
        )
        results[name] = {"run_id": run_id, "metrics": metrics}
    return results


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (Partner C)
# --------------------------------------------------------------------------- #
def tune_winner(pre, X_train, y_train, X_test, y_test) -> tuple[str, dict, dict, Pipeline]:
    """GridSearchCV over HistGradientBoostingClassifier. RandomForest is the
    accuracy winner among the baselines, but per the project spec the search
    grid targets HGB's (learning_rate, max_iter, max_leaf_nodes) knobs, and
    tuned HGB is the 4th logged run ("hgb-tuned")."""
    pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))])

    param_grid = {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_iter": [200, 500],
        "model__max_leaf_nodes": [31, 63],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)

    with mlflow.start_run(run_name="hgb-tuned") as run:
        gs.fit(X_train, y_train)

        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        y_pred = gs.best_estimator_.predict(X_test)
        metrics = {
            "cv_mean_accuracy": float(gs.best_score_),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        }
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(gs.best_estimator_, name="model")

        run_id = run.info.run_id
        print(f"AUDIT MARKER: run_id={run_id}")

    print(f"Best params: {best_params}")
    print(f"[hgb-tuned] cv_acc={metrics['cv_mean_accuracy']:.4f}  test_acc={metrics['test_accuracy']:.4f}")

    top3 = (
        pd.DataFrame(gs.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .head(3)[["params", "mean_test_score", "std_test_score"]]
    )
    print("\nTop 3 grid combinations:")
    print(top3.to_string(index=False))

    return run_id, best_params, metrics, gs.best_estimator_


# --------------------------------------------------------------------------- #
# Persist (Partner C)
# --------------------------------------------------------------------------- #
def persist_model(
    tuned_pipe: Pipeline,
    best_params: dict,
    metrics: dict,
    X_train,
    X_test,
    n_classes: int,
    model_path: str = MODEL_PATH,
    metadata_path: str = METADATA_PATH,
) -> dict:
    joblib.dump(tuned_pipe, model_path)

    metadata = {
        "model_version": "1.0.0",
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_type": "HistGradientBoostingClassifier",
        "best_params": best_params,
        "metrics": {
            "cv_mean_accuracy": metrics["cv_mean_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
        },
        "training_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "n_classes": n_classes,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Reload and confirm the persisted pipeline reproduces the same test score.
    reloaded = joblib.load(model_path)
    return metadata, reloaded


def write_runs_md(
    experiment_name: str, baseline_results: dict, tuned_run_id: str, tuned_metrics: dict, path: str = "RUNS.md"
) -> None:
    rows = []
    for name, r in baseline_results.items():
        rows.append((name, r["run_id"], r["metrics"]["test_accuracy"]))
    rows.append(("hgb-tuned", tuned_run_id, tuned_metrics["test_accuracy"]))

    lines = [
        "# MLflow Audit — forest-cover",
        "",
        f"**Experiment name:** `{experiment_name}`",
        "",
        "| run name | run id | test accuracy |",
        "|---|---|---|",
    ]
    for name, run_id, acc in rows:
        lines.append(f"| {name} | `{run_id}` | {acc:.4f} |")
    lines.append("")
    lines.append("View with `mlflow ui` from the project root.")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    # Pin the tracking store explicitly to a local mlruns/ file store. Some
    # MLflow installs default to a mlflow.db SQLite backend instead — the
    # spec and the audit both expect a plain file-based mlruns/ directory,
    # so don't rely on whatever the ambient default happens to be.
    mlflow.set_tracking_uri("file:./mlruns")

    numeric_cols, wilderness_cols, soil_cols = get_column_groups()

    print("=" * 70)
    print("1-2. Load and sample")
    print("=" * 70)
    df = load_and_sample()

    print("\n" + "=" * 70)
    print("3. EDA and split")
    print("=" * 70)
    assert len(numeric_cols) + len(wilderness_cols) + len(soil_cols) == 54
    assert df.isna().sum().sum() == 0, "unexpected missing values in sample"

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"train shape: {X_train.shape}  test shape: {X_test.shape}")
    print("train class proportions:")
    print(y_train.value_counts(normalize=True).sort_index().round(3))

    pre = build_preprocessor(numeric_cols, wilderness_cols, soil_cols)

    print("\n" + "=" * 70)
    print("4. Multi-model comparison with MLflow")
    print("=" * 70)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"AUDIT MARKER: experiment_name={EXPERIMENT_NAME}")
    baseline_results = run_baseline_comparison(pre, X_train, y_train, X_test, y_test)

    winner = max(baseline_results, key=lambda k: baseline_results[k]["metrics"]["cv_mean_accuracy"])
    print(f"\nBaseline winner by CV accuracy: {winner}")

    print("\n" + "=" * 70)
    print("5. Hyperparameter tuning")
    print("=" * 70)
    tuned_run_id, best_params, tuned_metrics, tuned_pipe = tune_winner(
        pre, X_train, y_train, X_test, y_test
    )

    print("\n" + "=" * 70)
    print("6. Persist the winner")
    print("=" * 70)
    n_classes = int(y_train.nunique())
    metadata, reloaded = persist_model(
        tuned_pipe, best_params, tuned_metrics, X_train, X_test, n_classes
    )
    reload_acc = accuracy_score(y_test, reloaded.predict(X_test))
    print(f"Saved -> {MODEL_PATH}, {METADATA_PATH}")
    print(f"Reloaded test accuracy: {reload_acc:.4f} (matches original: {np.isclose(reload_acc, tuned_metrics['test_accuracy'])})")

    write_runs_md(EXPERIMENT_NAME, baseline_results, tuned_run_id, tuned_metrics)
    print("Wrote RUNS.md")


if __name__ == "__main__":
    main()
