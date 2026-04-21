from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from tech_challenge.config import Settings
from tech_challenge.data.io import load_dataset
from tech_challenge.features.preprocessing import build_preprocessor, split_feature_types
from tech_challenge.logging_config import configure_logging
from tech_challenge.models.artifacts import (
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
)
from tech_challenge.models.cost import sweep_thresholds
from tech_challenge.models.metrics import compute_classification_metrics
from tech_challenge.utils.reproducibility import set_global_seed

LOGGER = logging.getLogger(__name__)


def _target_to_int(series: pd.Series) -> np.ndarray:
    return series.map({"No": 0, "Yes": 1}).to_numpy(dtype=np.int64)


def _results_to_markdown(results: pd.DataFrame) -> str:
    headers = results.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in results.to_dict(orient="records"):
        values = []
        for header in headers:
            value = record[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_baseline_pipelines(preprocessor: object, seed: int) -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, random_state=seed)),
            ]
        ),
    }


def cross_validate_models(
    pipelines: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    summary: dict[str, dict[str, float]] = {}
    for name, pipeline in pipelines.items():
        fold_metrics: list[dict[str, float]] = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            x_fold_train = x_train.iloc[train_idx]
            x_fold_valid = x_train.iloc[valid_idx]
            y_fold_train = y_train[train_idx]
            y_fold_valid = y_train[valid_idx]
            pipeline.fit(x_fold_train, y_fold_train)
            y_score = pipeline.predict_proba(x_fold_valid)[:, 1]
            fold_metrics.append(
                compute_classification_metrics(y_fold_valid, y_score, threshold=threshold)
            )
        summary[name] = {
            metric_name: float(np.mean([fold[metric_name] for fold in fold_metrics]))
            for metric_name in fold_metrics[0]
        }
    return summary


def train_baselines(settings: Settings) -> pd.DataFrame:
    import mlflow

    configure_logging()
    set_global_seed(settings.random_seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    dataframe = load_dataset(settings)
    x = dataframe.drop(columns=[settings.target_column])
    y = _target_to_int(dataframe[settings.target_column])
    numeric_features, categorical_features = split_feature_types(dataframe, settings.target_column)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_seed,
    )

    pipelines = build_baseline_pipelines(preprocessor, settings.random_seed)
    cv_metrics = cross_validate_models(
        pipelines,
        x_train,
        y_train,
        threshold=settings.decision_threshold,
        seed=settings.random_seed,
    )
    rows = []
    cost_summary = []

    for model_name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            pipeline.fit(x_train, y_train)
            y_score = pipeline.predict_proba(x_test)[:, 1]
            test_metrics = compute_classification_metrics(
                y_test, y_score, threshold=settings.decision_threshold
            )
            threshold_results = sweep_thresholds(
                y_test,
                y_score,
                false_positive_cost=settings.false_positive_cost,
                false_negative_cost=settings.false_negative_cost,
            )
            best_threshold = min(threshold_results, key=lambda item: item["expected_cost"])
            current_threshold = next(
                item
                for item in threshold_results
                if abs(item["threshold"] - settings.decision_threshold) < 1e-9
            )

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "random_seed": settings.random_seed,
                    "decision_threshold": settings.decision_threshold,
                    "dataset_file": settings.data_filename,
                }
            )
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            mlflow.log_metrics(
                {f"cv_{key}": value for key, value in cv_metrics[model_name].items()}
            )
            mlflow.log_metric(
                "expected_cost_at_threshold",
                float(current_threshold["expected_cost"]),
            )
            mlflow.log_metric(
                "best_expected_cost",
                float(best_threshold["expected_cost"]),
            )
            mlflow.log_metric(
                "best_threshold_by_cost",
                float(best_threshold["threshold"]),
            )

            artifact_dir = settings.reports_dir / "baseline_artifacts" / model_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            y_pred = (y_score >= settings.decision_threshold).astype(int)
            confusion_path = save_confusion_matrix(
                y_test,
                y_pred,
                artifact_dir / "confusion_matrix.png",
                title=f"{model_name} Confusion Matrix",
            )
            pr_curve_path = save_pr_curve(
                y_test,
                y_score,
                artifact_dir / "pr_curve.png",
                title=f"{model_name} Precision-Recall",
            )
            roc_curve_path = save_roc_curve(
                y_test,
                y_score,
                artifact_dir / "roc_curve.png",
                title=f"{model_name} ROC Curve",
            )
            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(pr_curve_path))
            mlflow.log_artifact(str(roc_curve_path))

            rows.append(
                {
                    "model": model_name,
                    **test_metrics,
                    "expected_cost_at_threshold": float(current_threshold["expected_cost"]),
                    "best_threshold_by_cost": float(best_threshold["threshold"]),
                    "best_expected_cost": float(best_threshold["expected_cost"]),
                    **{f"cv_{key}": value for key, value in cv_metrics[model_name].items()},
                }
            )
            cost_summary.append({"model": model_name, "thresholds": threshold_results})

            if model_name == "logistic_regression":
                settings.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline.named_steps["preprocessor"], settings.preprocessor_path)
                joblib.dump(pipeline, settings.baseline_bundle_path)
                mlflow.log_artifact(str(settings.baseline_bundle_path))
                mlflow.log_artifact(str(settings.preprocessor_path))

    results = pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.reports_dir / "baseline_results.json"
    output_path.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
    markdown_path = settings.reports_dir / "baseline_results.md"
    markdown_path.write_text(_results_to_markdown(results), encoding="utf-8")
    cost_path = settings.reports_dir / "baseline_cost_summary.json"
    cost_path.write_text(json.dumps(cost_summary, indent=2), encoding="utf-8")
    LOGGER.info("Baseline training finished", extra={"path": str(output_path)})
    return results


def main() -> None:
    settings = Settings()
    results = train_baselines(settings)
    print(json.dumps(results.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
