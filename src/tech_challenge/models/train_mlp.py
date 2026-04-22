from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tech_challenge.config import Settings
from tech_challenge.data.io import load_dataset
from tech_challenge.features.preprocessing import build_preprocessor, split_feature_types
from tech_challenge.logging_config import configure_logging
from tech_challenge.models.artifacts import (
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    save_training_history,
)
from tech_challenge.models.cost import sweep_thresholds
from tech_challenge.models.metrics import compute_classification_metrics
from tech_challenge.models.mlp import ChurnMLP
from tech_challenge.utils.reproducibility import set_global_seed

LOGGER = logging.getLogger(__name__)


def _target_to_int(series: pd.Series) -> np.ndarray:
    return series.map({"No": 0, "Yes": 1}).to_numpy(dtype=np.int64)


def _to_dataloader(
    features: np.ndarray, target: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp(settings: Settings) -> dict[str, object]:
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
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=settings.val_size,
        stratify=y_train,
        random_state=settings.random_seed,
    )

    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)

    if hasattr(x_train_t, "toarray"):
        x_train_t = x_train_t.toarray()
        x_valid_t = x_valid_t.toarray()
        x_test_t = x_test_t.toarray()

    train_loader = _to_dataloader(x_train_t, y_train, settings.batch_size, shuffle=True)
    valid_loader = _to_dataloader(x_valid_t, y_valid, settings.batch_size, shuffle=False)

    positive_samples = int(y_train.sum())
    negative_samples = int(len(y_train) - positive_samples)
    pos_weight = torch.tensor(
        [negative_samples / max(positive_samples, 1)],
        dtype=torch.float32,
    )
    model = ChurnMLP(
        input_dim=x_train_t.shape[1],
        hidden_dims=settings.mlp_hidden_dims,
        dropout=settings.mlp_dropout,
    )
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_valid_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    with mlflow.start_run(run_name="mlp_pytorch"):
        mlflow.log_params(
            {
                "batch_size": settings.batch_size,
                "learning_rate": settings.learning_rate,
                "max_epochs": settings.max_epochs,
                "patience": settings.patience,
                "hidden_dims": str(settings.mlp_hidden_dims),
                "dropout": settings.mlp_dropout,
                "random_seed": settings.random_seed,
                "decision_threshold": settings.decision_threshold,
                "positive_class_weight": float(pos_weight.item()),
            }
        )

        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            train_losses = []
            for features, target in train_loader:
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            valid_losses = []
            valid_scores = []
            with torch.no_grad():
                for features, target in valid_loader:
                    logits = model(features)
                    valid_losses.append(float(criterion(logits, target).item()))
                    valid_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

            valid_loss = float(np.mean(valid_losses))
            train_loss = float(np.mean(train_losses))
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            mlflow.log_metrics({"train_loss": train_loss, "valid_loss": valid_loss}, step=epoch)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= settings.patience:
                LOGGER.info("Early stopping triggered", extra={"status_code": 200})
                break

        if best_state is None:
            raise RuntimeError("MLP training did not produce a valid checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = torch.sigmoid(
                model(torch.tensor(x_test_t, dtype=torch.float32))
            ).cpu().numpy()

        test_metrics = compute_classification_metrics(
            y_test, test_scores, threshold=settings.decision_threshold
        )
        threshold_results = sweep_thresholds(
            y_test,
            test_scores,
            false_positive_cost=settings.false_positive_cost,
            false_negative_cost=settings.false_negative_cost,
        )
        best_threshold = min(threshold_results, key=lambda item: item["expected_cost"])
        best_threshold_metrics = compute_classification_metrics(
            y_test,
            test_scores,
            threshold=float(best_threshold["threshold"]),
        )

        settings.mlp_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, settings.preprocessor_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": x_train_t.shape[1],
                "hidden_dims": settings.mlp_hidden_dims,
                "dropout": settings.mlp_dropout,
                "threshold": float(best_threshold["threshold"]),
                "evaluation_threshold": settings.decision_threshold,
                "feature_columns": list(x.columns),
                "model_name": "mlp_pytorch",
                "model_version": "0.1.0",
                "threshold_selection_strategy": "best_expected_cost",
            },
            settings.mlp_bundle_path,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics(
            {f"best_threshold_{key}": value for key, value in best_threshold_metrics.items()}
        )
        mlflow.log_metric("best_threshold_by_cost", float(best_threshold["threshold"]))
        mlflow.log_metric("best_expected_cost", float(best_threshold["expected_cost"]))
        mlflow.log_artifact(str(settings.mlp_bundle_path))
        mlflow.log_artifact(str(settings.preprocessor_path))

        artifact_dir = settings.reports_dir / "mlp_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        history_plot_path = save_training_history(
            history,
            artifact_dir / "training_history.png",
            title="MLP Training History",
        )
        y_pred = (test_scores >= settings.decision_threshold).astype(int)
        confusion_path = save_confusion_matrix(
            y_test,
            y_pred,
            artifact_dir / "confusion_matrix.png",
            title="MLP Confusion Matrix",
        )
        pr_curve_path = save_pr_curve(
            y_test,
            test_scores,
            artifact_dir / "pr_curve.png",
            title="MLP Precision-Recall",
        )
        roc_curve_path = save_roc_curve(
            y_test,
            test_scores,
            artifact_dir / "roc_curve.png",
            title="MLP ROC Curve",
        )
        for path in (history_plot_path, confusion_path, pr_curve_path, roc_curve_path):
            mlflow.log_artifact(str(path))

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.reports_dir / "mlp_results.json"
    history_path = settings.reports_dir / "mlp_history.json"
    threshold_path = settings.reports_dir / "threshold_analysis.json"
    mlp_summary = {
        **test_metrics,
        "expected_cost_at_threshold": float(
            next(
                item["expected_cost"]
                for item in threshold_results
                if abs(item["threshold"] - settings.decision_threshold) < 1e-9
            )
        ),
        "best_threshold_by_cost": float(best_threshold["threshold"]),
        "best_expected_cost": float(best_threshold["expected_cost"]),
        "best_threshold_metrics": best_threshold_metrics,
    }
    metrics_path.write_text(json.dumps(mlp_summary, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    threshold_path.write_text(json.dumps(threshold_results, indent=2), encoding="utf-8")
    markdown_path = settings.reports_dir / "mlp_results.md"
    confusion = confusion_matrix(y_test, (test_scores >= settings.decision_threshold).astype(int))
    markdown_path.write_text(
        "\n".join(
            [
                "# MLP Results",
                "",
                f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
                f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
                f"- F1: {test_metrics['f1']:.4f}",
                f"- Precision: {test_metrics['precision']:.4f}",
                f"- Recall: {test_metrics['recall']:.4f}",
                f"- Accuracy: {test_metrics['accuracy']:.4f}",
                (
                    f"- Expected cost at threshold {settings.decision_threshold:.2f}: "
                    f"{mlp_summary['expected_cost_at_threshold']:.2f}"
                ),
                f"- Threshold used for evaluation table: {settings.decision_threshold:.2f}",
                f"- Best threshold by expected cost: {best_threshold['threshold']:.2f}",
                f"- Best expected cost: {best_threshold['expected_cost']:.2f}",
                f"- Best-threshold PR-AUC: {best_threshold_metrics['pr_auc']:.4f}",
                f"- Best-threshold F1: {best_threshold_metrics['f1']:.4f}",
                f"- Best-threshold Precision: {best_threshold_metrics['precision']:.4f}",
                f"- Best-threshold Recall: {best_threshold_metrics['recall']:.4f}",
                "",
                "## Confusion matrix",
                "",
                f"- TN: {int(confusion[0, 0])}",
                f"- FP: {int(confusion[0, 1])}",
                f"- FN: {int(confusion[1, 0])}",
                f"- TP: {int(confusion[1, 1])}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "metrics": test_metrics,
        "history": history,
        "thresholds": threshold_results,
        "best_threshold": best_threshold,
        "best_threshold_metrics": best_threshold_metrics,
        "paths": {
            "metrics": str(metrics_path),
            "history": str(history_path),
            "thresholds": str(threshold_path),
            "markdown": str(markdown_path),
        },
    }


def main() -> None:
    result = train_mlp(Settings())
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
