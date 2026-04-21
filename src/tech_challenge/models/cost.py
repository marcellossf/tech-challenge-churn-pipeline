from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    del tn, tp
    return float(fp * false_positive_cost + fn * false_negative_cost)


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    values = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.05)
    results = []
    for threshold in values:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results.append(
            {
                "threshold": float(threshold),
                "expected_cost": expected_cost(
                    y_true,
                    y_score,
                    float(threshold),
                    false_positive_cost,
                    false_negative_cost,
                ),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )
    return results
