"""Benchmark classification metrics computed from predictions."""

from __future__ import annotations

import numpy as np

# reported per run; all four are higher-is-better
METRIC_NAMES = ("accuracy", "macro_f1", "cohen_kappa", "mcc")


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy, macro F1, Cohen kappa and MCC for integer label predictions."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
    )

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.size == 0:
        raise ValueError("cannot compute metrics on empty predictions")

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
    }
