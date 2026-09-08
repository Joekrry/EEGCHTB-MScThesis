"""Known-answer tests for the benchmark metrics."""

from __future__ import annotations

import pytest

from eegchtb.evaluation.metrics import METRIC_NAMES, compute_metrics


def test_known_partial_case():
    # y_true=[0,0,1,1], y_pred=[0,1,1,1]: 3/4 correct; f1_0=0.6667, f1_1=0.8
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert set(m) == set(METRIC_NAMES)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_fully_inverted_binary_scores_are_minimal():
    # balanced 2-class, every label flipped: pins kappa and MCC to -1
    m = compute_metrics([0, 0, 1, 1], [1, 1, 0, 0])
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["cohen_kappa"] == pytest.approx(-1.0)
    assert m["mcc"] == pytest.approx(-1.0)
