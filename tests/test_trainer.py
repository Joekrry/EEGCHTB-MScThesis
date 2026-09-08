"""Tests for the deterministic supervised trainer."""

from __future__ import annotations

import numpy as np

from conftest import requires_torch

from eegchtb.data import EpochBundle


def _learnable_bundle(n=48, n_ch=3, n_t=128, seed=0):
    """Two-class bundle where class 1 carries a 10 Hz burst, so the model can learn."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    data = (rng.standard_normal((n, n_ch, n_t)) * 0.1).astype(np.float32)
    t = np.linspace(0, 1, n_t)
    burst = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    data[labels == 1] += burst
    return EpochBundle(
        data=data,
        labels=labels,
        dataset="synthetic",
        subject=1,
        session="T",
        sfreq=128.0,
        channel_names=[f"c{i}" for i in range(n_ch)],
        class_names={0: "a", 1: "b"},
        trial_ids=np.arange(n),
    )


def _train_once():
    from eegchtb.evaluation import EpochBundleDataset
    from eegchtb.models import EEGNet
    from eegchtb.training import TrainConfig, predict, set_seed, train_model

    ds = EpochBundleDataset(_learnable_bundle())
    set_seed(0)
    model = EEGNet(n_channels=3, n_times=128, n_classes=2, kernel_length=32)
    result = train_model(model, ds, ds, TrainConfig(epochs=5, batch_size=8, patience=10, seed=0))
    return predict(model, ds), result


@requires_torch()
def test_trains_deterministically_twice():
    preds_a, res_a = _train_once()
    preds_b, res_b = _train_once()
    assert np.array_equal(preds_a, preds_b)
    assert res_a.history["train_loss"] == res_b.history["train_loss"]


@requires_torch()
def test_training_reduces_loss_and_records_history_and_timings():
    _, result = _train_once()
    train_loss = result.history["train_loss"]
    assert train_loss[-1] < train_loss[0]  # learns on a learnable problem
    assert len(result.history["val_loss"]) == len(train_loss)
    assert "train_seconds" in result.timings
    assert result.epochs_run >= 1
