"""The shared grid recipe: canonical config, segment-recombine augmentation, and the trainer augment hook."""

from __future__ import annotations

import numpy as np
import pytest

from eegchtb.training import GRID_VAL_FRACTION, grid_train_config, make_augment, segment_recombine

torch = pytest.importorskip("torch")


def test_grid_config_is_uniform_and_leakage_safe():
    a = grid_train_config(seed=0)
    b = grid_train_config(seed=1)
    # only the seed differs; every other knob is identical across models/cells
    assert a.lr == b.lr and a.weight_decay == b.weight_decay
    assert a.scheduler == "cosine"
    assert a.patience > 0  # best-val early stopping, not a fixed schedule
    assert 0.0 < GRID_VAL_FRACTION < 0.5


def test_segment_recombine_shapes_and_labels():
    rng = np.random.default_rng(0)
    x = torch.randn(40, 3, 100)
    y = torch.arange(4).repeat_interleave(10)
    ax, ay = segment_recombine(x, y, n_out=40, n_classes=4, rng=rng)
    assert ax.shape[1:] == x.shape[1:]
    assert len(ax) == len(ay)
    assert set(ay.tolist()) <= set(range(4))


def test_augment_hook_grows_the_batch():
    x = torch.randn(40, 3, 100)
    y = torch.arange(4).repeat_interleave(10)
    augment = make_augment(x, y, seed=0, n_classes=4)
    xb, yb = x[:8], y[:8]
    ax, ay = augment(xb, yb)
    # the closure appends a batch-sized block of synthetic trials
    assert len(ax) > len(xb) and ax.shape[1:] == xb.shape[1:]
    assert len(ax) == len(ay)


def test_augment_hook_off_by_default():
    from eegchtb.training import TrainConfig

    assert TrainConfig().augment is None
