"""Tests for the PyTorch EpochBundle dataset adapter."""

from __future__ import annotations

import numpy as np

from conftest import requires_torch

from eegchtb.data import EpochBundle


def make_bundle(n_trials=5, n_channels=3, n_times=8, subject=3):
    rng = np.random.default_rng(0)
    return EpochBundle(
        data=rng.standard_normal((n_trials, n_channels, n_times)).astype(np.float32),
        labels=np.array([0, 1, 2, 0, 1])[:n_trials],
        dataset="bci_iv_2a",
        subject=subject,
        session="T",
        sfreq=250.0,
        channel_names=[f"c{i}" for i in range(n_channels)],
        class_names={0: "a", 1: "b", 2: "c"},
        trial_ids=np.arange(n_trials),
    )


@requires_torch()
def test_len_getitem_shape_and_dtype():
    import torch

    from eegchtb.evaluation import EpochBundleDataset

    b = make_bundle()
    ds = EpochBundleDataset(b)
    assert len(ds) == b.n_trials
    x, y = ds[0]
    assert x.shape == (b.n_channels, b.n_times)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert int(y) == int(b.labels[0])


@requires_torch()
def test_batched_tensors_shape_and_labels():
    from eegchtb.evaluation import EpochBundleDataset

    b = make_bundle()
    x, y = EpochBundleDataset(b).tensors()
    assert x.shape == (b.n_trials, b.n_channels, b.n_times)
    assert y.tolist() == b.labels.tolist()


@requires_torch()
def test_works_with_dataloader():
    from torch.utils.data import DataLoader

    from eegchtb.evaluation import EpochBundleDataset

    ds = EpochBundleDataset(make_bundle())
    xb, yb = next(iter(DataLoader(ds, batch_size=2)))
    assert xb.shape == (2, 3, 8)
    assert yb.shape == (2,)


@requires_torch()
def test_does_not_mutate_source_bundle():
    from eegchtb.evaluation import EpochBundleDataset

    b = make_bundle()
    before = b.data.copy()
    ds = EpochBundleDataset(b)
    ds.tensors()[0].add_(1.0)  # mutate the batched tensor in place
    ds[0][0].zero_()  # mutate an indexed item in place
    np.testing.assert_array_equal(b.data, before)


@requires_torch()
def test_subjects_and_trial_ids_retained_per_trial():
    import torch

    from eegchtb.evaluation import EpochBundleDataset

    ds = EpochBundleDataset(make_bundle(subject=7))
    assert torch.equal(ds.subjects, torch.full((5,), 7, dtype=torch.long))
    assert ds.trial_ids.tolist() == [0, 1, 2, 3, 4]
