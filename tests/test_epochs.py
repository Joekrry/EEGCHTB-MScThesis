"""Tests for the EpochBundle container using numpy."""

from __future__ import annotations

import numpy as np
import pytest

from eegchtb.data import EpochBundle, concatenate


def make_bundle(n_trials=6, n_channels=3, n_times=10, labels=None, session="T"):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_trials, n_channels, n_times)).astype(np.float32)
    if labels is None:
        labels = np.array([0, 1, 0, 1, 0, 1])[:n_trials]
    return EpochBundle(
        data=data,
        labels=labels,
        dataset="bci_iv_2a",
        subject=1,
        session=session,
        sfreq=250.0,
        channel_names=[f"C{i}" for i in range(n_channels)],
        class_names={0: "left_hand", 1: "right_hand"},
        times=np.arange(n_times) / 250.0,
        trial_ids=np.arange(n_trials),
    )


def test_shape_properties():
    b = make_bundle()
    assert b.n_trials == 6
    assert b.n_channels == 3
    assert b.n_times == 10


def test_class_distribution():
    b = make_bundle()
    assert b.class_distribution() == {0: 3, 1: 3}


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="labels length"):
        make_bundle(labels=np.array([0, 1]))


def test_channel_name_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="channel_names length"):
        EpochBundle(
            data=rng.standard_normal((2, 3, 5)).astype(np.float32),
            labels=np.array([0, 1]),
            dataset="x",
            subject=1,
            session="T",
            sfreq=250.0,
            channel_names=["only_one"],
            class_names={0: "a", 1: "b"},
        )


def test_filter_known_drops_unknown():
    b = make_bundle(labels=np.array([0, -1, 1, -1, 0, 1]))
    assert b.has_unknown_labels
    known = b.filter_known()
    assert known.n_trials == 4
    assert not known.has_unknown_labels
    assert set(known.labels.tolist()) == {0, 1}


def test_select_classes():
    b = make_bundle()
    only_left = b.select_classes([0])
    assert set(only_left.labels.tolist()) == {0}
    assert only_left.n_trials == 3


def test_save_load_roundtrip(tmp_path):
    b = make_bundle()
    path = tmp_path / "bundle.npz"
    b.save(path)
    loaded = EpochBundle.load(path)
    np.testing.assert_array_equal(loaded.data, b.data)
    np.testing.assert_array_equal(loaded.labels, b.labels)
    assert loaded.channel_names == b.channel_names
    assert loaded.class_names == b.class_names
    assert loaded.sfreq == b.sfreq
    assert loaded.dataset == b.dataset


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    """A failed save leaves any existing bundle intact and no partial or temp file."""
    b = make_bundle()
    path = tmp_path / "bundle.npz"
    b.save(path)
    original = path.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("write failed midway")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(RuntimeError):
        b.save(path)

    # the pre-existing bundle is byte-identical and still loads; nothing else remains
    assert path.read_bytes() == original
    EpochBundle.load(path)
    assert list(tmp_path.iterdir()) == [path]


def test_concatenate_preserves_trials_and_metadata():
    a = make_bundle(session="T")
    c = make_bundle(session="E")
    merged = concatenate([a, c])
    assert merged.n_trials == a.n_trials + c.n_trials
    assert merged.session == "E+T"
    assert merged.metadata["concatenated_from"] == ["T", "E"]


def test_concatenate_rejects_shape_mismatch():
    a = make_bundle(n_channels=3)
    b = make_bundle(n_channels=4)
    with pytest.raises(ValueError, match="channel count mismatch"):
        concatenate([a, b])
