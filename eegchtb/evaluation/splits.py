"""Leakage-safe subject-aware splits (subject-specific) over per-trial metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Split:
    """One train/val/test partition as sorted trial-index arrays, plus a provenance dict."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    meta: dict = field(default_factory=dict)


def _carve_val(
    pool: np.ndarray, labels: np.ndarray | None, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a pool of trial indices into (train, val); stratified by label when possible."""
    empty = np.array([], dtype=np.int64)
    if val_fraction <= 0.0 or len(pool) < 2:
        return np.sort(pool), empty

    from sklearn.model_selection import train_test_split

    stratify = None
    if labels is not None:
        classes, counts = np.unique(labels[pool], return_counts=True)
        # stratify only if every class has enough members to appear on both sides
        if len(classes) > 1 and counts.min() >= 2:
            stratify = labels[pool]
    try:
        train, val = train_test_split(
            pool, test_size=val_fraction, random_state=seed, stratify=stratify
        )
    except ValueError:
        # val partition too small to carry every class; fall back to a plain split
        train, val = train_test_split(
            pool, test_size=val_fraction, random_state=seed, stratify=None
        )
    return np.sort(train), np.sort(val)


def _make_split(train, val, test, meta: dict) -> "Split":
    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    return Split(train, val, test, {**meta, "sizes": sizes})


def _check_disjoint(split: Split) -> None:
    """Raise if train/val/test share any trial index (applies to every protocol)."""
    tr, va, te = split.train, split.val, split.test
    for a, b in ((tr, te), (va, te), (tr, va)):
        if np.intersect1d(a, b).size:
            raise ValueError(f"overlapping trial indices in split {split.meta}")


def subject_specific(
    subjects, test_mask, labels=None, *, seed: int = 0, val_fraction: float = 0.1
) -> list[Split]:
    """One Split per subject: test is that subject's test_mask trials, train/val its remaining trials."""
    subjects = np.asarray(subjects)
    test_mask = np.asarray(test_mask, dtype=bool)
    labels = None if labels is None else np.asarray(labels)

    splits: list[Split] = []
    for subject in np.unique(subjects):
        in_subject = subjects == subject
        test = np.nonzero(in_subject & test_mask)[0]
        pool = np.nonzero(in_subject & ~test_mask)[0]
        train, val = _carve_val(pool, labels, val_fraction, seed)
        split = _make_split(
            train,
            val,
            np.sort(test),
            {"protocol": "subject_specific", "subject": int(subject), "seed": seed},
        )
        # every index in this split must belong to exactly this subject
        used = np.concatenate([train, val, test]) if (len(train) + len(val) + len(test)) else np.array([], dtype=np.int64)
        if used.size and set(subjects[used].tolist()) != {int(subject)}:
            raise ValueError(f"subject_specific split for {subject} mixed other subjects")
        _check_disjoint(split)
        splits.append(split)
    return splits
