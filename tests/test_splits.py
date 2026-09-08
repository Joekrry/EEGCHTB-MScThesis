"""Tests for leakage-safe subject-aware splits."""

from __future__ import annotations

import numpy as np

from eegchtb.evaluation.splits import subject_specific


def _cohort(n_subjects=3, per_subject=8, n_classes=2):
    """Per-trial (subjects, labels) for a synthetic balanced cohort."""
    subjects = np.repeat(np.arange(1, n_subjects + 1), per_subject)
    labels = np.tile(np.arange(n_classes), per_subject // n_classes * n_subjects)
    return subjects, labels


def _with_test_mask(subjects, n_train_per_subject=4):
    """Mark each subject's trials after the first n as its evaluation session."""
    test_mask = np.zeros(len(subjects), dtype=bool)
    for s in np.unique(subjects):
        idx = np.nonzero(subjects == s)[0]
        test_mask[idx[n_train_per_subject:]] = True
    return test_mask


def test_subject_specific_stays_within_subject():
    subjects, labels = _cohort(n_subjects=2, per_subject=8)
    test_mask = _with_test_mask(subjects)

    for s in subject_specific(subjects, test_mask, labels, seed=0, val_fraction=0.25):
        used = np.concatenate([s.train, s.val, s.test])
        assert set(subjects[used].tolist()) == {s.meta["subject"]}
        assert len(np.intersect1d(s.train, s.test)) == 0


def test_subject_specific_covers_every_trial_of_its_subject():
    subjects, labels = _cohort(n_subjects=3, per_subject=8)
    test_mask = _with_test_mask(subjects)
    splits = subject_specific(subjects, test_mask, labels, seed=0, val_fraction=0.25)
    assert len(splits) == 3  # one split per subject
    for s in splits:
        union = np.concatenate([s.train, s.val, s.test])
        expected = set(np.nonzero(subjects == s.meta["subject"])[0].tolist())
        assert set(union.tolist()) == expected  # every trial of the subject used once
        assert len(np.intersect1d(s.val, s.test)) == 0


def test_subject_specific_is_deterministic_for_a_fixed_seed():
    subjects, labels = _cohort(n_subjects=4, per_subject=10)
    test_mask = _with_test_mask(subjects, n_train_per_subject=6)
    a = subject_specific(subjects, test_mask, labels, seed=7, val_fraction=0.2)
    b = subject_specific(subjects, test_mask, labels, seed=7, val_fraction=0.2)
    for x, y in zip(a, b):
        assert np.array_equal(x.train, y.train)
        assert np.array_equal(x.val, y.val)
        assert np.array_equal(x.test, y.test)
