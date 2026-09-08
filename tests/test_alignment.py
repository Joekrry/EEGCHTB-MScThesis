"""Euclidean Alignment: aligned trials have an identity mean covariance, and sessions are aligned independently."""

from __future__ import annotations

import numpy as np

from eegchtb.data.alignment import align_sessions, euclidean_align


def _mean_cov(x):
    return (np.einsum("nct,ndt->ncd", x, x) / x.shape[2]).mean(axis=0)


def test_align_gives_identity_mean_covariance():
    rng = np.random.default_rng(0)
    # correlated channels via a random mixing matrix -> non-identity covariance
    src = rng.standard_normal((60, 4, 200))
    mix = rng.standard_normal((4, 4))
    x = np.einsum("cd,ndt->nct", mix, src)
    aligned = euclidean_align(x)
    assert np.allclose(_mean_cov(aligned), np.eye(4), atol=1e-6)


def test_align_sessions_is_per_session():
    rng = np.random.default_rng(1)
    a = np.einsum("cd,ndt->nct", rng.standard_normal((3, 3)), rng.standard_normal((30, 3, 100)))
    b = np.einsum("cd,ndt->nct", rng.standard_normal((3, 3)), rng.standard_normal((30, 3, 100)))
    x = np.concatenate([a, b])
    sessions = np.array([0] * 30 + [1] * 30)
    out = align_sessions(x, sessions)
    # each session, aligned by its own reference, is individually whitened
    assert np.allclose(_mean_cov(out[:30]), np.eye(3), atol=1e-6)
    assert np.allclose(_mean_cov(out[30:]), np.eye(3), atol=1e-6)


def test_align_sessions_validates_length():
    x = np.zeros((5, 2, 10))
    try:
        align_sessions(x, [0, 1])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched session_ids length")
