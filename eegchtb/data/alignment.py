"""Euclidean Alignment (EA) for cross-session/-subject EEG transfer.

Reference: H. He and D. Wu, "Transfer Learning for Brain-Computer Interfaces: A
Euclidean Space Data Alignment Approach", IEEE TBME 67(2), 2020; P. Zanini et al.,
"Transfer Learning: A Riemannian Geometry Framework...", IEEE TBME 65(5), 2018.
"""

from __future__ import annotations

import numpy as np


def euclidean_reference_inv_sqrt(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """R^{-1/2} for a set of trials x (n_trials, n_channels, n_times): the inverse square root of the mean per-trial spatial covariance."""
    if x.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times); got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("expected at least one trial; got an empty set")

    n_times = x.shape[2]
    covs = np.einsum("nct,ndt->ncd", x, x) / n_times
    R = covs.mean(axis=0)

    evals, evecs = np.linalg.eigh(R)
    evals = np.clip(evals, eps, None)
    inv_sqrt_evals = 1.0 / np.sqrt(evals)
    return (evecs * inv_sqrt_evals) @ evecs.T


def euclidean_align(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Align one set of trials by its own Euclidean reference: x -> R^{-1/2} x, giving an identity mean covariance."""
    return apply_reference(x, euclidean_reference_inv_sqrt(x, eps))


def apply_reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Whiten trials x by a reference computed elsewhere, for applying a training set's R^{-1/2} to held-out trials."""
    return np.einsum("cd,ndt->nct", w, x)


def align_sessions(x: np.ndarray, session_ids, eps: float = 1e-6) -> np.ndarray:
    """Align each recording independently: trials sharing a session id are whitened by that session's own reference, removing the between-session shift."""
    session_ids = np.asarray(session_ids)
    if len(session_ids) != len(x):
        raise ValueError(f"session_ids ({len(session_ids)}) must match trials ({len(x)})")

    # Aligned trials are always floating point, so do not inherit an integer input dtype.
    out = np.empty(x.shape, dtype=np.result_type(x.dtype, np.float32))
    for sid in np.unique(session_ids):
        mask = session_ids == sid
        out[mask] = euclidean_align(x[mask], eps)
    return out
