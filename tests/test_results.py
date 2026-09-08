"""Tests for the run-result schema and config fingerprints."""

from __future__ import annotations

from eegchtb.evaluation.results import RunResult, config_fingerprint


def _sample_result() -> RunResult:
    config = {"dataset": "bci_iv_2a", "model": "eegnet", "preprocessing": "standard", "seed": 0}
    return RunResult(
        run_id="run-1",
        dataset="bci_iv_2a",
        model="eegnet",
        preprocessing="standard",
        protocol="subject_specific",
        seed=0,
        fold="S1",
        metrics={"accuracy": 0.8, "macro_f1": 0.79, "cohen_kappa": 0.73, "mcc": 0.74},
        n_trials={"train": 2000, "val": 200, "test": 288},
        timings={"train_seconds": 12.5, "eval_seconds": 0.4, "wall_seconds": 13.1},
        config_fingerprint=config_fingerprint(config),
    )


def test_json_roundtrip():
    r = _sample_result()
    assert RunResult.from_json(r.to_json()) == r


def test_fingerprint_is_deterministic_and_order_independent():
    a = config_fingerprint({"model": "eegnet", "seed": 0})
    b = config_fingerprint({"seed": 0, "model": "eegnet"})
    assert a == b
    assert a != config_fingerprint({"model": "eegnet", "seed": 1})
