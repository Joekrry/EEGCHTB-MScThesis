"""Cross-model interface parity (Gate 4).

Certifies that EEGNet, EEG-Conformer and GRF-Net all run through the one shared
trainer and result/manifest contract, so a benchmark comparison differs only by
architecture and config, not by hidden interface behaviour: identical output files,
identical result/metric schema, and a (batch, n_classes) forward for the
representative 2a and 2b tensor shapes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from conftest import requires_torch
from test_experiment import _FakeLoader, _config

MODEL_NAMES = ("eegnet", "eeg_conformer", "grf_net")
RESULT_KEYS = {
    "run_id", "dataset", "model", "preprocessing", "protocol", "seed", "fold",
    "metrics", "n_trials", "timings", "config_fingerprint", "created_at",
}
RUN_FILES = ("result.json", "manifest.json", "history.json", ".run_complete")


@requires_torch()
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_model_runs_through_common_contract(model, tmp_path):
    from eegchtb.experiment import run_experiment

    config = dataclasses.replace(_config(), model=model)
    path = run_experiment(config, tmp_path, loader=_FakeLoader(), run_id=f"parity_{model}")
    for fname in RUN_FILES:
        assert (path / fname).exists(), f"{model} missing {fname}"
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    assert set(result) == RESULT_KEYS
    assert result["model"] == model


@requires_torch()
def test_all_models_share_result_and_metric_schema(tmp_path):
    from eegchtb.experiment import run_experiment

    schemas = []
    for model in MODEL_NAMES:
        path = run_experiment(dataclasses.replace(_config(), model=model), tmp_path,
                              loader=_FakeLoader(), run_id=f"schema_{model}")
        r = json.loads((path / "result.json").read_text(encoding="utf-8"))
        schemas.append((set(r), set(r["metrics"]), set(r["n_trials"])))
    # every model produces the same result keys, metric names and n_trials partitions
    assert all(s == schemas[0] for s in schemas), "models diverge in result/metric schema"


@requires_torch()
@pytest.mark.parametrize("n_channels,n_times,n_classes", [(22, 1000, 4), (3, 1000, 2)])
def test_forward_shape_parity(n_channels, n_times, n_classes):
    import torch

    from eegchtb.experiment import MODELS, ExperimentConfig, _model_kwargs

    meta = {
        "sfreq": 250.0,
        "channel_names": [f"C{i}" for i in range(n_channels)],
        "class_names": {i: str(i) for i in range(n_classes)},
    }
    for model in MODEL_NAMES:
        config = ExperimentConfig(dataset="x", preprocessing="none",
                                  protocol="subject_specific", model=model)
        kwargs = _model_kwargs(config, meta, n_channels, n_times, n_classes)
        net = MODELS[model](**kwargs)
        out = net(torch.zeros(2, n_channels, n_times))
        assert out.shape == (2, n_classes), f"{model} produced {tuple(out.shape)}"
