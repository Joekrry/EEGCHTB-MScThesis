"""Tests for the preprocessing framework (behaviour tests gated on MNE)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_mne

from eegchtb.preprocessing import (
    STEP_REGISTRY,
    Pipeline,
    PipelineConfig,
    build_step,
    register_step,
)
from eegchtb.preprocessing.steps import PreprocessingStep


# pure logic
def test_registry_has_core_steps():
    expected = {
        "bandpass",
        "resample",
        "rereference",
        "standardize",
        "hjorth_laplacian",
        "csd",
    }
    assert expected <= set(STEP_REGISTRY)


def test_build_step_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        build_step("does_not_exist", {})


def test_build_step_instantiates_with_params():
    step = build_step("bandpass", {"l_freq": 4.0, "h_freq": 40.0})
    assert step.name == "bandpass"
    params = step.params()
    assert params["l_freq"] == 4.0
    assert params["h_freq"] == 40.0
    # the FIR design is now declared and recorded, not inherited from MNE
    assert params["fir_design"] == "firwin"
    assert params["phase"] == "zero"
    assert params["fir_window"] == "hamming"


def test_resolved_config_captures_design_defaults():
    pipe = Pipeline.from_dict(
        {
            "name": "m",
            "steps": [
                {"name": "bandpass", "params": {"l_freq": 8.0, "h_freq": 30.0}},
                {"name": "hjorth_laplacian", "params": {"k": 4}},
            ],
        }
    )
    resolved = pipe.resolved_config()
    bandpass = resolved["steps"][0]["params"]
    # the YAML only gave l_freq/h_freq; the resolved spec pins the FIR design too
    assert bandpass["l_freq"] == 8.0
    assert bandpass["fir_design"] == "firwin"
    assert bandpass["phase"] == "zero"
    # the spatial step records its full effective params so a change to k or the
    # weighting scheme changes the fingerprint
    hjorth = resolved["steps"][1]
    assert hjorth["name"] == "hjorth_laplacian"
    assert hjorth["params"]["k"] == 4
    assert hjorth["params"]["weighting"] == "inverse_distance"
    assert hjorth["params"]["montage"] == "standard_1020"


def test_pipeline_config_roundtrips_through_dict():
    data = {
        "name": "standard",
        "steps": [
            {"name": "bandpass", "params": {"l_freq": 4.0, "h_freq": 40.0}},
            {"name": "resample", "params": {"sfreq": 250.0}},
        ],
    }
    config = PipelineConfig.from_dict(data)
    assert config.name == "standard"
    assert [s.name for s in config.steps] == ["bandpass", "resample"]
    assert config.to_dict() == data


def test_pipeline_builds_steps_in_order_and_fails_fast():
    pipe = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "bandpass"}, {"name": "resample"}]}
    )
    assert [s.name for s in pipe.steps] == ["bandpass", "resample"]
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        Pipeline.from_dict({"name": "p", "steps": [{"name": "nope"}]})


def test_register_step_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        @register_step
        class _Dup(PreprocessingStep):
            name = "bandpass"

            def apply(self, raw):
                return raw


def test_from_yaml_loads_a_pipeline(tmp_path):
    yaml_text = (
        "name: from_file\n"
        "steps:\n"
        "  - name: bandpass\n"
        "    params:\n"
        "      l_freq: 1.0\n"
        "      h_freq: 40.0\n"
        "  - name: resample\n"
        "    params:\n"
        "      sfreq: 128.0\n"
    )
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    pipe = Pipeline.from_yaml(path)
    assert pipe.name == "from_file"
    assert [s.name for s in pipe.steps] == ["bandpass", "resample"]


# behaviour
def _sine_raw(sfreq=250.0, seconds=20, freqs=(10.0, 60.0)):
    """A 3-channel synthetic Raw that is a sum of sines at the given freqs."""
    import mne

    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    data = np.tile(signal, (3, 1))
    info = mne.create_info(["C3", "Cz", "C4"], sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _band_power(data, sfreq, centre, width=2.0):
    freqs = np.fft.rfftfreq(data.shape[1], 1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(data, axis=1))
    mask = (freqs >= centre - width) & (freqs <= centre + width)
    return float(spectrum[:, mask].mean())


@requires_mne()
def test_resample_step_changes_sampling_rate():
    raw = _sine_raw()
    pipe = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "resample", "params": {"sfreq": 125.0}}]}
    )
    out = pipe.apply(raw)
    assert out.info["sfreq"] == 125.0


@requires_mne()
def test_bandpass_attenuates_out_of_band_signal():
    raw = _sine_raw(freqs=(10.0, 60.0))
    p10_in = _band_power(raw.get_data(), 250.0, 10.0)
    p60_in = _band_power(raw.get_data(), 250.0, 60.0)

    pipe = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "bandpass", "params": {"l_freq": 1.0, "h_freq": 40.0}}]}
    )
    out = pipe.apply(raw)

    p10_out = _band_power(out.get_data(), 250.0, 10.0)
    p60_out = _band_power(out.get_data(), 250.0, 60.0)

    # the 60 Hz component (above the 40 Hz cutoff) is strongly attenuated,
    # while the 10 Hz component (in the passband) is largely preserved
    assert p60_out < 0.2 * p60_in
    assert p10_out > 0.5 * p10_in


@requires_mne()
def test_apply_copies_by_default_and_leaves_input_untouched():
    raw = _sine_raw()
    original_sfreq = raw.info["sfreq"]
    pipe = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "resample", "params": {"sfreq": 125.0}}]}
    )
    out = pipe.apply(raw)
    assert raw.info["sfreq"] == original_sfreq  # input unchanged
    assert out.info["sfreq"] == 125.0


@requires_mne()
def test_pipeline_is_deterministic():
    raw = _sine_raw()
    pipe = Pipeline.from_dict(
        {
            "name": "p",
            "steps": [
                {"name": "bandpass", "params": {"l_freq": 4.0, "h_freq": 40.0}},
                {"name": "resample", "params": {"sfreq": 128.0}},
            ],
        }
    )
    first = pipe.apply(raw).get_data()
    second = pipe.apply(raw).get_data()
    np.testing.assert_array_equal(first, second)


@requires_mne()
def test_standardize_normalises_each_channel():
    import mne

    rng = np.random.default_rng(0)
    # three channels with different offsets and scales
    data = np.vstack(
        [
            5.0 + 2.0 * rng.standard_normal(2000),
            -3.0 + 0.5 * rng.standard_normal(2000),
            100.0 + 10.0 * rng.standard_normal(2000),
        ]
    )
    info = mne.create_info(["a", "b", "c"], 250.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    out = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "standardize"}]}
    ).apply(raw)

    result = np.asarray(out.get_data())
    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(result.std(axis=1), 1.0, rtol=1e-3)


@requires_mne()
def test_rereference_runs_and_preserves_shape():
    raw = _sine_raw()
    pipe = Pipeline.from_dict(
        {"name": "p", "steps": [{"name": "rereference", "params": {"ref_channels": "average"}}]}
    )
    out = pipe.apply(raw)
    # average reference changes values, not the (channels, samples) shape
    assert len(out.ch_names) == len(raw.ch_names)
    assert out.n_times == raw.n_times


@requires_mne()
def test_csd_runs_and_preserves_channels_and_samples():
    import mne

    # a full 10-20 montage so the surface Laplacian is well defined
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    ]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ch_names), 2000)) * 1e-5  # volts
    info = mne.create_info(ch_names, 250.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    out = Pipeline.from_dict({"name": "p", "steps": [{"name": "csd"}]}).apply(raw)

    # CSD transforms the values but keeps the same channels and samples
    assert len(out.ch_names) == len(ch_names)
    assert out.n_times == raw.n_times
    assert not np.array_equal(np.asarray(out.get_data()), data)
