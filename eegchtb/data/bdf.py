"""BDF loader for the Galea headset's own recordings; mne imported lazily.

Mirrors the raw-loading and preprocessing-application half of gdf.py
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import events as ev
from .base import BaseEEGLoader, EpochWindow
from .channels import GALEA_BCI_IV_2A_OVERLAP_CHANNELS, select_channels
from .epochs import STANDARDIZED_UNITS, VALID_UNITS, EpochBundle
from ..preprocessing import Pipeline

MARKER_RE = re.compile(r"marker\s*=\s*(\d+)$", re.IGNORECASE)


class GaleaBdfLoader(BaseEEGLoader):
    """Loader for Galea headset BDF recordings, windowed into MI-class trials."""

    name = "galea_bdf"
    expected_sfreq: float = 250.0
    cue_to_label = ev.GALEA_CUE_TO_LABEL
    class_names = ev.BCI_IV_2A_CLASSES  # same 4 MI classes, same label ids

    def __init__(
        self,
        root: str | Path,
        window: EpochWindow | None = None,
        preprocessing: Pipeline | dict | str | Path | None = None,
        cache_dir: str | Path | None = None,
        channels: Sequence[str] | None = GALEA_BCI_IV_2A_OVERLAP_CHANNELS,
    ) -> None:
        self.root = Path(root)
        self.window = window or EpochWindow()
        self.preprocessing = self._coerce_pipeline(preprocessing)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Channel subset kept from every recording, matched by name. Defaults to
        # C3/Cz/C4/Pz the only channels the grid's trained snapshots actually
        # use is BCI IV 2a's montage overlaps Galea on just those 4
        self.channels = list(channels) if channels is not None else None

    def _recordings_dir(self) -> Path:
        """The dataset's own subdir if root points at a shared data dir, else root."""
        candidate = self.root / "galea_recordings_joe"
        return candidate if candidate.exists() else self.root

    def _recording_paths(self) -> list[Path]:
        return sorted(self._recordings_dir().glob("*.bdf"))

    def _bdf_path(self, session: str) -> Path:
        return self._recordings_dir() / f"{session}.bdf"

    def available_subjects(self) -> list[int]:
        return [1] if self._recording_paths() else []

    def available_sessions(self, subject: int) -> list[str]:
        return [p.stem for p in self._recording_paths()]

    def load_raw(self, session: str):
        """Read a recording as a continuous mne.Raw, restricted to the configured channels."""
        import mne

        path = self._bdf_path(session)
        if not path.exists():
            raise FileNotFoundError(f"BDF file not found: {path}")

        raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
        self._select_channels(raw)
        return raw

    def _select_channels(self, raw) -> None:
        if self.channels is None:
            return
        picks = select_channels(raw.ch_names, self.channels)
        raw.pick(picks)

    def load_preprocessed(self, session: str) -> tuple[Any, dict | None]:
        """Read a recording, crop it to the experimental period, and run the Pipeline."""
        raw = self._crop_to_markers(self.load_raw(session))
        return self._apply_preprocessing(raw)

    def _crop_to_markers(self, raw):
        """Trim the recording to the marked experimental period, in place.
        """
        markers = self._parse_markers(raw)
        if not markers:
            return raw

        onsets = [onset for onset, _ in markers]
        tmin = max(0.0, min(onsets) + self.window.tmin)
        # crop is inclusive of tmax, but the last trial needs the sample *after*
        # its window too, or epochs_from_raw drops it as overrunning the edge
        last_sample = max(onsets) + self.window.tmax + 1.0 / float(raw.info["sfreq"])
        tmax = min(float(raw.times[-1]), last_sample)
        if tmax <= tmin:
            return raw

        raw.crop(tmin=tmin, tmax=tmax)
        return raw

    def load_session(self, subject: int, session: str) -> EpochBundle:
        """Load, crop, preprocess, and window one recording's MI blocks into an EpochBundle."""
        raw = self._crop_to_markers(self.load_raw(session))
        raw, prep_metadata = self._apply_preprocessing(raw)
        bundle = self.epochs_from_raw(raw, subject, session, output_units=self._output_units())
        if prep_metadata:
            bundle.metadata["preprocessing"] = prep_metadata
        return bundle

    def _output_units(self) -> str:
        pipeline = self.preprocessing
        if pipeline is not None and any(s.name == "standardize" for s in pipeline.steps):
            return STANDARDIZED_UNITS
        return self.window.units

    def _parse_markers(self, raw) -> list[tuple[float, int]]:
        """(onset_seconds, code) for every RAW1:Marker=N annotation, in file order.

        Onsets are returned relative to the start of raw's data array. Annotation
        onsets stay in the original recording's time frame across a crop, so
        first_time has to come off or every window is cut late by the amount
        trimmed from the head.
        """
        offset = float(raw.first_time)
        markers = []
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            match = MARKER_RE.search(str(desc))
            if match is not None:
                markers.append((float(onset) - offset, int(match.group(1))))
        markers.sort(key=lambda m: m[0])
        return markers

    def epochs_from_raw(
        self, raw, subject: int, session: str, output_units: str | None = None
    ) -> EpochBundle:
        """Cut one cue-locked [tmin, tmax] trial window from each MI marker.

        The protocol writes a marker at cue onset and defines the trial as the
        [0, 4] s window after it, exactly as in 2a/2b. Inter-marker time is rest,
        instruction and inter-run breaks, so it is never windowed into extra
        trials: one marker is one trial.
        """
        sfreq = float(raw.info["sfreq"])
        channel_names = list(raw.ch_names)
        n_times = self.window.n_times(sfreq)
        data_v = raw.get_data()  # (n_channels, n_samples), volts
        n_samples = data_v.shape[1]

        windows = []
        labels = []
        trial_ids = []
        dropped = 0
        for onset, code in self._parse_markers(raw):
            if code not in self.cue_to_label:
                continue  # baseline block (idle eyes/eye movement) or unknown code
            start = int(round((onset + self.window.tmin) * sfreq))
            if start < 0 or start + n_times > n_samples:
                dropped += 1  # cue too close to an edge to hold a whole window
                continue
            windows.append(data_v[:, start : start + n_times])
            labels.append(self.cue_to_label[code])
            trial_ids.append(len(trial_ids))

        if windows:
            data = np.stack(windows, axis=0)
        else:
            data = np.zeros((0, len(channel_names), n_times), dtype=np.float64)

        units = output_units or self.window.units
        if units == "uV":
            data = data * 1e6
        elif units not in VALID_UNITS:
            raise ValueError(f"output_units must be one of {VALID_UNITS}; got {units!r}")

        filenames = getattr(raw, "filenames", None)
        source_file = str(filenames[0]) if filenames and filenames[0] is not None else None

        return EpochBundle(
            data=data,
            labels=np.array(labels, dtype=np.int64),
            dataset=self.name,
            subject=subject,
            session=session,
            sfreq=sfreq,
            channel_names=channel_names,
            class_names=dict(self.class_names),
            times=np.arange(n_times, dtype=np.float64) / sfreq + self.window.tmin,
            trial_ids=np.array(trial_ids, dtype=np.int64),
            units=units,
            metadata={
                "source_file": source_file,
                # cues that could not hold a whole window; a silent drop here would
                # quietly shrink an already small evaluation set
                "n_cues_dropped": dropped,
            },
        )

    def _coerce_pipeline(
        self, preprocessing: Pipeline | dict | str | Path | None
    ) -> Pipeline | None:
        if preprocessing is None or isinstance(preprocessing, Pipeline):
            return preprocessing
        if isinstance(preprocessing, dict):
            return Pipeline.from_dict(preprocessing)

        path = Path(preprocessing)
        if path.suffix:
            return Pipeline.from_yaml(path)

        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "preprocessing"
            / f"{preprocessing}.yaml"
        )
        return Pipeline.from_yaml(config_path)

    def _apply_preprocessing(self, raw) -> tuple[Any, dict | None]:
        pipeline = self.preprocessing
        if pipeline is None:
            return raw, None

        working = raw.copy()
        applied_steps: list[str] = []
        resolved_steps: list[dict] = []
        for step in pipeline.steps:
            resolved = step.resolved_provenance(working)
            if resolved is not None:
                resolved_steps.append({"name": step.name, "resolved": resolved})
            working = step.apply(working)
            applied_steps.append(step.name)

        metadata = {
            "name": pipeline.name,
            "config": pipeline.resolved_config(),
            "applied_steps": applied_steps,
        }
        if resolved_steps:
            metadata["resolved_steps"] = resolved_steps
        return working, metadata
