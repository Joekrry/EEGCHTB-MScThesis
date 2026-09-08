"""Shared MNE/GDF base loader for BCI Competition IV 2a/2b; mne and scipy imported lazily."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import events as ev
from .base import BaseEEGLoader, EpochWindow
from .channels import GALEA_EEG_CHANNELS, select_channels
from .epochs import STANDARDIZED_UNITS, VALID_UNITS, EpochBundle
from ..preprocessing import Pipeline


def _safe_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


class GdfMotorImageryLoader(BaseEEGLoader, abstract=True):
    """Base class for GDF-backed motor-imagery loaders (IV-2a / IV-2b)."""

    # subclass-provided constants
    expected_sfreq: float = 250.0
    cue_to_label: dict[int, int] = {}
    class_names: dict[int, str] = {}
    # window (s) around a cue used to detect a co-occurring rejected-trial mark
    reject_pre_s: float = 4.0
    # EEG channel names in file order; if None, the "EEG:"/"EEG-"
    # prefix is stripped instead. Set by subclasses whose GDF stores channels as
    # numbered placeholders (IV-2a) so a montage and the spatial steps are well
    # defined.
    eeg_channel_names: list[str] | None = None
    # Name of the folder the official BCI Competition IV archive unzips into
    # (e.g. "BCICIV_2a_gdf"), tried as an extra layout candidate in _resolve so
    # a shared data_root works whether or not the archive was renamed/flattened.
    archive_dir_name: str | None = None
    # Letter the competition prefixes its filenames with ("A" for IV-2a, "B" for
    # IV-2b), e.g. A01T.gdf / B0101T.gdf.
    subject_prefix: str = ""
    # Subject numbers and session suffixes the dataset publishes.
    SUBJECTS: Sequence[int] = ()
    SESSIONS: Sequence[str] = ()

    def __init__(
        self,
        root: str | Path,
        window: EpochWindow | None = None,
        drop_eog: bool = True,
        preprocessing: Pipeline | dict | str | Path | None = None,
        cache_dir: str | Path | None = None,
        labels_dir: str | Path | None = None,
        channels: Sequence[str] | None = GALEA_EEG_CHANNELS,
    ) -> None:
        self.root = Path(root)
        self.window = window or EpochWindow()
        self.drop_eog = drop_eog
        self.preprocessing = self._coerce_pipeline(preprocessing)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Channel subset kept from every session, matched by name. Defaults to the
        # 10 Galea channels so results extrapolate to the Galea headset; pass None
        # to keep all EEG channels.
        self.channels = list(channels) if channels is not None else None
        # the released true-label .mat files, alongside the GDFs unless overridden
        self.labels_dir = Path(labels_dir) if labels_dir is not None else self.root

    TRUE_LABELS_SUBDIR = "truelabels"

    @staticmethod
    def _session_subdir(session: str) -> str:
        """Evaluation session names end in E; every other session is training."""
        return "test" if session.endswith("E") else "train"

    def _resolve(self, base: Path, subdir: str, filename: str) -> Path:
        """First layout candidate that exists, else the flat path so errors name it."""
        candidates = [
            base / self.name / subdir / filename,  # base is the shared data dir
            base / subdir / filename,  # base is this dataset's own dir
            base / filename,  # flat, as the archives unzip
        ]
        if self.archive_dir_name is not None:
            candidates += [
                # base is the shared data dir holding the unrenamed archive folder
                base / self.archive_dir_name / subdir / filename,
                base / self.archive_dir_name / filename,
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    # <prefix><subject:02d><session>
    def _stem(self, subject: int, session: str) -> str:
        return f"{self.subject_prefix}{subject:02d}{session}"

    def _gdf_path(self, subject: int, session: str) -> Path:
        return self._resolve(
            self.root, self._session_subdir(session), f"{self._stem(subject, session)}.gdf"
        )

    def _true_labels_path(self, subject: int, session: str) -> Path | None:
        """.mat true-label file for an eval session, or None if labels are in the GDF."""
        if not session.endswith("E"):
            return None  # training cues are already labelled in the GDF
        return self._resolve(
            self.labels_dir, self.TRUE_LABELS_SUBDIR, f"{self._stem(subject, session)}.mat"
        )

    def available_subjects(self) -> list[int]:
        return [
            s for s in self.SUBJECTS
            if any(self._gdf_path(s, sess).exists() for sess in self.SESSIONS)
        ]

    def available_sessions(self, subject: int) -> list[str]:
        return [s for s in self.SESSIONS if self._gdf_path(subject, s).exists()]

    def load_raw(self, subject: int, session: str):
        """Read a session as a continuous mne.Raw, with cue events left on as annotations."""
        import mne

        path = self._gdf_path(subject, session)
        if not path.exists():
            raise FileNotFoundError(f"GDF file not found: {path}")

        raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
        if self.drop_eog:
            eog = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
            if eog:
                raw.drop_channels(eog)
        self._rename_channels(raw)
        self._select_channels(raw)
        return raw

    def _select_channels(self, raw) -> None:
        """Restrict to the configured channel subset (Galea by default), matched by name."""
        if self.channels is None:
            return
        picks = select_channels(raw.ch_names, self.channels)
        raw.pick(picks)

    @staticmethod
    def _strip_eeg_prefix(name: str) -> str:
        for prefix in ("EEG:", "EEG-", "EEG "):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _rename_channels(self, raw) -> None:
        """Normalise GDF channel names to canonical 10-20 labels (rename IV-2a's numbered channels by position, else strip the EEG prefix)."""
        names = self.eeg_channel_names
        if names is None:
            raw.rename_channels({c: self._strip_eeg_prefix(c) for c in raw.ch_names})
            return
        eeg = [c for c in raw.ch_names if "EOG" not in c.upper()]
        if len(eeg) != len(names):
            raise ValueError(
                f"{self.name}: expected {len(names)} EEG channels to rename, "
                f"got {len(eeg)}: {eeg}"
            )
        mapping = dict(zip(eeg, names))
        for c in raw.ch_names:
            mapping.setdefault(c, self._strip_eeg_prefix(c))
        raw.rename_channels(mapping)

    def load_session(self, subject: int, session: str) -> EpochBundle:
        """Load, optionally preprocess, and epoch one session."""
        cache_path = self._cache_path(subject, session)
        if cache_path is not None and cache_path.exists():
            return EpochBundle.load(cache_path)

        raw = self.load_raw(subject, session)
        raw, prep_metadata = self._apply_preprocessing(raw)
        units = self._output_units()
        bundle = self.epochs_from_raw(raw, subject, session, output_units=units)
        if prep_metadata:
            bundle.metadata["preprocessing"] = prep_metadata

        if cache_path is not None:
            bundle.save(cache_path)
        return bundle

    def epochs_from_raw(
        self,
        raw,
        subject: int,
        session: str,
        output_units: str | None = None,
    ) -> EpochBundle:
        """Cut a continuous Raw into an EpochBundle at the Raw's current sampling rate."""
        import mne

        sfreq = float(raw.info["sfreq"])
        channel_names = list(raw.ch_names)
        units = output_units or self.window.units

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

        # MNE assigns its own integer ids; map them back to GDF codes.
        mne_id_to_code = {mid: _safe_int(desc) for desc, mid in event_id.items()}
        cue_codes = set(self.cue_to_label) | {ev.EVENT_CUE_UNKNOWN}
        cue_event_id = {}
        for desc, mid in event_id.items():
            if _safe_int(desc) in cue_codes:
                cue_event_id[desc] = mid
        if not cue_event_id:
            raise ValueError(
                f"No cue events found for S{subject} {session}; "
                f"annotations present: {sorted(event_id)}"
            )

        # Ordinal trial index (position among cues) for each original event row.
        cue_ordinal: dict[int, int] = {}
        ordinal = 0
        for row, ev_row in enumerate(events):
            if mne_id_to_code.get(int(ev_row[2])) in cue_codes:
                cue_ordinal[row] = ordinal
                ordinal += 1

        epochs = mne.Epochs(
            raw,
            events,
            event_id=cue_event_id,
            tmin=self.window.tmin,
            tmax=self.window.tmax,
            baseline=self.window.baseline,
            preload=True,
            reject=None,
            on_missing="ignore",
            verbose="ERROR",
        )

        data = epochs.get_data(copy=False)  # (n_trials, n_channels, n_times), volts
        n_times = self.window.n_times(sfreq)
        data = data[:, :, :n_times]
        if units == "uV":
            data = data * 1e6
        elif units not in VALID_UNITS:
            raise ValueError(f"output_units must be one of {VALID_UNITS}; got {units!r}")

        kept_codes = []
        for c in epochs.events[:, 2]:
            kept_codes.append(mne_id_to_code.get(int(c)))
            
        trial_id_list = []
        for s in epochs.selection:
            trial_id_list.append(cue_ordinal[int(s)])
        trial_ids = np.array(trial_id_list, dtype=np.int64)

        labels = self._assign_labels(
            subject, session, kept_codes, trial_ids, n_total_cues=ordinal
        )

        # provenance: the source file if this Raw still knows it (a preprocessed
        # Raw usually does), otherwise None
        filenames = getattr(raw, "filenames", None)
        source_file = None
        if filenames and filenames[0] is not None:
            source_file = str(filenames[0])

        bundle = EpochBundle(
            data=data,
            labels=labels,
            dataset=self.name,
            subject=subject,
            session=session,
            sfreq=sfreq,
            channel_names=channel_names,
            class_names=dict(self.class_names),
            times=epochs.times[:n_times].copy(),
            trial_ids=trial_ids,
            units=units,
            metadata={"source_file": source_file},
        )

        if self.window.drop_rejected:
            keep = self._rejected_keep_mask(events, mne_id_to_code, epochs, sfreq)
            bundle = bundle.select_trials(keep)

        return bundle

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
            # some steps resolve montage-dependent detail at run time (e.g. the
            # Hjorth Laplacian's neighbour graph); capture it before applying
            resolved = step.resolved_provenance(working)
            if resolved is not None:
                resolved_steps.append({"name": step.name, "resolved": resolved})
            working = step.apply(working)
            applied_steps.append(step.name)

        metadata = {
            "name": pipeline.name,
            "config": pipeline.resolved_config(),
            "applied_steps": applied_steps,
            "output_units": self._output_units(),
        }
        if resolved_steps:
            metadata["resolved_steps"] = resolved_steps
        return working, metadata

    def _output_units(self) -> str:
        pipeline = self.preprocessing
        if pipeline is not None and any(s.name == "standardize" for s in pipeline.steps):
            return STANDARDIZED_UNITS
        return self.window.units

    def _cache_path(self, subject: int, session: str) -> Path | None:
        if self.cache_dir is None:
            return None

        key = {
            "dataset": self.name,
            "subject": subject,
            "session": session,
            "window": asdict(self.window),
            "drop_eog": self.drop_eog,
            "channels": self.channels,
            "preprocessing": (
                None
                if self.preprocessing is None
                else self.preprocessing.resolved_config()
            ),
        }
        encoded = json.dumps(key, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        safe_session = session.replace("/", "_").replace("\\", "_")
        filename = f"subject-{subject:02d}_{safe_session}_{digest}.npz"
        return self.cache_dir / self.name / filename

    def _assign_labels(
        self,
        subject: int,
        session: str,
        kept_codes: Sequence[int | None],
        trial_ids: np.ndarray,
        n_total_cues: int,
    ) -> np.ndarray:
        """Assign a 0-based label to each kept epoch (training cues directly, hidden eval cues from the .mat)."""
        label_list = []
        for code in kept_codes:
            if code is not None:
                label_list.append(self.cue_to_label.get(code, ev.UNKNOWN_LABEL))
            else:
                label_list.append(ev.UNKNOWN_LABEL)
        labels = np.array(label_list, dtype=np.int64)

        if np.all(labels == ev.UNKNOWN_LABEL):
            true_labels = self._load_true_labels(subject, session)
            if true_labels is not None:
                if len(true_labels) < n_total_cues:
                    raise ValueError(
                        f"true-label file for S{subject} {session} has "
                        f"{len(true_labels)} entries but session has "
                        f"{n_total_cues} cues"
                    )
                labels = true_labels[trial_ids]
        return labels

    def _load_true_labels(self, subject: int, session: str) -> np.ndarray | None:
        path = self._true_labels_path(subject, session)
        if path is None or not Path(path).exists():
            return None
        from scipy.io import loadmat

        mat = loadmat(str(path))
        if "classlabel" not in mat:
            raise KeyError(f"'classlabel' missing from {path}; keys: {list(mat)}")
        raw_labels = np.asarray(mat["classlabel"]).ravel().astype(int)
        return np.array(
            [ev.classlabel_to_label(x) for x in raw_labels], dtype=np.int64
        )

    def _rejected_keep_mask(
        self, events, mne_id_to_code, epochs, sfreq: float
    ) -> np.ndarray:
        """Mask of trials to keep after dropping any that overlap a reject mark."""
        reject_samples = []
        for e in events:
            if mne_id_to_code.get((int(e[2]))) == ev.EVENT_REJECTED_TRIAL:
                reject_samples.append(int(e[0]))
                
        pre = int(round(self.reject_pre_s * sfreq))
        post = int(round(self.window.tmax * sfreq))
        keep = np.ones(len(epochs.events), dtype=bool)
        for i, cue_sample in enumerate(epochs.events[:, 0]):
            for rs in reject_samples:
                if cue_sample - pre <= rs <= cue_sample + post:
                    keep[i] = False
                    break
        return keep
