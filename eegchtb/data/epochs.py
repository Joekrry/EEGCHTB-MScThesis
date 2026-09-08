"""The EpochBundle, the common return type for every loader."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .events import UNKNOWN_LABEL
from ..run_output import atomic_write

# Units a bundle's data may carry: microvolts, volts, or per-channel z-scored by
# a standardize preprocessing step (in which case the scale is dimensionless).
STANDARDIZED_UNITS = "standardized"
VALID_UNITS = ("uV", "V", STANDARDIZED_UNITS)


@dataclass
class EpochBundle:
    """Trial-level EEG epochs with labels and metadata: data (n_trials, n_channels, n_times) float32, labels (n_trials,) int (-1 = hidden-eval cue), units "uV"/"V"/"standardized"."""

    data: np.ndarray
    labels: np.ndarray
    dataset: str
    subject: int
    session: str
    sfreq: float
    channel_names: list[str]
    class_names: dict[int, str]
    times: np.ndarray | None = None
    trial_ids: np.ndarray | None = None
    units: str = "uV"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels).astype(np.int64, copy=False)

        if self.units not in VALID_UNITS:
            raise ValueError(f"units must be one of {VALID_UNITS}; got {self.units!r}")
        if self.data.ndim != 3:
            raise ValueError(
                f"data must be 3D (n_trials, n_channels, n_times); "
                f"got shape {self.data.shape}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be 1D; got shape {self.labels.shape}")
        if len(self.labels) != self.data.shape[0]:
            raise ValueError(
                f"labels length ({len(self.labels)}) != n_trials "
                f"({self.data.shape[0]})"
            )
        if len(self.channel_names) != self.data.shape[1]:
            raise ValueError(
                f"channel_names length ({len(self.channel_names)}) != n_channels "
                f"({self.data.shape[1]})"
            )
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=np.float64)
            if len(self.times) != self.data.shape[2]:
                raise ValueError(
                    f"times length ({len(self.times)}) != n_times "
                    f"({self.data.shape[2]})"
                )
        if self.trial_ids is not None:
            self.trial_ids = np.asarray(self.trial_ids).astype(np.int64, copy=False)
            if len(self.trial_ids) != self.data.shape[0]:
                raise ValueError("trial_ids length != n_trials")

    # shape helpers
    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_times(self) -> int:
        return self.data.shape[2]

    # label helpers
    def class_distribution(self) -> dict[int, int]:
        """Return {label: count} over all trials (including -1)."""
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    @property
    def has_unknown_labels(self) -> bool:
        return bool(np.any(self.labels == UNKNOWN_LABEL))

    def filter_known(self) -> "EpochBundle":
        """Return a copy containing only trials with a known (>=0) label."""
        mask = self.labels >= 0
        return self.select_trials(mask)

    def select_classes(self, labels: Sequence[int]) -> "EpochBundle":
        """Return a copy containing only trials whose label is in labels."""
        wanted = [int(x) for x in labels]
        return self.select_trials(np.isin(self.labels, wanted))

    def select_trials(self, mask: np.ndarray) -> "EpochBundle":
        """Return a copy holding only the trials selected by a boolean mask."""
        return EpochBundle(
            data=self.data[mask],
            labels=self.labels[mask],
            dataset=self.dataset,
            subject=self.subject,
            session=self.session,
            sfreq=self.sfreq,
            channel_names=list(self.channel_names),
            class_names=dict(self.class_names),
            times=None if self.times is None else self.times.copy(),
            trial_ids=None if self.trial_ids is None else self.trial_ids[mask],
            units=self.units,
            metadata=dict(self.metadata),
        )

    # persistence
    def save(self, path) -> None:
        """Save to a compressed .npz file atomically via atomic_write (round-trips via load)."""
        with atomic_write(path) as handle:
            np.savez_compressed(
                handle,
                data=self.data,
                labels=self.labels,
                dataset=self.dataset,
                subject=self.subject,
                session=self.session,
                sfreq=self.sfreq,
                channel_names=np.array(self.channel_names, dtype=object),
                class_label_keys=np.array(list(self.class_names.keys()), dtype=np.int64),
                class_label_vals=np.array(list(self.class_names.values()), dtype=object),
                times=np.array([]) if self.times is None else self.times,
                trial_ids=np.array([]) if self.trial_ids is None else self.trial_ids,
                units=self.units,
                metadata=np.array(json.dumps(self.metadata), dtype=object),
            )

    @classmethod
    def load(cls, path) -> "EpochBundle":
        with np.load(path, allow_pickle=True) as f:
            times = f["times"]
            trial_ids = f["trial_ids"]
            class_names: dict[int, str] = {}
            for key, value in zip(f["class_label_keys"], f["class_label_vals"]):
                class_names[int(key)] = str(value)
            metadata = {}
            if "metadata" in f:
                metadata = json.loads(str(f["metadata"]))
            return cls(
                data=f["data"],
                labels=f["labels"],
                dataset=str(f["dataset"]),
                subject=int(f["subject"]),
                session=str(f["session"]),
                sfreq=float(f["sfreq"]),
                channel_names=[str(c) for c in f["channel_names"]],
                class_names=class_names,
                times=None if times.size == 0 else times,
                trial_ids=None if trial_ids.size == 0 else trial_ids,
                units=str(f["units"]),
                metadata=metadata,
            )

    def __repr__(self) -> str:
        return (
            f"EpochBundle(dataset={self.dataset!r}, subject={self.subject}, "
            f"session={self.session!r}, n_trials={self.n_trials}, "
            f"n_channels={self.n_channels}, n_times={self.n_times}, "
            f"sfreq={self.sfreq}, units={self.units!r}, "
            f"dist={self.class_distribution()})"
        )


def concatenate(bundles: Sequence[EpochBundle]) -> EpochBundle:
    """Concatenate bundles along the trial axis (must share channels, n_times, sfreq, units)."""
    if not bundles:
        raise ValueError("cannot concatenate an empty sequence of bundles")

    first = bundles[0]
    for b in bundles[1:]:
        if b.n_channels != first.n_channels:
            raise ValueError("channel count mismatch when concatenating bundles")
        if b.n_times != first.n_times:
            raise ValueError("n_times mismatch when concatenating bundles")
        if b.sfreq != first.sfreq:
            raise ValueError("sfreq mismatch when concatenating bundles")
        if b.units != first.units:
            raise ValueError("units mismatch when concatenating bundles")

    subjects = {b.subject for b in bundles}
    sessions = {b.session for b in bundles}
    datasets = {b.dataset for b in bundles}

    ids = []
    for b in bundles:
        if b.trial_ids is not None:
            ids.append(b.trial_ids)
    trial_ids = np.concatenate(ids) if len(ids) == len(bundles) else None

    merged_dataset = first.dataset if len(datasets) == 1 else "+".join(sorted(datasets))
    merged_subject = first.subject if len(subjects) == 1 else -1
    merged_session = first.session if len(sessions) == 1 else "+".join(sorted(sessions))

    return EpochBundle(
        data=np.concatenate([b.data for b in bundles], axis=0),
        labels=np.concatenate([b.labels for b in bundles], axis=0),
        dataset=merged_dataset,
        subject=merged_subject,
        session=merged_session,
        sfreq=first.sfreq,
        channel_names=list(first.channel_names),
        class_names=dict(first.class_names),
        times=None if first.times is None else first.times.copy(),
        trial_ids=trial_ids,
        units=first.units,
        metadata=_merge_metadata(bundles),
    )


def _merge_metadata(bundles: Sequence[EpochBundle]) -> dict:
    """Record which sessions were merged, keeping any metadata all of them agree on.

    Provenance the loader attached (notably the preprocessing block) is identical
    across a subject's sessions, so dropping it would leave a concatenated bundle
    with no record of how it was produced. Keys the bundles disagree on are
    ambiguous once merged, so they are dropped rather than resolved arbitrarily.
    """
    merged = {"concatenated_from": [b.session for b in bundles]}
    first = bundles[0]
    for key, value in first.metadata.items():
        if key == "concatenated_from":
            continue
        if all(key in b.metadata and b.metadata[key] == value for b in bundles[1:]):
            merged[key] = value
    return merged
