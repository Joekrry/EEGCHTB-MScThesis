"""Continuous-signal preprocessing steps; each transforms an mne.Raw in place and returns it."""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

if TYPE_CHECKING:
    from mne.io import BaseRaw


@dataclass
class PreprocessingStep(abc.ABC):
    """One named, parameterised transformation of a continuous mne.Raw."""

    # registry key; set on each concrete step, not a dataclass field
    name: ClassVar[str]

    @abc.abstractmethod
    def apply(self, raw: BaseRaw) -> BaseRaw:
        """Transform raw in place and return it."""

    def params(self) -> dict:
        """The step's parameters as a plain dict, for provenance and config."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def resolved_provenance(self, raw: BaseRaw) -> dict | None:
        """Runtime-resolved provenance beyond params() (e.g. the Hjorth Laplacian's neighbour graph); None when there is nothing extra to record."""
        return None


STEP_REGISTRY: dict[str, type[PreprocessingStep]] = {}


def register_step(cls: type[PreprocessingStep]) -> type[PreprocessingStep]:
    """Class decorator that registers a step under its .name for config lookup."""
    if cls.name in STEP_REGISTRY:
        raise ValueError(f"preprocessing step {cls.name!r} is already registered")
    STEP_REGISTRY[cls.name] = cls
    return cls


def build_step(name: str, params: dict) -> PreprocessingStep:
    """Instantiate a registered step by name with the given parameters."""
    try:
        cls = STEP_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown preprocessing step {name!r}; "
            f"available: {sorted(STEP_REGISTRY)}"
        ) from None
    return cls(**params)


@register_step
@dataclass
class BandpassFilter(PreprocessingStep):
    """Band-pass filter via mne.Raw.filter (l_freq only = high-pass, h_freq only = low-pass); the zero-phase Hamming firwin FIR design is pinned explicitly and captured in params()."""

    name: ClassVar[str] = "bandpass"
    l_freq: float | None = None
    h_freq: float | None = None
    method: str = "fir"
    fir_design: str = "firwin"
    fir_window: str = "hamming"
    phase: str = "zero"
    filter_length: str = "auto"
    l_trans_bandwidth: str | float = "auto"
    h_trans_bandwidth: str | float = "auto"
    pad: str = "reflect_limited"

    def apply(self, raw: BaseRaw) -> BaseRaw:
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            fir_design=self.fir_design,
            fir_window=self.fir_window,
            phase=self.phase,
            filter_length=self.filter_length,
            l_trans_bandwidth=self.l_trans_bandwidth,  # type: ignore[arg-type]
            h_trans_bandwidth=self.h_trans_bandwidth,  # type: ignore[arg-type]
            pad=self.pad,
            verbose="ERROR",
        )
        return raw


@register_step
@dataclass
class Resample(PreprocessingStep):
    """Resample the continuous signal via mne.Raw.resample before epoching (so cue annotations move with it); the FFT design is pinned explicitly and captured in params()."""

    name: ClassVar[str] = "resample"
    sfreq: float = 250.0
    method: str = "fft"
    window: str = "auto"
    npad: str | int = "auto"
    pad: str = "auto"

    def apply(self, raw: BaseRaw) -> BaseRaw:
        raw.resample(
            sfreq=self.sfreq,
            method=self.method,
            window=self.window,
            npad=self.npad,  # type: ignore[arg-type]  # mne accepts int or "auto"; stub narrows to str
            pad=self.pad,
            verbose="ERROR",
        )
        return raw


@register_step
@dataclass
class Rereference(PreprocessingStep):
    """Set an EEG reference via mne.Raw.set_eeg_reference (e.g. "average", "REST", or a channel name)."""

    name: ClassVar[str] = "rereference"
    ref_channels: str = "average"

    def apply(self, raw: BaseRaw) -> BaseRaw:
        raw.set_eeg_reference(ref_channels=self.ref_channels, verbose="ERROR")
        return raw


@register_step
@dataclass
class Standardize(PreprocessingStep):
    """Per-channel z-score over the whole recording: (x - mean) / (std + eps)."""

    name: ClassVar[str] = "standardize"
    eps: float = 1e-7

    def apply(self, raw: BaseRaw) -> BaseRaw:
        def _zscore(channel):
            return (channel - channel.mean()) / (channel.std() + self.eps)

        raw.apply_function(_zscore, channel_wise=True, verbose="ERROR")
        return raw


@register_step
@dataclass
class ClipOutliers(PreprocessingStep):
    """Per-channel amplitude clipping: values beyond threshold robust-sigma from the channel median are clipped in place.
    Suppresses large-amplitude artifacts before they disrupt per-channel standardize."""

    name: ClassVar[str] = "clip_outliers"
    threshold: float = 8.0  # robust (MAD-derived) standard deviations
    eps: float = 1e-7

    def apply(self, raw: BaseRaw) -> BaseRaw:
        def _clip(channel):
            median = np.median(channel)
            robust_std = np.median(np.abs(channel - median)) * 1.4826 + self.eps
            limit = self.threshold * robust_std
            return np.clip(channel, median - limit, median + limit)

        raw.apply_function(_clip, channel_wise=True, verbose="ERROR")
        return raw


@register_step
@dataclass
class CurrentSourceDensity(PreprocessingStep):
    """Spherical-spline current-source density (surface Laplacian) via compute_current_source_density; kept only as an isolated comparison baseline (underdetermined on sparse Galea montages, so pipelines use HjorthLaplacian)."""

    # Reference:
    #   Perrin, F., Pernier, J., Bertrand, O. and Echallier, J.F., 1989.
    #   Spherical splines for scalp potential and current density mapping.
    #   Electroencephalography and Clinical Neurophysiology, 72(2), pp.184-187.
    #   (Used for: the spherical-spline surface Laplacian MNE implements. Rejected
    #   as the pipeline spatial step because it is underdetermined on a sparse
    #   montage; the pipeline uses HjorthLaplacian instead.)

    name: ClassVar[str] = "csd"
    montage: str = "standard_1020"

    def apply(self, raw: BaseRaw) -> BaseRaw:
        import mne

        if raw.get_montage() is None:
            raw.set_montage(
                self.montage, match_case=False, on_missing="ignore", verbose="ERROR"
            )
        csd = mne.preprocessing.compute_current_source_density(
            raw, copy=True, verbose="ERROR"
        )
        # mne types the result as Raw-or-Epochs; we pass a Raw, so it is a Raw
        return cast("BaseRaw", csd)


def hjorth_laplacian_operator(
    coords: np.ndarray, k: int = 4, weighting: str = "inverse_distance"
) -> tuple[np.ndarray, list[dict]]:
    """Build the (n x n) montage-adaptive Hjorth surface-Laplacian operator (L[i,i]=1, L[i,j]=-w_ij over channel i's k_eff=min(k, n-1) nearest neighbours, rows sum to zero) from 3-D coordinates; returns L and the resolved neighbour graph."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3); got {coords.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if weighting not in ("inverse_distance", "uniform"):
        raise ValueError(
            f"weighting must be 'inverse_distance' or 'uniform'; got {weighting!r}"
        )

    n = coords.shape[0]
    k_eff = min(k, n - 1)
    operator = np.eye(n, dtype=float)
    graph: list[dict] = []
    for i in range(n):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        distances[i] = np.inf  # never a neighbour of itself
        neighbours = np.argsort(distances, kind="stable")[:k_eff]
        if k_eff == 0:
            weights = np.zeros(0, dtype=float)
        elif weighting == "uniform":
            weights = np.full(k_eff, 1.0 / k_eff, dtype=float)
        else:
            inverse = 1.0 / distances[neighbours]
            weights = inverse / inverse.sum()
        operator[i, neighbours] = -weights
        graph.append(
            {
                "channel": int(i),
                "neighbours": [int(j) for j in neighbours],
                "weights": [float(w) for w in weights],
            }
        )
    return operator, graph


@register_step
@dataclass
class HjorthLaplacian(PreprocessingStep):
    """Montage-adaptive Hjorth surface Laplacian: re-reference each channel against a weighted average of its k_eff = min(k, n_channels - 1) nearest electrodes (a single spatial high-pass that stays well conditioned on sparse montages, unlike CSD)."""

    # Reference:
    #   Hjorth, B., 1975. An on-line transformation of EEG scalp potentials into
    #   orthogonal source derivations. Electroencephalography and Clinical
    #   Neurophysiology, 39(5), pp.526-530.
    #   (Used for: the nearest-neighbour surface-Laplacian source derivation this
    #   step implements. The rejected dense-montage alternative is the spherical
    #   spline CSD of Perrin et al. (1989); see CurrentSourceDensity.)

    name: ClassVar[str] = "hjorth_laplacian"
    k: int = 4
    weighting: str = "inverse_distance"
    montage: str = "standard_1020"

    def _coordinates(self, raw: BaseRaw) -> tuple[list[str], np.ndarray]:
        """Channel names and their 3-D positions in channel order; sets the default montage if none is present."""
        if raw.get_montage() is None:
            raw.set_montage(
                self.montage, match_case=False, on_missing="ignore", verbose="ERROR"
            )
        montage = raw.get_montage()
        if montage is None:
            raise ValueError(
                "hjorth_laplacian: raw has no montage and none could be set"
            )
        positions = cast(dict, montage.get_positions()["ch_pos"])
        names = list(raw.ch_names)
        coords = np.array(
            [positions.get(name, np.full(3, np.nan)) for name in names], dtype=float
        )
        missing = [name for name, xyz in zip(names, coords) if not np.all(np.isfinite(xyz))]
        if missing:
            raise ValueError(
                f"hjorth_laplacian: no montage position for channels {missing}"
            )
        return names, coords

    def _operator(self, raw: BaseRaw) -> tuple[list[str], np.ndarray, list[dict]]:
        names, coords = self._coordinates(raw)
        operator, graph = hjorth_laplacian_operator(
            coords, k=self.k, weighting=self.weighting
        )
        return names, operator, graph

    def apply(self, raw: BaseRaw) -> BaseRaw:
        _, operator, _ = self._operator(raw)
        # channel_wise=False passes the whole (n_channels, n_times) array, so the
        # linear operator can mix channels; row sums of zero make it a spatial
        # high-pass that leaves a signal common to all channels near zero.
        raw.apply_function(
            lambda data: operator @ data, channel_wise=False, verbose="ERROR"
        )
        return raw

    def resolved_provenance(self, raw: BaseRaw) -> dict:
        names, _, graph = self._operator(raw)
        neighbours = [
            {
                "channel": names[entry["channel"]],
                "neighbours": [names[j] for j in entry["neighbours"]],
                "weights": entry["weights"],
            }
            for entry in graph
        ]
        return {
            "k": self.k,
            "k_effective": min(self.k, len(names) - 1),
            "weighting": self.weighting,
            "montage": self.montage,
            "channels": names,
            "neighbours": neighbours,
        }
