"""Loader interface and the shared epoch-window configuration."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass

from .epochs import EpochBundle, concatenate


@dataclass(frozen=True)
class EpochWindow:
    """How a recording is sliced into trials: round((tmax - tmin) * sfreq) samples each."""

    tmin: float = 0.0
    tmax: float = 4.0
    units: str = "uV"
    baseline: tuple[float, float] | None = None
    drop_rejected: bool = False

    def __post_init__(self) -> None:
        if self.tmax <= self.tmin:
            raise ValueError(f"tmax ({self.tmax}) must be > tmin ({self.tmin})")
        if self.units not in ("uV", "V"):
            raise ValueError(f"units must be 'uV' or 'V'; got {self.units!r}")
        if self.baseline is not None and self.baseline[1] <= self.baseline[0]:
            raise ValueError(f"baseline must be (start, stop) with stop > start; got {self.baseline}")

    def n_times(self, sfreq: float) -> int:
        """Number of samples per trial at sfreq Hz."""
        return int(round((self.tmax - self.tmin) * sfreq))


class BaseEEGLoader(abc.ABC):
    """Base class for dataset loaders; every one returns EpochBundles."""

    # dataset identifier, e.g. "bci_iv_2a"; declared by every concrete subclass
    name: str

    def __init_subclass__(cls, abstract: bool = False, **kwargs) -> None:
        """Require a concrete loader to declare `name`; pass abstract=True to exempt a shared base.

        A loader that forgets `name` would silently stamp the wrong dataset on
        every bundle it returns, so the class is refused rather than its bundles.
        ABCMeta has not filled in __abstractmethods__ by this point, so classes
        still carrying an unimplemented abstract method are detected directly.
        """
        super().__init_subclass__(**kwargs)
        has_abstract_methods = any(
            getattr(getattr(cls, attr, None), "__isabstractmethod__", False)
            for attr in dir(cls)
        )
        if abstract or has_abstract_methods:
            return
        if not isinstance(getattr(cls, "name", None), str):
            raise TypeError(f"{cls.__name__} must declare a string `name`")

    @abc.abstractmethod
    def available_subjects(self) -> list[int]:
        """Subject numbers present on disk for this dataset."""

    @abc.abstractmethod
    def available_sessions(self, subject: int) -> list[str]:
        """Session identifiers available for a subject."""

    @abc.abstractmethod
    def load_session(self, subject: int, session: str) -> EpochBundle:
        """Load a single session's trials for a subject."""

    def load_subject(
        self, subject: int, sessions: Iterable[str] | None = None
    ) -> EpochBundle:
        """Load and concatenate sessions for one subject (all by default)."""
        session_ids = (
            list(sessions) if sessions is not None else self.available_sessions(subject)
        )
        bundles = [self.load_session(subject, s) for s in session_ids]
        return concatenate(bundles)

    def load_all(self, subjects: Iterable[int] | None = None) -> list[EpochBundle]:
        """Load every (subject, session) bundle, one entry per session."""
        subject_ids = (
            list(subjects) if subjects is not None else self.available_subjects()
        )
        return [
            self.load_session(subject, session)
            for subject in subject_ids
            for session in self.available_sessions(subject)
        ]
