"""EEG data loading; every loader returns a common EpochBundle."""

from __future__ import annotations

from . import events
from .base import BaseEEGLoader, EpochWindow
from .bci_iv_2a import BCICompetitionIV2aLoader
from .bci_iv_2b import BCICompetitionIV2bLoader
from .bdf import GaleaBdfLoader
from .channels import GALEA_BCI_IV_2A_OVERLAP_CHANNELS, GALEA_EEG_CHANNELS, select_channels
from .epochs import EpochBundle, concatenate

# registry of file-based dataset loaders by name
DATASET_LOADERS = {
    BCICompetitionIV2aLoader.name: BCICompetitionIV2aLoader,
    BCICompetitionIV2bLoader.name: BCICompetitionIV2bLoader,
}


def get_loader(name: str, **kwargs):
    """Construct a DATASET_LOADERS loader by name; kwargs go to its constructor."""
    try:
        cls = DATASET_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASET_LOADERS)}."
        ) from None
    return cls(**kwargs)
