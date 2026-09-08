"""Training layer: central seed control and the reusable supervised trainer."""

from __future__ import annotations

from .recipes import GRID_VAL_FRACTION, grid_train_config, make_augment, segment_recombine
from .seed import set_seed
from .trainer import TrainConfig, TrainResult, predict, resolve_device, train_model

__all__ = [
    "set_seed",
    "TrainConfig",
    "TrainResult",
    "train_model",
    "predict",
    "resolve_device",
    "grid_train_config",
    "make_augment",
    "segment_recombine",
    "GRID_VAL_FRACTION",
]
