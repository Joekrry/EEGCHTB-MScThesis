"""The single shared training recipe for the benchmark grid.

The grid compares architectures, so every model must be trained identically, same optimiser, schedule, selection rule, augmentation and seed policy
"""

from __future__ import annotations

import numpy as np
import torch

from .trainer import TrainConfig

# stratified validation carved from the training data for best-val selection
GRID_VAL_FRACTION = 0.15
# uniform optimiser recipe (chosen once on validation, not per model)
GRID_LR = 1e-3
GRID_WEIGHT_DECAY = 1e-4
GRID_SCHEDULER = "cosine"
GRID_EPOCHS = 500
GRID_PATIENCE = 250
GRID_WARMUP_EPOCHS = 20
GRID_BATCH_SIZE = 64
GRID_N_SEGMENTS = 8


def grid_train_config(*, seed: int, device: str = "auto", desc: str | None = None,
                      epochs: int = GRID_EPOCHS, batch_size: int = GRID_BATCH_SIZE,
                      augment=None) -> TrainConfig:
    """The canonical grid TrainConfig: uniform Adam + cosine + best-val early stopping. Identical for every architecture."""
    return TrainConfig(
        epochs=epochs,
        lr=GRID_LR,
        weight_decay=GRID_WEIGHT_DECAY,
        batch_size=batch_size,
        patience=GRID_PATIENCE,
        scheduler=GRID_SCHEDULER,
        warmup_epochs=GRID_WARMUP_EPOCHS,
        device=device,
        seed=seed,
        desc=desc,
        augment=augment,
    )


def segment_recombine(x: torch.Tensor, y: torch.Tensor, n_out: int, n_classes: int,
                      rng: np.random.Generator, n_segments: int = GRID_N_SEGMENTS):
    """Build n_out synthetic trials by stitching same-class time segments drawn from (x, y). Leakage-safe: it only ever mixes training trials."""
    n_times = x.shape[2]
    bounds = [round(s * n_times / n_segments) for s in range(n_segments + 1)]
    per_class = n_out // n_classes
    aug_x, aug_y = [], []
    for cls in range(n_classes):
        pool = torch.nonzero(y == cls, as_tuple=True)[0]
        if per_class == 0 or len(pool) == 0:
            continue
        picks = torch.from_numpy(rng.integers(0, len(pool), size=(per_class, n_segments))).to(x.device)
        src = pool[picks]
        trial = torch.empty((per_class, x.shape[1], n_times), device=x.device)
        for s in range(n_segments):
            a, b = bounds[s], bounds[s + 1]
            trial[:, :, a:b] = x[src[:, s]][:, :, a:b]
        aug_x.append(trial)
        aug_y.append(torch.full((per_class,), cls, dtype=torch.long, device=x.device))
    if not aug_x:
        return x[:0], y[:0]
    return torch.cat(aug_x), torch.cat(aug_y)


def make_augment(x_all: torch.Tensor, y_all: torch.Tensor, *, seed: int, n_classes: int,
                 n_segments: int = GRID_N_SEGMENTS):
    """A per-batch augmentation closure for TrainConfig.augment: appends a batch-sized block of segment-recombined trials sampled from the whole training pool.

    x_all / y_all are the full training tensors; the pool is moved to the batch's
    device lazily so the closure works on CPU or GPU.
    """
    rng = np.random.default_rng(seed)
    pool = {"x": x_all, "y": y_all, "moved": False}

    def augment(xb: torch.Tensor, yb: torch.Tensor):
        if not pool["moved"]:
            pool["x"] = pool["x"].to(xb.device)
            pool["y"] = pool["y"].to(xb.device)
            pool["moved"] = True
        ax, ay = segment_recombine(pool["x"], pool["y"], len(yb), n_classes, rng, n_segments)
        return torch.cat([xb, ax]), torch.cat([yb, ay])

    return augment
