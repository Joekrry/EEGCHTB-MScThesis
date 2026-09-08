"""Reusable deterministic supervised training loop for the benchmark models."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .seed import set_seed


def resolve_device(device: str = "auto") -> str:
    """Resolve a device spec; "auto" uses CUDA when available and falls back to CPU."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _warmup_cosine_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """LR multiplier: linear ramp 0->1 over warmup_epochs, then cosine decay 1->0 to total_epochs."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    remaining = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / remaining
    return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))


@dataclass
class TrainConfig:
    """Optimiser, schedule, early-stopping and reproducibility settings for one run."""

    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    patience: int = 20  # epochs without val improvement before stopping
    scheduler: str | None = None  # None or "cosine"
    device: str = "auto"  # CUDA when available, else CPU
    seed: int = 0
    max_norm: bool = True  # call model.apply_max_norm_constraints() after each step
    grad_clip: float | None = 1.0  # clip gradient L2 norm before each optimiser step; None disables
    warmup_epochs: int = 0  # linear LR ramp from 0 to lr over this many epochs before the schedule proper
    augment: Callable | None = None
    desc: str | None = None  # progress bar label
    leave_progress: bool = True  # keep the finished bar on screen


@dataclass
class TrainResult:
    """History, best-epoch checkpoint metadata and timings from a training run."""

    history: dict = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_run: int = 0
    timings: dict = field(default_factory=dict)


def _evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    config: TrainConfig,
) -> TrainResult:
    """Train model with Adam and cross-entropy; early-stop on val loss, restore the best weights."""
    set_seed(config.seed)
    device = torch.device(resolve_device(config.device))
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size)
        if val_dataset is not None and len(cast(Sized, val_dataset)) > 0
        else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: _warmup_cosine_factor(e, config.warmup_epochs, config.epochs),
        )
        if config.scheduler == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss()

    history: dict = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    since_improved = 0
    start = time.perf_counter()

    epoch = 0
    bar = tqdm(
        range(config.epochs),
        desc=config.desc,
        disable=config.desc is None,
        leave=config.leave_progress,
        unit="epoch",
    )
    for epoch in bar:
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if config.augment is not None:
                xb, yb = config.augment(xb, yb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            constrain = getattr(model, "apply_max_norm_constraints", None)
            if config.max_norm and callable(constrain):
                constrain()
            running += loss.item() * len(yb)
            n += len(yb)
        history["train_loss"].append(running / max(n, 1))
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss = _evaluate_loss(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
                since_improved = 0
            else:
                since_improved += 1
            bar.set_postfix(
                train_loss=history["train_loss"][-1], val_loss=val_loss, best=best_epoch
            )
            if since_improved >= config.patience:
                break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bar.set_postfix(train_loss=history["train_loss"][-1])

    model.load_state_dict(best_state)
    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_run=epoch + 1,
        timings={"train_seconds": time.perf_counter() - start},
    )


def predict(model: nn.Module, dataset: Dataset, *, device: str = "auto", batch_size: int = 64) -> np.ndarray:
    """Predicted integer labels for every trial in dataset (argmax over logits)."""
    dev = torch.device(resolve_device(device))
    model.to(dev).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(dev)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)
