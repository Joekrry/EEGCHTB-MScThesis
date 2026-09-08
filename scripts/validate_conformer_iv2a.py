"""Reproduce the EEG-Conformer BCI IV-2a subject-specific baseline.

Same data/split/metric/manifest contract as the EEGNet baseline, trained with
EEG-Conformer's published recipe (Adam lr 2e-4, betas 0.5/0.999, 2000 epochs,
batch 72, segmentation-and-recombination augmentation), the reference defaults
(emb 40, depth 6, 10 heads) and "standard" preprocessing (4-40 Hz + z-score).

Reference: Y. Song et al., "EEG Conformer", IEEE TNSRE 31, 2022.
  https://github.com/eeyhsong/EEG-Conformer

Deviations (leakage-safety and determinism):
  - Checkpoint chosen by lowest loss on a held-out slice of T, not best accuracy
    on test session E (--val-fraction 0 reverts to final-epoch). Pass
    --report-best-test-epoch to reproduce the reference's own selection
    (best test epoch, all of T) as an optimistic implementation-validation figure.
  - Per-channel z-score, not the reference's single global one.
  - Augmentation draws are seeded.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eegchtb.data.channels import GALEA_EEG_CHANNELS
from eegchtb.experiment import ExperimentConfig, channels_tag, run_experiment
from eegchtb.training import predict, resolve_device

SUBJECTS = list(range(1, 10))
METRICS = ("accuracy", "macro_f1", "cohen_kappa", "mcc")


def segment_recombine(
    x: torch.Tensor,
    y: torch.Tensor,
    n_out: int,
    n_classes: int,
    rng: np.random.Generator,
    n_segments: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """S&R augmentation (Song et al.): each synthetic trial splices n_segments
    equal-width time segments, each taken from a random same-class trial."""
    n_times = x.shape[2]
    bounds = [round(s * n_times / n_segments) for s in range(n_segments + 1)]
    per_class = n_out // n_classes
    aug_x, aug_y = [], []
    for cls in range(n_classes):
        pool = torch.nonzero(y == cls, as_tuple=True)[0]
        if per_class == 0 or len(pool) == 0:
            continue
        # (per_class, n_segments) random source-trial indices within this class
        picks = torch.from_numpy(
            rng.integers(0, len(pool), size=(per_class, n_segments))
        ).to(x.device)
        src = pool[picks]  # indices into x, per output trial and segment
        trial = torch.empty((per_class, x.shape[1], n_times), device=x.device)
        for s in range(n_segments):
            a, b = bounds[s], bounds[s + 1]
            trial[:, :, a:b] = x[src[:, s]][:, :, a:b]
        aug_x.append(trial)
        aug_y.append(torch.full((per_class,), cls, dtype=torch.long, device=x.device))
    return torch.cat(aug_x), torch.cat(aug_y)


def _val_loss(model, loader, criterion, device) -> float:
    """Mean cross-entropy over a held-out loader"""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


def make_conformer_trainer(
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    n_segments: int = 8,
    report_best_test_epoch: bool = False,
):
    """Build a run_experiment trainer that reproduces the EEG-Conformer recipe.

    report_best_test_epoch reproduces the reference's own selection: the test
    session is scored every epoch and the best test-accuracy epoch is reported.
    """

    def train(model, train_ds, val_ds, test_ds, config):
        device = torch.device(resolve_device(config.device))
        model.to(device)

        # the training split is both the training set and the S&R pool (any
        # validation trials are already carved out of it by the split)
        x_all, y_all = train_ds.tensors()
        x_all, y_all = x_all.to(device), y_all.to(device)
        n_classes = int(y_all.max().item()) + 1

        # deterministic batch order and augmentation draws for the fixed seed
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, generator=generator
        )
        # best-test-epoch selection reports on the test set itself, so it needs
        # no validation carve
        val_loader = (
            DataLoader(val_ds, batch_size=config.batch_size)
            if not report_best_test_epoch and val_ds is not None and len(val_ds) > 0
            else None
        )
        y_test = test_ds.tensors()[1].numpy() if report_best_test_epoch else None
        rng = np.random.default_rng(config.seed)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        criterion = nn.CrossEntropyLoss()

        history: dict = {"train_loss": [], "val_loss": [], "test_acc": []}
        best_state = copy.deepcopy(model.state_dict())
        best_val, best_epoch = float("inf"), 0
        best_test_acc, best_test_preds = -1.0, None
        start = time.perf_counter()
        desc = f"S{config.subject:02d}" if config.subject is not None else None

        print(
            f"[{desc or 'run'}] dataset={config.dataset} preprocessing={config.preprocessing} "
            f"model={config.model} protocol={config.protocol} channels={channels_tag(config.channels)} "
            f"seed={config.seed} epochs={config.epochs} batch_size={config.batch_size} "
            f"val_fraction={config.val_fraction} lr={lr} betas={betas} n_segments={n_segments} "
            f"selection={'best_test_epoch' if report_best_test_epoch else 'best_val' if val_loader is not None else 'final'} "
            f"device={device}"
        )

        bar = tqdm(range(config.epochs), desc=desc, unit="epoch")
        for epoch in bar:
            model.train()
            running, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                # concatenate one synthetic S&R batch onto each real batch
                aug_x, aug_y = segment_recombine(
                    x_all, y_all, config.batch_size, n_classes, rng, n_segments
                )
                xb = torch.cat([xb, aug_x])
                yb = torch.cat([yb, aug_y])
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(yb)
                seen += len(yb)
            history["train_loss"].append(running / max(seen, 1))
            if report_best_test_epoch:
                # the reference's protocol: score the test session every epoch and
                # keep the best test accuracy.
                preds = predict(model, test_ds, device=config.device)
                acc = float((preds == y_test).mean())
                history["test_acc"].append(acc)
                if acc > best_test_acc:
                    best_test_acc, best_epoch, best_test_preds = acc, epoch, preds
                bar.set_postfix(
                    train_loss=history["train_loss"][-1], test_acc=acc, best=best_epoch
                )
            elif val_loader is not None:
                # leakage-safe model selection: keep the lowest val-loss checkpoint
                # over the fixed schedule. Validation is a held-out slice of session
                # T, so this is the honest analogue of the reference's best-epoch pick.
                v = _val_loss(model, val_loader, criterion, device)
                history["val_loss"].append(v)
                if v < best_val - 1e-6:
                    best_val, best_epoch = v, epoch
                    best_state = copy.deepcopy(model.state_dict())
                bar.set_postfix(
                    train_loss=history["train_loss"][-1], val_loss=v, best=best_epoch
                )
            else:
                bar.set_postfix(train_loss=history["train_loss"][-1])
        timings = {"train_seconds": time.perf_counter() - start}

        if report_best_test_epoch:
            history["best_epoch"] = best_epoch
            history["best_test_acc"] = best_test_acc
            return best_test_preds, timings, history

        if val_loader is not None:
            model.load_state_dict(best_state)  # restore the best-val checkpoint
            history["best_epoch"] = best_epoch
            history["best_val_loss"] = best_val
        else:
            history["best_epoch"] = config.epochs - 1  # no val: final-epoch model
            history["best_val_loss"] = None

        preds = predict(model, test_ds, device=config.device)
        return preds, timings, history

    return train


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default="runs/conformer-iv2a")
    parser.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)  # EEG-Conformer's schedule
    parser.add_argument("--batch-size", type=int, default=72)  # EEG-Conformer's batch
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help=(
            "fraction of session T held out to pick the best checkpoint "
            "(leakage-safe). 0 disables selection and reports the final epoch."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "EEG channels to keep (default: the 10 Galea channels, the project's "
            "target montage). Pass 'all' for the full 22-channel published baseline."
        ),
    )
    parser.add_argument(
        "--report-best-test-epoch",
        action="store_true",
        help=(
            "reproduce the reference's protocol: train on all of session T and "
            "report the best test-epoch accuracy. This selects on the test set "
            "(optimistic, not leakage-safe); use only as an implementation check."
        ),
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="summary JSON path (default: <output_root>/seed<seed>/summary_<montage>.json)",
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save each run's trained model as model.pt in its run directory (default: on)",
    )
    args = parser.parse_args(argv)

    if args.channels is None:
        channels: list[str] | None = list(GALEA_EEG_CHANNELS)
    elif len(args.channels) == 1 and args.channels[0].lower() == "all":
        channels = None  # keep every channel (the 22-channel baseline)
    else:
        channels = args.channels

    val_fraction = args.val_fraction
    if args.report_best_test_epoch:
        # the reference trains on all of session T; no validation carve is used
        val_fraction = 0.0
        print("reference protocol: best test-epoch selection on all of session T "
              "(optimistic, selects on the test set - implementation check only)")

    seed_root = Path(args.output_root) / f"seed{args.seed}"
    trainer = make_conformer_trainer(report_best_test_epoch=args.report_best_test_epoch)
    rows = []
    for subject in SUBJECTS:
        config = ExperimentConfig(
            dataset="bci_iv_2a",
            preprocessing="standard",
            protocol="subject_specific",
            model="eeg_conformer",
            seed=args.seed,
            subject=subject,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=val_fraction,  # held out to select the best checkpoint
            device=args.device,
            channels=channels,
        )
        run_id = (
            f"{config.dataset}_{config.preprocessing}_{config.model}"
            f"_{config.protocol}_{channels_tag(config.channels)}_S{subject:02d}"
        )
        path = run_experiment(config, seed_root, run_id=run_id, trainer=trainer, save_snapshot=args.save_snapshots)
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        history = json.loads((path / "history.json").read_text(encoding="utf-8"))
        rows.append({"subject": subject, **{m: result["metrics"][m] for m in METRICS}})
        best_epoch = history.get("best_epoch")
        print(f"S{subject:02d}  acc={rows[-1]['accuracy']:.4f}  "
              f"kappa={rows[-1]['cohen_kappa']:.4f}  best_epoch={best_epoch}")

    summary = {
        "protocol": {
            "dataset": "bci_iv_2a",
            "preprocessing": "standard",
            "evaluation": "subject_specific (train session T, test session E)",
            "model": "eeg_conformer",
            "training": "EEG-Conformer recipe: Adam(lr=2e-4, betas=(0.5, 0.999)), S&R augmentation",
            "channels": channels_tag(channels),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_fraction": val_fraction,
            "model_selection": (
                "best test-epoch accuracy, all of session T (reference protocol; "
                "selects on the test set - optimistic, not leakage-safe)"
                if args.report_best_test_epoch
                else "best validation-loss checkpoint over the full schedule (leakage-safe)"
                if val_fraction > 0
                else "final epoch (no selection)"
            ),
        },
        "per_subject": rows,
        "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        "std": {m: float(np.std([r[m] for r in rows])) for m in METRICS},
    }
    summary_path = (
        Path(args.summary)
        if args.summary
        else seed_root / f"summary_{channels_tag(channels)}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nmean accuracy {summary['mean']['accuracy']:.4f} "
          f"(sd {summary['std']['accuracy']:.4f})  kappa {summary['mean']['cohen_kappa']:.4f}")
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
