"""Reproduce the EEGNet BCI IV-2a subject-specific baseline (Gate 3).

Frozen protocol: per subject, train on session T and test on session E (the
competition's own split), "standard" preprocessing (4-40 Hz band-pass +
per-channel z-score), EEGNet with a temporal kernel of half the sampling rate
(the reference rule; 125 at 250 Hz), one seed. Training mirrors the EEG-TCNet
baseline harness that the comparison figure comes from: a fixed 750-epoch
schedule on the whole T session, batch 64, Adam lr 1e-3, no validation and no
early stopping (final-epoch model). Writes one run directory per subject plus a
compact summary committed as the validation evidence.

Usage:
    python scripts/validate_eegnet_iv2a.py [output_root]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eegchtb.experiment import ExperimentConfig, channels_tag, run_experiment

SUBJECTS = list(range(1, 10))
METRICS = ("accuracy", "macro_f1", "cohen_kappa", "mcc")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default="runs/eegnet-iv2a-baseline")
    parser.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=750)  # EEG-TCNet's fixed schedule
    parser.add_argument(
        "--summary",
        default=None,
        help="summary JSON path (default: <output_root>/seed<seed>/summary.json)",
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save each run's trained model as model.pt in its run directory (default: on)",
    )
    args = parser.parse_args(argv)

    # keep each seed's run directories in their own seed{N} subdirectory
    seed_root = Path(args.output_root) / f"seed{args.seed}"
    rows = []
    for subject in SUBJECTS:
        config = ExperimentConfig(
            dataset="bci_iv_2a",
            preprocessing="standard",
            protocol="subject_specific",
            model="eegnet",
            seed=args.seed,
            subject=subject,
            epochs=args.epochs,
            batch_size=64,
            val_fraction=0.0,  # no val carve -> fixed schedule
            device=args.device,
        )
        # self-describing leaf name (seed is already the parent seed{N} directory)
        run_id = (
            f"{config.dataset}_{config.preprocessing}_{config.model}"
            f"_{config.protocol}_{channels_tag(config.channels)}_S{subject:02d}"
        )
        path = run_experiment(config, seed_root, run_id=run_id, save_snapshot=args.save_snapshots)
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        rows.append({"subject": subject, **{m: result["metrics"][m] for m in METRICS}})
        print(f"S{subject:02d}  acc={rows[-1]['accuracy']:.4f}  kappa={rows[-1]['cohen_kappa']:.4f}")

    summary = {
        "protocol": {
            "dataset": "bci_iv_2a",
            "preprocessing": "standard",
            "evaluation": "subject_specific (train session T, test session E)",
            "model": "eegnet",
            "kernel_length": "sfreq/2 (125 at 250 Hz)",
            "seed": args.seed,
            "epochs": args.epochs,
            "early_stopping": False,
            "batch_size": 64,
            "val_fraction": 0.0,
        },
        "per_subject": rows,
        "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        "std": {m: float(np.std([r[m] for r in rows])) for m in METRICS},
    }
    summary_path = Path(args.summary) if args.summary else seed_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nmean accuracy {summary['mean']['accuracy']:.4f} "
          f"(sd {summary['std']['accuracy']:.4f})  kappa {summary['mean']['cohen_kappa']:.4f}")
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
