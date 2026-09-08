"""Cross-subject control: score every 2a snapshot against every 2a subject.

The Galea transfer results sit at chance, could mean the headset is the obstacle, 
or that a subject-specific model never transfers to a new person

python scripts/eval_cross_subject.py --grid-root runs/grid-iv2a --out runs/cross_subject_2a
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from eegchtb.data.base import EpochWindow
from eegchtb.data.bci_iv_2a import BCICompetitionIV2aLoader
from eegchtb.data.bci_iv_2b import BCICompetitionIV2bLoader
from eegchtb.data.epochs import concatenate
from eegchtb.evaluation import EpochBundleDataset, compute_metrics
from eegchtb.experiment import load_snapshot
from eegchtb.run_output import atomic_write
from eegchtb.training import predict

SUBJECTS = list(range(1, 10))
METRICS = ("accuracy", "macro_f1", "cohen_kappa", "mcc")
# loader and the released evaluation sessions, which training never sees; 2b
# splits its evaluation across two sessions where 2a has one
DATASETS = {
    "bci_iv_2a": (BCICompetitionIV2aLoader, ("E",)),
    "bci_iv_2b": (BCICompetitionIV2bLoader, ("04E", "05E")),
}


def _load_eval_bundles(dataset, data_root, preprocessing, channels, window):
    """Every subject's evaluation trials under one preprocessing recipe. 
    each (preprocessing, subject) pair is read once and reused across every snapshot scored against it.
    """
    loader_cls, sessions = DATASETS[dataset]
    loader = loader_cls(
        data_root, window=window, preprocessing=preprocessing, channels=channels
    )
    bundles = {}
    for subject in SUBJECTS:
        per_session = [
            loader.load_session(subject, session).filter_known() for session in sessions
        ]
        bundle = per_session[0] if len(per_session) == 1 else concatenate(per_session)
        bundles[subject] = EpochBundleDataset(bundle), bundle.labels
    return bundles


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", default="runs/grid-iv2a")
    parser.add_argument("--out", default="runs/cross_subject_2a")
    parser.add_argument("--dataset", default="bci_iv_2a", choices=sorted(DATASETS))
    parser.add_argument("--montage", default="galea")
    parser.add_argument("--data-root", default="eegchtb/data")
    parser.add_argument("--preprocessing", nargs="+",
                        default=["minimal", "standard", "deep", "maximum"])
    parser.add_argument("--models", nargs="+",
                        default=["eegnet", "eeg_conformer", "grf_net"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    grid_root = Path(args.grid_root)
    out_root = Path(args.out)

    for preprocessing in args.preprocessing:
        bundles = None  # built lazily, from the first manifest of this recipe

        for model_name in args.models:
            cell = (f"{args.dataset}_{preprocessing}_{model_name}"
                    f"_subject_specific_{args.montage}")
            rows = []

            for seed, train_subject in itertools.product(args.seeds, SUBJECTS):
                snapshot_path = (grid_root / f"{cell}_seed{seed}"
                                 / f"S{train_subject:02d}" / "model.pt")
                if not snapshot_path.exists():
                    continue

                manifest = json.loads(
                    (snapshot_path.parent / "manifest.json").read_text(encoding="utf-8")
                )
                if bundles is None:
                    bundles = _load_eval_bundles(
                        args.dataset,
                        args.data_root,
                        preprocessing,
                        manifest["channels_used"],
                        EpochWindow(**manifest["config"]["window"]),
                    )

                model, _ = load_snapshot(snapshot_path)
                for test_subject in SUBJECTS:
                    dataset, labels = bundles[test_subject]
                    metrics = compute_metrics(labels, predict(model, dataset, device=args.device))
                    rows.append({
                        "seed": seed,
                        "train_subject": train_subject,
                        "test_subject": test_subject,
                        "within": train_subject == test_subject,
                        **metrics,
                    })

            if not rows:
                print(f"skip {cell}: no snapshots found")
                continue

            within = [r for r in rows if r["within"]]
            cross = [r for r in rows if not r["within"]]
            summary = {
                "cell": cell,
                "n_pairs": len(rows),
                "within_subject": {
                    "n": len(within),
                    "mean": {m: float(np.mean([r[m] for r in within])) for m in METRICS},
                    "std": {m: float(np.std([r[m] for r in within])) for m in METRICS},
                },
                "cross_subject": {
                    "n": len(cross),
                    "mean": {m: float(np.mean([r[m] for r in cross])) for m in METRICS},
                    "std": {m: float(np.std([r[m] for r in cross])) for m in METRICS},
                },
                "per_pair": rows,
            }
            with atomic_write(out_root / f"{cell}.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"{cell}: within {summary['within_subject']['mean']['accuracy']:.3f}  "
                  f"cross {summary['cross_subject']['mean']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
