"""Evaluate trained grid snapshot(s) against real Galea BDF recordings.

python scripts/eval_galea.py --grid-root runs/grid-iv2a --dataset bci_iv_2a --preprocessing minimal --model eegnet --seeds 0 1 2 3 4

"""
from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from eegchtb.data.base import EpochWindow
from eegchtb.data.bdf import GaleaBdfLoader
from eegchtb.evaluation import EpochBundleDataset, compute_metrics
from eegchtb.experiment import load_snapshot
from eegchtb.training import predict

SUBJECTS = list(range(1, 10))
METRICS = ("accuracy", "macro_f1", "cohen_kappa", "mcc")


def evaluate_snapshot(
    snapshot_path: str | Path,
    *,
    data_root: str | Path = "eegchtb/data",
    session: str = "recording_2026-08-21_15-45-40",
    device: str = "auto",
    preprocessing: str | None = None,
    classes: Sequence[int] | None = None,
) -> dict:
    """Score one trained snapshot's predictions against a Galea recording

    classes restricts the Galea trials to a label subset. A 2b snapshot has only
    two outputs (left=0, right=1)
    """
    snapshot_path = Path(snapshot_path)
    manifest_path = snapshot_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]

    model, snapshot = load_snapshot(snapshot_path)

    loader = GaleaBdfLoader(
        data_root,
        window=EpochWindow(**config["window"]),
        preprocessing=preprocessing or config["preprocessing"],
        channels=manifest["channels_used"],  # exact order the model was trained on
    )
    bundle = loader.load_session(1, session)
    if classes is not None:
        bundle = bundle.select_classes(list(classes))
    if bundle.n_trials == 0:
        raise ValueError(f"no MI trials found in session {session!r}")
    if bundle.channel_names != snapshot["channel_names"]:
        raise ValueError(
            f"channel mismatch: snapshot expects {snapshot['channel_names']}, "
            f"got {bundle.channel_names}"
        )
    if bundle.data.shape[2] != snapshot["n_times"]:
        raise ValueError(
            f"n_times mismatch: snapshot expects {snapshot['n_times']}, "
            f"got {bundle.data.shape[2]}"
        )

    preds = predict(model, EpochBundleDataset(bundle), device=device)
    metrics = compute_metrics(bundle.labels, preds)

    return {
        "snapshot": str(snapshot_path),
        "config": config,
        "preprocessing_used": preprocessing or config["preprocessing"],
        "metrics": metrics,
        "n_trials": bundle.n_trials,
        "classes": sorted(set(bundle.labels.tolist())),
    }


def _resolve_recording(args) -> tuple[str, str]:
    """(data_root, session) for the recording to evaluate against.

    --recording points straight at a .bdf file anywhere on disk; otherwise fall
    back to --session/--data-root (session resolved inside
    <data-root>/galea_recordings_joe, as GaleaBdfLoader expects).
    """
    if args.recording:
        path = Path(args.recording)
        return str(path.parent), path.stem
    return args.data_root, args.session


def _eval_kwargs(args) -> dict:
    data_root, session = _resolve_recording(args)
    return dict(
        data_root=data_root,
        session=session,
        device=args.device,
        preprocessing=args.galea_preprocessing,
        classes=args.classes,
    )


def _run_single(args) -> None:
    _, session = _resolve_recording(args)
    result = evaluate_snapshot(args.snapshot, **_eval_kwargs(args))
    config = result["config"]

    print(f"snapshot:   {result['snapshot']}")
    print(f"trained on: {config['dataset']} / {config['preprocessing']} / {config['model']}"
          f" (subject {config['subject']})")
    print(f"tested on:  galea_bdf / {session}, preprocessed with "
          f"{result['preprocessing_used']!r} ({result['n_trials']} trials, "
          f"classes {result['classes']})")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")


def _run_seeds(args) -> None:
    grid_root = Path(args.grid_root)
    cell_name = f"{args.dataset}_{args.preprocessing}_{args.model}_subject_specific_{args.montage}"

    _, session = _resolve_recording(args)
    eval_kwargs = _eval_kwargs(args)
    rows = []
    for seed, subject in itertools.product(args.seeds, SUBJECTS):
        snapshot_path = grid_root / f"{cell_name}_seed{seed}" / f"S{subject:02d}" / "model.pt"
        if not snapshot_path.exists():
            print(f"skip seed{seed} S{subject:02d}: no snapshot at {snapshot_path}")
            continue
        result = evaluate_snapshot(snapshot_path, **eval_kwargs)
        row = {"seed": seed, "subject": subject, **result["metrics"]}
        rows.append(row)
        print(f"seed{seed} S{subject:02d}  "
              + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS))

    if not rows:
        raise SystemExit(f"no snapshots found under {grid_root} for cell {cell_name!r}")

    summary = {
        "cell": cell_name,
        "session": session,
        "n_snapshots": len(rows),
        "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        "std": {m: float(np.std([r[m] for r in rows])) for m in METRICS},
        "per_run": rows,
    }

    print(f"\n{cell_name}  ({len(rows)} snapshots across {len(args.seeds)} seeds)")
    for m in METRICS:
        print(f"  {m}: {summary['mean'][m]:.4f} +/- {summary['std'][m]:.4f}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {summary_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", help="path to a single run's model.pt")
    parser.add_argument(
        "--grid-root", default=None,
        help="root dir the grid wrote cells into (e.g. runs/grid-iv2a); "
             "batch-evaluates every seed/subject snapshot in the named cell",
    )
    parser.add_argument("--dataset", default="bci_iv_2a")
    parser.add_argument("--preprocessing", default="standard")
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--montage", default="galea")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="seeds to evaluate in --grid-root batch mode",
    )
    parser.add_argument("--summary", default=None, help="batch mode: path to write a JSON summary")
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="restrict Galea trials to these labels (2b snapshots are 2-class, so "
             "transferring to them needs '--classes 0 1' to drop feet and tongue)",
    )
    parser.add_argument(
        "--galea-preprocessing", default=None,
        help="override the Galea-side preprocessing recipe (e.g. galea_clean) instead of "
             "replaying the snapshot's own training recipe",
    )
    parser.add_argument("--data-root", default="eegchtb/data")
    parser.add_argument(
        "--session",
        default="recording_2026-08-21_15-45-40",
        help="Galea recording (bdf filename stem) to evaluate against, resolved under "
             "--data-root/galea_recordings_joe/. Ignored if --recording is given.",
    )
    parser.add_argument(
        "--recording", default=None,
        help="path to a .bdf file to evaluate against, anywhere on disk "
             "(overrides --session/--data-root)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if bool(args.snapshot) == bool(args.grid_root):
        raise SystemExit("pass exactly one of: a snapshot path, or --grid-root")

    if args.snapshot:
        _run_single(args)
    else:
        _run_seeds(args)


if __name__ == "__main__":
    main()
