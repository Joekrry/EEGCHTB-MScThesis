"""GRF-Net BCI IV-2a validation under its original methodology (the MiniRocket protocol). """

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

from eegchtb.data.alignment import align_sessions
from eegchtb.data.channels import GALEA_EEG_CHANNELS
from eegchtb.evaluation import RunResult, Split, compute_metrics, config_fingerprint, write_run_manifest
from eegchtb.experiment import ExperimentConfig, _assemble, _build_loader, _subjects_to_load, channels_tag
from eegchtb.models import GRFNet
from eegchtb.run_output import atomic_write, run_directory
from eegchtb.training import resolve_device, segment_recombine, set_seed

SUBJECTS = list(range(1, 10))
METRICS = ("accuracy", "macro_f1", "cohen_kappa", "mcc")


def _train_fold(x_tr, y_tr, x_te, *, sfreq, n_classes, epochs, batch_size, lr, seed, device, n_segments=8, desc=None):
    """Train GRF-Net on one pooled fold and predict the held-out trials with the final-epoch model."""
    set_seed(seed)
    dev = torch.device(resolve_device(device))
    model = GRFNet(x_tr.shape[1], x_tr.shape[2], n_classes, sfreq=sfreq).to(dev)
    x_tr, y_tr, x_te = x_tr.to(dev), y_tr.to(dev), x_te.to(dev)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loss = []
    for _ in tqdm(range(epochs), desc=desc, unit="epoch", leave=False):
        model.train(); running, seen = 0.0, 0
        for xb, yb in loader:
            ax, ay = segment_recombine(x_tr, y_tr, batch_size, n_classes, rng, n_segments)
            xb = torch.cat([xb, ax]); yb = torch.cat([yb, ay])
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            running += loss.item() * len(yb); seen += len(yb)
        train_loss.append(running / max(seen, 1))
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(x_te[i:i + 128]).argmax(1) for i in range(0, x_te.shape[0], 128)])
    return preds.cpu().numpy(), train_loss


def _resolve_channels(names):
    if names is None:
        return None
    if len(names) == 1 and names[0].lower() == "all":
        return None
    if len(names) == 1 and names[0].lower() == "galea":
        return list(GALEA_EEG_CHANNELS)
    return names


def _pooled_subject_data(subject, channels):
    """Load a subject's pooled trials (both sessions) via the core loaders; the pooled split is done in this script, not by a framework protocol. test_mask marks the E session, used as the session id for Euclidean Alignment."""
    cfg = ExperimentConfig(dataset="bci_iv_2a", preprocessing="standard", protocol="subject_specific",
                           model="grf_net", subject=subject, channels=channels)
    loader = _build_loader(cfg)
    data, labels, _subjects, test_mask, meta = _assemble(loader, _subjects_to_load(cfg, loader))
    return data, labels, test_mask, meta


PROTOCOL = "minirocket_pooled_cv"


def _write_subject_run(seed_root, run_id, *, config, meta, channels_used, subject, seed,
                       n_pooled, n_folds, fold_metrics, fold_train_loss, subj_metrics, elapsed):
    """Write the standard per-subject/-seed run directory: result.json, manifest.json, history.json, and a .run_complete marker (via run_directory), matching the other validation scripts."""
    with run_directory(seed_root, run_id) as directory:
        # the pooled trials are the run's membership; per-fold test indices live in the config
        split = Split(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                      np.arange(n_pooled, dtype=np.int64),
                      {"protocol": PROTOCOL, "subject": subject, "seed": seed})
        write_run_manifest(directory, config=config, split=split, fold=f"S{subject:02d}",
                           channels_used=channels_used,
                           preprocessing_provenance=meta.get("preprocessing"))
        result = RunResult(
            run_id=run_id, dataset="bci_iv_2a", model="grf_net", preprocessing="standard",
            protocol=PROTOCOL, seed=seed, fold=f"S{subject:02d}", metrics=subj_metrics,
            n_trials={"train": n_pooled - n_pooled // n_folds, "val": 0, "test": n_pooled // n_folds},
            timings={"train_seconds": elapsed},
            config_fingerprint=config_fingerprint(config),
        )
        with atomic_write(directory / "result.json", mode="w", encoding="utf-8") as fh:
            fh.write(result.to_json(indent=2))
        history = {"fold_train_loss": fold_train_loss,
                   "fold_accuracies": [fm["accuracy"] for fm in fold_metrics],
                   "fold_metrics": fold_metrics}
        with atomic_write(directory / "history.json", mode="w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default="runs/grf_net-iv2a")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--folds", type=int, default=5, help="pooled stratified CV folds (trial level)")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", nargs="+", default=["all"], help="channel names, or 'all' / 'galea'")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=False,
                        help="apply per-session Euclidean Alignment before pooling (unsupervised, leakage-safe)")
    args = parser.parse_args(argv)

    channels = _resolve_channels(args.channels)
    tag = channels_tag(channels)
    per_seed = []
    for seed in args.seeds:
        seed_root = Path(args.output_root) / f"seed{seed}"
        subj_rows = []
        for subject in args.subjects:
            data, labels, test_mask, meta = _pooled_subject_data(subject, channels)
            if args.align:
                # align each session (T vs E) by its own reference before pooling
                data = align_sessions(data, np.asarray(test_mask, dtype=int))
            n_classes = len(meta["class_names"])
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            fold_metrics, fold_train_loss, fold_test_idx = [], [], []
            start = time.perf_counter()
            print(f"seed {seed} S{subject:02d}: training {args.folds} folds x {args.epochs} epochs "
                  f"({len(data)} trials, {tag})...", flush=True)
            for fold, (tr_idx, te_idx) in enumerate(skf.split(data, labels)):
                x_tr = torch.tensor(data[tr_idx], dtype=torch.float32)
                y_tr = torch.tensor(labels[tr_idx], dtype=torch.long)
                x_te = torch.tensor(data[te_idx], dtype=torch.float32)
                preds, train_loss = _train_fold(x_tr, y_tr, x_te, sfreq=meta["sfreq"], n_classes=n_classes,
                                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                    seed=seed, device=args.device,
                                    desc=f"S{subject:02d} fold {fold + 1}/{args.folds}")
                fold_metrics.append(compute_metrics(labels[te_idx], preds))
                fold_train_loss.append(train_loss)
                fold_test_idx.append([int(i) for i in te_idx])
            subj = {m: float(np.mean([fm[m] for fm in fold_metrics])) for m in METRICS}
            elapsed = round(time.perf_counter() - start, 1)
            subj_rows.append({"subject": subject, **subj,
                              "folds": [round(fm["accuracy"], 4) for fm in fold_metrics],
                              "train_seconds": elapsed})
            run_id = f"bci_iv_2a_standard_grf_net_{PROTOCOL}_{tag}_S{subject:02d}"
            config = {"dataset": "bci_iv_2a", "preprocessing": "standard", "protocol": PROTOCOL,
                      "model": "grf_net", "seed": seed, "subject": subject, "channels": tag,
                      "n_folds": args.folds, "epochs": args.epochs, "lr": args.lr,
                      "euclidean_alignment": bool(args.align), "cv_fold_test_indices": fold_test_idx}
            _write_subject_run(seed_root, run_id, config=config, meta=meta, channels_used=meta["channel_names"],
                               subject=subject, seed=seed, n_pooled=len(data), n_folds=args.folds,
                               fold_metrics=fold_metrics, fold_train_loss=fold_train_loss,
                               subj_metrics=subj, elapsed=elapsed)
            print(f"seed {seed} S{subject:02d} acc={subj['accuracy']:.4f} kappa={subj['cohen_kappa']:.4f} "
                  f"folds={subj_rows[-1]['folds']}", flush=True)
        cohort = {m: float(np.mean([r[m] for r in subj_rows])) for m in METRICS}
        cohort_sd = {m: float(np.std([r[m] for r in subj_rows])) for m in METRICS}
        per_seed.append(cohort["accuracy"])
        summary = {
            "protocol": f"minirocket pooled-session, {args.folds}-fold trial-level CV (original-methodology build check; not the grid protocol)",
            "reference": "MiniRocket 92.57% on 2a, Hwaidi and Ghanem (arXiv 2508.16179), pooled sessions 50:20:30",
            "seed": seed, "channels": tag, "epochs": args.epochs,
            "euclidean_alignment": bool(args.align),
            "per_subject": subj_rows, "cohort_mean": cohort, "cohort_sd": cohort_sd,
        }
        seed_root.mkdir(parents=True, exist_ok=True)
        with atomic_write(seed_root / "summary.json", mode="w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"seed {seed} cohort mean acc={cohort['accuracy']:.4f} (sd {cohort_sd['accuracy']:.4f})")

    print(f"overall mean acc={float(np.mean(per_seed)):.4f} +/- {float(np.std(per_seed)):.4f}")


if __name__ == "__main__":
    main()
