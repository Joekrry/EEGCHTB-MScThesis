"""Single-experiment runner: one config selects dataset, preprocessing, protocol, model and seed, and produces one self-cleaning run directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data.base import EpochWindow
from .data.channels import GALEA_EEG_CHANNELS
from .data.epochs import EpochBundle
from .evaluation import (
    EpochBundleDataset,
    RunResult,
    compute_metrics,
    config_fingerprint,
    subject_specific,
    write_run_manifest,
)
from .models import EEGConformer, EEGNet, GRFNet
from .run_output import atomic_write, new_run_id, run_directory
from .training import TrainConfig, predict, set_seed, train_model

MODELS = {
    "eegnet": EEGNet,
    "eeg_conformer": EEGConformer,
    "grf_net": GRFNet,
}

# Models whose stem sizes derive from the sampling rate (rate-adaptive) and that
# support geometry-seeded spatial init, so the runner passes sfreq / coords through.
_RATE_ADAPTIVE = ("grf_net",)


@dataclass
class ExperimentConfig:
    """Everything that identifies one benchmark run."""

    dataset: str  # "bci_iv_2a" | "bci_iv_2b"
    preprocessing: str  # pipeline config name
    protocol: str  # "subject_specific"
    model: str = "eegnet"
    seed: int = 0
    subject: int | None = None  # which fold/test subject to run (first if None)
    # trial epoching window; a first-order determinant of accuracy, so it is part
    # of the run's identity and recorded in the manifest and fingerprint.
    window: EpochWindow = field(default_factory=EpochWindow)
    data_root: str = "eegchtb/data"
    epochs: int = 200
    batch_size: int = 64
    val_fraction: float = 0.1
    patience: int = 20
    device: str = "auto"  # CUDA when available, else CPU
    # EEG channels kept from every dataset (matched by name); the 10 Galea
    # channels by default, so results extrapolate to the Galea headset. None
    # keeps all channels.
    channels: list[str] | None = field(
        default_factory=lambda: list(GALEA_EEG_CHANNELS)
    )
    model_kwargs: dict = field(default_factory=dict)  # overrides the derived defaults


def _build_loader(config: ExperimentConfig):
    from .data import get_loader

    return get_loader(
        config.dataset,
        root=config.data_root,
        window=config.window,
        preprocessing=config.preprocessing,
        channels=config.channels,
    )


def _subjects_to_load(config: ExperimentConfig, loader) -> list[int]:
    """Only a subject-specific run on a named subject can skip loading the rest of the cohort."""
    if config.protocol == "subject_specific" and config.subject is not None:
        return [config.subject]
    return loader.available_subjects()


def _assemble(loader, subjects_to_load=None):
    """Load the given sessions into combined arrays with per-trial subject and an eval-session test mask."""
    datas, labels, subjects, test_mask = [], [], [], []
    meta: dict = {}
    for subject in (subjects_to_load if subjects_to_load is not None else loader.available_subjects()):
        for session in loader.available_sessions(subject):
            bundle = loader.load_session(subject, session).filter_known()
            if bundle.n_trials == 0:
                continue
            datas.append(bundle.data)
            labels.append(bundle.labels)
            subjects.append(np.full(bundle.n_trials, subject, dtype=np.int64))
            test_mask.append(np.full(bundle.n_trials, session.endswith("E"), dtype=bool))
            meta = {
                "channel_names": list(bundle.channel_names),
                "class_names": dict(bundle.class_names),
                "sfreq": bundle.sfreq,
                # what preprocessing resolved to at run time (e.g. the Hjorth
                # Laplacian neighbour graph); identical across a run's sessions,
                # which share one montage
                "preprocessing": bundle.metadata.get("preprocessing"),
            }
    if not datas:
        raise ValueError("no trials assembled from the loader")
    return (
        np.concatenate(datas),
        np.concatenate(labels),
        np.concatenate(subjects),
        np.concatenate(test_mask),
        meta,
    )


def _select_split(config: ExperimentConfig, subjects, labels, test_mask):
    if config.protocol != "subject_specific":
        raise ValueError(f"unknown protocol {config.protocol!r}")
    splits = subject_specific(
        subjects, test_mask, labels, seed=config.seed, val_fraction=config.val_fraction
    )

    if config.subject is None:
        return splits[0]
    for split in splits:
        if split.meta["subject"] == config.subject:
            return split
    raise ValueError(f"subject {config.subject} not found for protocol {config.protocol}")


def _subset(data, labels, idx, meta, dataset) -> EpochBundle:
    return EpochBundle(
        data=data[idx],
        labels=labels[idx],
        dataset=dataset,
        subject=-1,
        session="split",
        sfreq=meta["sfreq"],
        channel_names=meta["channel_names"],
        class_names=meta["class_names"],
    )


def _model_kwargs(config: ExperimentConfig, meta: dict, n_channels, n_times, n_classes) -> dict:
    """Model constructor arguments, with reference-prescribed defaults derived from the data."""
    kwargs = {"n_channels": n_channels, "n_times": n_times, "n_classes": n_classes}
    if config.model == "eegnet":
        # EEGNet's reference guidance is a temporal kernel of half the sampling
        # rate (the published default of 64 assumes 128 Hz data), so derive it
        # from the pipeline's actual output rate rather than hardcoding it.
        kwargs["kernel_length"] = max(1, int(round(meta["sfreq"] / 2)))
    elif config.model in _RATE_ADAPTIVE:
        # GRF-Net's stem derives its dilation and pooling sizes from the sampling
        # rate, so pass the pipeline's output rate through.
        kwargs["sfreq"] = meta["sfreq"]
    kwargs.update(config.model_kwargs)
    # Geometry-seeded spatial init needs the montage coordinates, resolved from the
    # channel names the run actually used rather than requested through the config.
    if config.model in _RATE_ADAPTIVE and kwargs.get("spatial_init") == "geometry" and "channel_coords" not in kwargs:
        from .data.channels import channel_positions

        kwargs["channel_coords"] = channel_positions(meta["channel_names"])
    return kwargs


def _config_dict(config: ExperimentConfig) -> dict:
    cfg = {
        "dataset": config.dataset,
        "preprocessing": config.preprocessing,
        "protocol": config.protocol,
        "model": config.model,
        "seed": config.seed,
        "subject": config.subject,
        "channels": config.channels,
        "window": asdict(config.window),
    }
    # only record model overrides when present, so the existing baselines' run
    # fingerprints (empty model_kwargs) are unchanged while GRF-Net ablation cells,
    # which differ only in model_kwargs, still get distinct fingerprints.
    if config.model_kwargs:
        cfg["model_kwargs"] = config.model_kwargs
    return cfg


def channels_tag(channels) -> str:
    """A short montage label for run names: "galea", "all", or "<n>ch" for a custom subset."""
    if channels is None:
        return "all"
    from .data.channels import GALEA_EEG_CHANNELS

    if list(channels) == list(GALEA_EEG_CHANNELS):
        return "galea"
    return f"{len(channels)}ch"


def run_name(config: ExperimentConfig) -> str:
    """Self-describing run name from every identifying field (dataset, preprocessing, protocol, fold, model, channels, seed)."""
    fold = f"S{config.subject:02d}" if config.subject is not None else "all"
    return (
        f"{config.dataset}_{config.preprocessing}_{config.protocol}_{fold}"
        f"_{config.model}_{channels_tag(config.channels)}_seed{config.seed}"
    )


def _default_run_id(config: ExperimentConfig) -> str:
    """The self-describing run name plus a timestamp, for uniqueness across repeats."""
    return new_run_id(run_name(config))


def _write_snapshot(path, model, model_kwargs, config, meta, n_channels, n_times, n_classes, metrics) -> None:
    """Save the trained best-epoch model and everything needed to rebuild it, atomically. Weights are moved to CPU so the snapshot reloads on any device."""
    import torch

    snapshot = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model": config.model,
        "model_kwargs": model_kwargs,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "channel_names": list(meta["channel_names"]),
        "sfreq": meta["sfreq"],
        "dataset": config.dataset,
        "seed": config.seed,
        "metrics": metrics,
    }
    with atomic_write(path) as handle:
        torch.save(snapshot, handle)


def load_snapshot(path):
    """Rebuild a model from a run's model.pt snapshot and load its trained weights; returns (eval-mode model, snapshot dict)."""
    import torch

    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    model = MODELS[snapshot["model"]](**snapshot["model_kwargs"])
    model.load_state_dict(snapshot["state_dict"])
    model.eval()
    return model, snapshot


def run_experiment(
    config: ExperimentConfig,
    output_root,
    *,
    loader=None,
    run_id: str | None = None,
    trainer=None,
    save_snapshot: bool = False,
) -> Path:
    """Run one experiment end to end inside a self-cleaning run directory; return its path. save_snapshot writes the trained best-epoch model as model.pt in the run directory."""
    loader = loader or _build_loader(config)
    run_id = run_id or _default_run_id(config)

    with run_directory(output_root, run_id) as directory:
        data, labels, subjects, test_mask, meta = _assemble(
            loader, _subjects_to_load(config, loader)
        )
        split = _select_split(config, subjects, labels, test_mask)
        n_channels, n_times = data.shape[1], data.shape[2]
        n_classes = len(meta["class_names"])

        train_ds = EpochBundleDataset(_subset(data, labels, split.train, meta, config.dataset))
        val_ds = (
            EpochBundleDataset(_subset(data, labels, split.val, meta, config.dataset))
            if len(split.val) > 0
            else None
        )
        test_ds = EpochBundleDataset(_subset(data, labels, split.test, meta, config.dataset))

        set_seed(config.seed)
        model_kwargs = _model_kwargs(config, meta, n_channels, n_times, n_classes)
        model = MODELS[config.model](**model_kwargs)
        if trainer is not None:
            preds, timings, history = trainer(model, train_ds, val_ds, test_ds, config)
        else:
            train_result = train_model(
                model,
                train_ds,
                val_ds,
                TrainConfig(
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    seed=config.seed,
                    patience=config.patience,
                    device=config.device,
                    desc=run_id,
                ),
            )
            preds = predict(model, test_ds)
            timings, history = train_result.timings, train_result.history
        metrics = compute_metrics(labels[split.test], preds)

        cfg = _config_dict(config)
        # the resolved preprocessing (every step's effective params, not just the
        # config name) so editing a YAML changes the fingerprint and the record
        pipeline = getattr(loader, "preprocessing", None)
        cfg["preprocessing_resolved"] = (
            pipeline.resolved_config() if pipeline is not None else None
        )
        write_run_manifest(
            directory,
            config=cfg,
            split=split,
            channels_used=meta["channel_names"],
            preprocessing_provenance=meta.get("preprocessing"),
        )
        result = RunResult(
            run_id=run_id,
            dataset=config.dataset,
            model=config.model,
            preprocessing=config.preprocessing,
            protocol=config.protocol,
            seed=config.seed,
            fold=str(split.meta.get("subject", "")),
            metrics=metrics,
            n_trials={"train": len(split.train), "val": len(split.val), "test": len(split.test)},
            timings=timings,
            config_fingerprint=config_fingerprint(cfg),
        )
        with atomic_write(directory / "result.json", mode="w", encoding="utf-8") as handle:
            handle.write(result.to_json(indent=2))
        # per-epoch learning curves, kept beside the metrics for later plotting
        with atomic_write(directory / "history.json", mode="w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if save_snapshot:
            _write_snapshot(
                directory / "model.pt", model, model_kwargs, config, meta,
                n_channels, n_times, n_classes, metrics,
            )

    return Path(output_root) / run_id


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one EEG benchmark experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--protocol", required=True, choices=["subject_specific"])
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--data-root", default="eegchtb/data")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "EEG channels to keep (default: the 10 Galea channels). "
            "Pass 'all' to keep every channel."
        ),
    )
    parser.add_argument("--tmin", type=float, default=0.0, help="epoch start (s) relative to the cue")
    parser.add_argument("--tmax", type=float, default=4.0, help="epoch end (s) relative to the cue")
    parser.add_argument(
        "--baseline",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="baseline-correction window (s); omit for none",
    )
    parser.add_argument(
        "--drop-rejected",
        action="store_true",
        help="drop trials overlapping an artifact-rejection mark",
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the trained model as model.pt in the run directory (default: on)",
    )
    args = parser.parse_args(argv)

    if args.channels is None:
        channels: list[str] | None = list(GALEA_EEG_CHANNELS)
    elif len(args.channels) == 1 and args.channels[0].lower() == "all":
        channels = None
    else:
        channels = args.channels

    window = EpochWindow(
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=tuple(args.baseline) if args.baseline is not None else None,
        drop_rejected=args.drop_rejected,
    )

    config = ExperimentConfig(
        dataset=args.dataset,
        preprocessing=args.preprocessing,
        protocol=args.protocol,
        model=args.model,
        seed=args.seed,
        subject=args.subject,
        window=window,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        channels=channels,
    )
    path = run_experiment(config, Path(args.output_root), save_snapshot=args.save_snapshots)
    print(f"run complete: {path}")


if __name__ == "__main__":
    main()
