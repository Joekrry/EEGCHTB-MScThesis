"""Benchmark-grid enumeration, validation and tracking. """

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.channels import GALEA_EEG_CHANNELS
from ..evaluation.metrics import METRIC_NAMES

SEEDS_FILE = "seeds.json"
SUMMARY_FILE = "summary.json"
_SEED_UPPER = 2**31  # seeds fit a 32-bit signed int, safe for numpy/torch
from ..run_output import atomic_write, is_complete

MONTAGES: dict[str, list[str] | None] = {"all": None, "galea": list(GALEA_EEG_CHANNELS)}
SUBJECT_COUNTS = {"bci_iv_2a": 9, "bci_iv_2b": 9}
EVALUATIONS = ("subject_specific",)
AXES = ("models", "preprocessings", "datasets", "evaluations", "montages", "seeds")

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GridCell:
    """One fully-specified benchmark cell; expands to one trained model per subject."""

    model: str
    preprocessing: str
    dataset: str
    evaluation: str
    montage: str
    seed: int

    def run_id(self) -> str:
        return (f"{self.dataset}_{self.preprocessing}_{self.model}"
                f"_{self.evaluation}_{self.montage}_seed{self.seed}")


def load_grid(path) -> dict:
    """Load and shape-check a grid YAML: every axis must be present."""
    import yaml

    with open(path, encoding="utf-8") as f:
        grid = yaml.safe_load(f)
    missing = [a for a in AXES if a not in grid]
    if missing:
        raise ValueError(f"grid config missing axes: {missing}")
    return grid


def enumerate_cells(grid: dict) -> list[GridCell]:
    """Every cell of the fully-crossed matrix, in a deterministic axis order."""
    return [
        GridCell(model, prep, dataset, evaluation, montage, int(seed))
        for model in grid["models"]
        for prep in grid["preprocessings"]
        for dataset in grid["datasets"]
        for evaluation in grid["evaluations"]
        for montage in grid["montages"]
        for seed in grid["seeds"]
    ]


def cell_dir(root, cell: GridCell) -> Path:
    return Path(root) / cell.run_id()


def is_cell_complete(root, cell: GridCell) -> bool:
    """A cell is complete when its directory carries the .run_complete marker."""
    return is_complete(cell_dir(root, cell))


def runs_per_cell(cell: GridCell) -> int:
    """Trained models a cell expands to 0 if the dataset is unknown."""
    return SUBJECT_COUNTS.get(cell.dataset, 0)


def completed_runs_in_cell(root, cell: GridCell) -> int:
    """How many of a cell's per-subject runs have finished (each carries its own .run_complete), so a partially-done cell is counted at subject granularity on resume."""
    directory = cell_dir(root, cell)
    if not directory.exists():
        return 0
    return sum(1 for sub in directory.iterdir() if sub.is_dir() and is_complete(sub))


def aggregate_cell_metrics(root, cell: GridCell) -> dict:
    """Per-metric mean and sd across a cell's finished per-subject runs (one seed, one
    model/preprocessing/dataset/evaluation/montage cell -> nine subjects)."""
    directory = cell_dir(root, cell)
    per_subject = []
    for sub in sorted(directory.iterdir()) if directory.exists() else []:
        if sub.is_dir() and is_complete(sub) and (sub / "result.json").exists():
            metrics = json.loads((sub / "result.json").read_text(encoding="utf-8"))["metrics"]
            per_subject.append({"subject": sub.name, **metrics})
    values = {m: np.array([row[m] for row in per_subject], dtype=float) for m in METRIC_NAMES}
    return {
        "cell": cell.run_id(),
        "n_subjects": len(per_subject),
        "metrics_mean": {m: float(values[m].mean()) if len(per_subject) else None for m in METRIC_NAMES},
        "metrics_sd": {m: float(values[m].std()) if len(per_subject) else None for m in METRIC_NAMES},
        "per_subject": per_subject,
    }


def write_cell_summary(root, cell: GridCell, path=None) -> Path:
    """Compute and atomically persist a cell's mean-across-subjects summary.json."""
    summary = aggregate_cell_metrics(root, cell)
    out = Path(path) if path is not None else cell_dir(root, cell) / SUMMARY_FILE
    with atomic_write(out, mode="w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out


def validate_grid(grid: dict) -> list[str]:
    """Prerequisite problems that would make the grid unrunnable (empty list == ok)."""
    from ..experiment import MODELS

    problems: list[str] = []
    for model in grid["models"]:
        if model not in MODELS:
            problems.append(f"unknown model: {model!r} (have {sorted(MODELS)})")
    for dataset in grid["datasets"]:
        if dataset not in SUBJECT_COUNTS:
            problems.append(f"unknown/uncounted dataset: {dataset!r}")
    for evaluation in grid["evaluations"]:
        if evaluation not in EVALUATIONS:
            problems.append(f"unknown evaluation: {evaluation!r} (have {EVALUATIONS})")
    for montage in grid["montages"]:
        if montage not in MONTAGES:
            problems.append(f"unknown montage: {montage!r} (have {sorted(MONTAGES)})")
    for prep in grid["preprocessings"]:
        if not (REPO_ROOT / "configs" / "preprocessing" / f"{prep}.yaml").exists():
            problems.append(f"missing preprocessing config: configs/preprocessing/{prep}.yaml")
    return problems


def plan(grid: dict, root) -> dict:
    """Resumable execution plan: cell/run counts and which cells are done vs pending."""
    cells = enumerate_cells(grid)
    complete = [c for c in cells if is_cell_complete(root, c)]
    pending = [c for c in cells if not is_cell_complete(root, c)]
    total_runs = sum(runs_per_cell(c) for c in cells)
    # a complete cell counts as all its runs done; an incomplete cell counts its
    # finished per-subject runs, so partially-done cells still reduce the remaining
    # count on resume (not just fully-complete cells)
    done_runs = sum(
        runs_per_cell(c) if is_cell_complete(root, c) else completed_runs_in_cell(root, c)
        for c in cells
    )
    return {
        "n_cells": len(cells),
        "n_complete_cells": len(complete),
        "n_pending_cells": len(pending),
        "total_runs": total_runs,
        "done_runs": done_runs,
        "pending_runs": total_runs - done_runs,
        "pending_cells": [c.run_id() for c in pending],
    }


def write_execution_manifest(grid: dict, root, path=None) -> Path:
    """Write the enumerated plan to an atomic manifest."""
    manifest = {"axes": {a: grid[a] for a in AXES}, **plan(grid, root)}
    out = Path(path) if path is not None else Path(root) / "grid_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out, mode="w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def random_seeds(n: int) -> list[int]:
    """n distinct random seeds in [0, 2**31)."""
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(secrets.randbelow(_SEED_UPPER))
    return sorted(seeds)


def resolve_seeds(root, n_random: int) -> list[int]:
    """The random seed set for this grid: reuse the one persisted under root, else draw
    n_random seeds and persist them. Drawn once and shared by every cell, so all models
    train on the same seeds and the set is reproducible."""
    path = Path(root) / SEEDS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    seeds = random_seeds(n_random)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(seeds, fh)
    return seeds
