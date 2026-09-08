"""Benchmark-grid enumeration, validation, resumability and manifest (no training)."""

from __future__ import annotations

import json

import pytest

from eegchtb.experiments import (
    aggregate_cell_metrics, cell_dir, enumerate_cells, is_cell_complete, load_grid,
    plan, random_seeds, resolve_seeds, runs_per_cell, validate_grid, write_cell_summary,
    write_execution_manifest,
)
from eegchtb.experiments.grid import REPO_ROOT, SUMMARY_FILE
from eegchtb.run_output import COMPLETION_MARKER

GOOD = {
    "models": ["eegnet", "grf_net"],
    "preprocessings": ["standard", "minimal"],
    "datasets": ["bci_iv_2a"],
    "evaluations": ["subject_specific"],
    "montages": ["all", "galea"],
    "seeds": [0, 1, 2],
}


def test_enumerate_is_full_cross_and_deterministic():
    cells = enumerate_cells(GOOD)
    assert len(cells) == 2 * 2 * 1 * 1 * 2 * 3  # 24
    assert enumerate_cells(GOOD) == cells       # deterministic order
    assert len({c.run_id() for c in cells}) == len(cells)  # unique ids


def test_runs_per_cell_and_total():
    cells = enumerate_cells(GOOD)
    assert all(runs_per_cell(c) == 9 for c in cells)
    assert plan(GOOD, "runs/none")["total_runs"] == 9 * len(cells)


def test_validate_flags_bad_axes():
    bad = {**GOOD, "models": ["eegnet", "nope"], "montages": ["all", "weird"],
           "evaluations": ["subject_specific", "kfold"], "preprocessings": ["ghost"]}
    problems = validate_grid(bad)
    joined = " ".join(problems)
    assert "nope" in joined and "weird" in joined and "kfold" in joined and "ghost" in joined
    assert validate_grid(GOOD) == []


def test_resumability(tmp_path):
    cells = enumerate_cells(GOOD)
    assert plan(GOOD, tmp_path)["n_pending_cells"] == len(cells)
    # mark one cell complete
    d = cell_dir(tmp_path, cells[0]); d.mkdir(parents=True)
    (d / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
    assert is_cell_complete(tmp_path, cells[0])
    p = plan(GOOD, tmp_path)
    assert p["n_complete_cells"] == 1
    assert p["n_pending_cells"] == len(cells) - 1
    assert p["pending_runs"] == p["total_runs"] - 9


def test_pending_runs_counts_finished_subjects(tmp_path):
    # a cell that is not yet marked complete but has some subject runs done must
    # reduce the remaining count on resume (subject granularity, not cell)
    one_cell = {**GOOD, "models": ["eegnet"], "preprocessings": ["standard"],
                "evaluations": ["subject_specific"], "montages": ["all"], "seeds": [0]}
    cell = enumerate_cells(one_cell)[0]
    d = cell_dir(tmp_path, cell)
    d.mkdir(parents=True)
    for s in range(1, 5):  # 4 of 9 subject runs finished
        sub = d / f"S{s:02d}"
        sub.mkdir()
        (sub / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
    p = plan(one_cell, tmp_path)
    assert p["total_runs"] == 9
    assert p["done_runs"] == 4
    assert p["pending_runs"] == 5           # not 9
    assert p["n_complete_cells"] == 0       # the cell itself is still unmarked


def _fake_completed_cell(tmp_path, accuracies):
    """A cell directory with one fake completed subject run per given accuracy
    (macro_f1/cohen_kappa/mcc scaled off it so mean/sd differ per metric)."""
    one_cell = {**GOOD, "models": ["eegnet"], "preprocessings": ["standard"],
                "evaluations": ["subject_specific"], "montages": ["all"], "seeds": [0]}
    cell = enumerate_cells(one_cell)[0]
    d = cell_dir(tmp_path, cell)
    d.mkdir(parents=True)
    for i, acc in enumerate(accuracies, start=1):
        sub = d / f"S{i:02d}"
        sub.mkdir()
        metrics = {"accuracy": acc, "macro_f1": acc - 0.1, "cohen_kappa": acc - 0.2, "mcc": acc - 0.2}
        (sub / "result.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
        (sub / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
    return cell


def test_aggregate_cell_metrics_means_across_subjects(tmp_path):
    cell = _fake_completed_cell(tmp_path, [0.6, 0.8, 1.0])  # mean 0.8, sd ~0.1633
    summary = aggregate_cell_metrics(tmp_path, cell)
    assert summary["cell"] == cell.run_id()
    assert summary["n_subjects"] == 3
    assert summary["metrics_mean"]["accuracy"] == pytest.approx(0.8)
    assert summary["metrics_sd"]["accuracy"] == pytest.approx(0.163299, abs=1e-5)
    assert len(summary["per_subject"]) == 3


def test_aggregate_cell_metrics_empty_cell(tmp_path):
    one_cell = {**GOOD, "models": ["eegnet"], "preprocessings": ["standard"],
                "evaluations": ["subject_specific"], "montages": ["all"], "seeds": [0]}
    cell = enumerate_cells(one_cell)[0]
    summary = aggregate_cell_metrics(tmp_path, cell)  # cell dir doesn't even exist yet
    assert summary["n_subjects"] == 0
    assert summary["metrics_mean"]["accuracy"] is None


def test_write_cell_summary_persists_and_matches_aggregate(tmp_path):
    cell = _fake_completed_cell(tmp_path, [0.5, 0.9])
    out = write_cell_summary(tmp_path, cell)
    assert out == cell_dir(tmp_path, cell) / SUMMARY_FILE
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == aggregate_cell_metrics(tmp_path, cell)


def test_backfill_cell_summaries_only_touches_complete_cells_missing_one(tmp_path):
    from scripts.run_grid import _backfill_cell_summaries

    one_model = {**GOOD, "models": ["eegnet"], "preprocessings": ["standard"],
                "evaluations": ["subject_specific"], "montages": ["all"], "seeds": [0]}
    cell = _fake_completed_cell(tmp_path, [0.7, 0.9])
    (cell_dir(tmp_path, cell) / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
    assert not (cell_dir(tmp_path, cell) / SUMMARY_FILE).exists()

    n = _backfill_cell_summaries(one_model, tmp_path)
    assert n == 1
    assert (cell_dir(tmp_path, cell) / SUMMARY_FILE).exists()
    assert _backfill_cell_summaries(one_model, tmp_path) == 0  # idempotent, nothing left to backfill


def test_execution_manifest_written(tmp_path):
    out = write_execution_manifest(GOOD, tmp_path)
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["axes"]["models"] == GOOD["models"]
    assert m["n_cells"] == len(enumerate_cells(GOOD))


@pytest.mark.parametrize(
    "name", ["grid_iv2a.yaml", "grid_iv2a_22ch.yaml", "grid_iv2b.yaml", "grid_iv2b_22ch.yaml"])
def test_shipped_grid_configs_are_valid(name):
    grid = load_grid(REPO_ROOT / "configs" / name)
    assert validate_grid(grid) == []


def test_iv2a_primary_grid_is_galea_only():
    # the primary 2a grid is the 4-channel Galea intersection; the 22-channel montage
    # is the separate optional grid
    primary = load_grid(REPO_ROOT / "configs" / "grid_iv2a.yaml")
    full = load_grid(REPO_ROOT / "configs" / "grid_iv2a_22ch.yaml")
    assert primary["montages"] == ["galea"]
    assert full["montages"] == ["all"]


def test_stop_handler_sets_flag_on_first_signal():
    import signal

    from scripts.run_grid import install_stop_handler

    stop = {"requested": False}
    previous = signal.getsignal(signal.SIGINT)
    try:
        install_stop_handler(stop)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)  # simulate one Ctrl-C
        assert stop["requested"] is True  # graceful: flag set, no exception raised
    finally:
        signal.signal(signal.SIGINT, previous)


def test_random_seeds_are_distinct():
    seeds = random_seeds(5)
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert all(0 <= s < 2**31 for s in seeds)


def test_resolve_seeds_persists_and_reuses(tmp_path):
    first = resolve_seeds(tmp_path, 5)
    assert (tmp_path / "seeds.json").exists()
    # a second call reuses the persisted set (so re-runs resume the same cells)
    assert resolve_seeds(tmp_path, 5) == first
    # and every cell shares that set -> paired-across-seed comparisons stay valid
    import json as _json
    assert _json.loads((tmp_path / "seeds.json").read_text()) == first


def test_grid_run_saves_a_loadable_snapshot(tmp_path):
    # the grid path (its own trainer + run_experiment) must write a model.pt that
    # load_snapshot can rebuild, so any run is reloadable and replicable
    pytest.importorskip("torch")
    from eegchtb.experiment import ExperimentConfig, load_snapshot, run_experiment
    from scripts.run_grid import _grid_trainer
    from test_experiment import _FakeLoader

    config = ExperimentConfig(dataset="fake", preprocessing="none", protocol="subject_specific",
                              model="eegnet", seed=0, subject=1, epochs=3, batch_size=4,
                              val_fraction=0.15)
    path = run_experiment(config, tmp_path, loader=_FakeLoader(), run_id="cell",
                          trainer=_grid_trainer, save_snapshot=True)
    assert (path / "model.pt").exists()
    model, snap = load_snapshot(path / "model.pt")
    assert snap["model"] == "eegnet" and "state_dict" in snap and snap["seed"] == 0


def test_iv2b_primary_grid_is_galea_only():
    # mirrors the 2a pairing: the primary 2b grid is the Galea montage (here identical
    # to "all" since Galea's C3/Cz/C4 IS 2b's full montage - kept as its own explicit
    # cell rather than assumed, so the grid record states the montage, not just infers
    # it), the 22ch/"all" montage is the separate optional grid
    primary = load_grid(REPO_ROOT / "configs" / "grid_iv2b.yaml")
    full = load_grid(REPO_ROOT / "configs" / "grid_iv2b_22ch.yaml")
    assert primary["montages"] == ["galea"]
    assert full["montages"] == ["all"]
    assert primary["datasets"] == full["datasets"] == ["bci_iv_2b"]
