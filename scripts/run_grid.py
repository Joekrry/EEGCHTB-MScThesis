"""Run (or dry-run) the benchmark grid.

Usage:
    python -m scripts.run_grid configs/grid_iv2a.yaml runs/grid-iv2a            # dry run
    python -m scripts.run_grid configs/grid_iv2a.yaml runs/grid-iv2a/b --execute  # run it
"""

from __future__ import annotations

import argparse
import dataclasses
import signal
from pathlib import Path

from tqdm.auto import tqdm

from eegchtb.experiments import (
    MONTAGES, cell_dir, enumerate_cells, is_cell_complete, load_grid, plan,
    resolve_seeds, runs_per_cell, validate_grid, write_cell_summary, write_execution_manifest,
)
from eegchtb.experiments.grid import SUMMARY_FILE
from eegchtb.run_output import COMPLETION_MARKER, is_complete

# grid YAML axis -> the CLI flag that restricts it to a chosen subset
FILTER_AXES = ("models", "preprocessings", "datasets", "evaluations", "montages", "seeds")


def install_stop_handler(stop: dict) -> None:
    """Graceful shutdown on Ctrl-C """
    def handler(signum, frame):
        if stop["requested"]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        stop["requested"] = True
        print("\nstop requested: finishing the current run, then exiting cleanly "
              "(press Ctrl-C again to abort now)", flush=True)
    signal.signal(signal.SIGINT, handler)


def _grid_trainer(model, train_ds, val_ds, test_ds, config):
    """Train one run through the shared grid recipe (uniform Adam+cosine, best-val, shared augmentation)."""
    from eegchtb.training import grid_train_config, make_augment, predict, train_model

    x_all, y_all = train_ds.tensors()
    n_classes = int(y_all.max().item()) + 1
    augment = make_augment(x_all, y_all, seed=config.seed, n_classes=n_classes)
    tcfg = grid_train_config(seed=config.seed, device=config.device,
                             epochs=config.epochs, batch_size=config.batch_size, augment=augment)
    # transient per-run epoch bar (nested under the grid-cells bar)
    tcfg = dataclasses.replace(tcfg, desc=f"{config.model} S{config.subject:02d}", leave_progress=False)
    result = train_model(model, train_ds, val_ds, tcfg)
    preds = predict(model, test_ds, device=config.device)
    return preds, result.timings, result.history


def _execute_cell(cell, root, device, data_root, subjects_filter=None, save_snapshots=True, stop=None):
    """Run a cell's subjects; return True if the cell finished (so it gets a
    .run_complete marker), False if a stop request ended it mid-cell."""
    from eegchtb.data import get_loader
    from eegchtb.experiment import ExperimentConfig, run_experiment
    from eegchtb.training import GRID_VAL_FRACTION
    from eegchtb.training.recipes import GRID_BATCH_SIZE, GRID_EPOCHS

    directory = cell_dir(root, cell)
    directory.mkdir(parents=True, exist_ok=True)
    channels = MONTAGES[cell.montage]
    subjects = get_loader(cell.dataset, root=data_root).available_subjects()
    if subjects_filter is not None:
        subjects = [s for s in subjects if s in subjects_filter]
    if not subjects:
        raise SystemExit(f"no subjects to run for {cell.dataset!r} under {data_root!r} "
                         "(dataset missing, or --subjects excluded them all)")
    for subject in tqdm(subjects, desc=cell.run_id(), unit="subj", leave=False):
        if stop is not None and stop["requested"]:
            return False  # graceful stop between runs: leave the cell unmarked to resume
        run_id = f"S{subject:02d}"
        if is_complete(directory / run_id):
            continue  # resume: this subject already done
        config = ExperimentConfig(
            dataset=cell.dataset, preprocessing=cell.preprocessing, protocol=cell.evaluation,
            model=cell.model, seed=cell.seed, subject=subject, channels=channels,
            data_root=data_root, val_fraction=GRID_VAL_FRACTION, epochs=GRID_EPOCHS,
            batch_size=GRID_BATCH_SIZE, device=device,
        )
        run_experiment(config, directory, run_id=run_id, trainer=_grid_trainer,
                       save_snapshot=save_snapshots)
    # mean/sd across the cell's nine subjects, written before the completion marker so
    # a cell is never marked complete without its summary
    write_cell_summary(root, cell)
    (directory / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
    return True


def _backfill_cell_summaries(grid, root) -> int:
    """Cells already marked complete before this feature existed (or from any other
    gap) get their summary.json filled in retroactively; cheap, read-only over
    existing result.json files. Returns the number backfilled."""
    n = 0
    for cell in enumerate_cells(grid):
        if is_cell_complete(root, cell) and not (cell_dir(root, cell) / SUMMARY_FILE).exists():
            write_cell_summary(root, cell)
            n += 1
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", nargs="?", default="configs/grid_iv2a.yaml")
    parser.add_argument("output_root", nargs="?", default="runs/grid-iv2a")
    parser.add_argument("--execute", action="store_true", help="run pending cells (default: dry run)")
    parser.add_argument("--random-seeds", type=int, default=None, metavar="N",
                        help="use N random seeds (drawn once, persisted to seeds.json, shared by every cell) "
                             "instead of the config's fixed seed list")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="eegchtb/data", help="dataset root passed to the loaders")
    parser.add_argument("--save-snapshots", action=argparse.BooleanOptionalAction, default=True,
                        help="save each run's trained model as model.pt for reload/replication (default: on)")
    # select an exact subset to run: each flag restricts one grid axis to the given values
    parser.add_argument("--models", nargs="+", help="restrict the models axis")
    parser.add_argument("--preprocessings", nargs="+", help="restrict the preprocessings axis")
    parser.add_argument("--datasets", nargs="+", help="restrict the datasets axis")
    parser.add_argument("--evaluations", nargs="+", help="restrict the evaluations axis")
    parser.add_argument("--montages", nargs="+", help="restrict the montages axis")
    parser.add_argument("--seeds", type=int, nargs="+", help="restrict the seeds axis")
    parser.add_argument("--subjects", type=int, nargs="+", help="restrict which subjects run per cell")
    args = parser.parse_args(argv)

    grid = load_grid(args.grid)
    if args.random_seeds is not None and args.seeds is not None:
        raise SystemExit("use either --random-seeds or --seeds, not both")
    if args.random_seeds is not None:
        seeds = resolve_seeds(args.output_root, args.random_seeds)
        grid["seeds"] = seeds
        print(f"random seeds ({len(seeds)}, shared across all cells, persisted to seeds.json): {seeds}")
    # apply the subset filters: any axis flag given replaces that axis
    for axis in FILTER_AXES:
        chosen = getattr(args, axis)
        if chosen is not None:
            grid[axis] = chosen
    problems = validate_grid(grid)
    p = plan(grid, args.output_root)
    print(f"grid: {p['n_cells']} cells ({p['n_complete_cells']} complete, {p['n_pending_cells']} pending)")
    print(f"runs (one model per subject): {p['total_runs']} total  |  {p['done_runs']} done  |  {p['pending_runs']} pending")
    if problems:
        print("PREREQUISITE PROBLEMS:")
        for msg in problems:
            print(f"  - {msg}")

    if not args.execute:
        manifest = write_execution_manifest(grid, args.output_root)
        print(f"dry run: execution manifest written to {manifest}")
        print("re-run with --execute to run the pending cells")
        return

    if problems:
        raise SystemExit("refusing to execute: fix the prerequisite problems above")
    backfilled = _backfill_cell_summaries(grid, args.output_root)
    if backfilled:
        print(f"backfilled summary.json for {backfilled} already-complete cell(s)")
    stop = {"requested": False}
    install_stop_handler(stop)
    cells = [c for c in enumerate_cells(grid) if not is_cell_complete(args.output_root, c)]
    for cell in tqdm(cells, desc="grid cells", unit="cell"):
        if stop["requested"]:
            break
        _execute_cell(cell, args.output_root, args.device, args.data_root,
                      subjects_filter=args.subjects, save_snapshots=args.save_snapshots, stop=stop)
    write_execution_manifest(grid, args.output_root)
    if stop["requested"]:
        q = plan(grid, args.output_root)
        print(f"\nstopped cleanly: progress saved. {q['done_runs']}/{q['total_runs']} runs done, "
              f"{q['pending_runs']} pending across {q['n_pending_cells']} unfinished cells; "
              "re-run the same command to resume.")
    else:
        print("grid complete")


if __name__ == "__main__":
    main()
