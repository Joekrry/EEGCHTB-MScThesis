# Note from me
This is my MSc thesis project and can't be forked from my university account, I am porting this to my main GitHub account because I like to have ownership of my work and this somewhat immortalises it.


# EEGCHTB: EEG motor-imagery benchmarking framework

Benchmarks EEGNet, EEG-Conformer and a novel model (GRF-Net) for EEG motor-imagery
classification across four preprocessing pipelines and two datasets (BCI
Competition IV-2a/2b)

---

## Table of Contents

1. [Structure](#structure)
   - [Layers](#layers)
1. [Install](#install)
   - [Prerequisites](#prerequisites)
   - [Windows](#windows)
   - [Linux / macOS](#linux--macos)
   - [Docker](#docker)
1. [Usage](#usage)
   - [Python API](#python-api)
   - [One experiment](#one-experiment)
   - [Baseline validation](#baseline-validation)
   - [Benchmark grid](#benchmark-grid)
   - [Cross-subject and Galea transfer](#cross-subject-and-galea-transfer)
1. [Testing](#testing)
1. [Results](#results)
   - [EEGNet baseline](#eegnet-baseline)
   - [EEG-Conformer baseline](#eeg-conformer-baseline)
   - [GRF-Net baseline](#grf-net-baseline)
   - [Benchmark grid and statistics](#benchmark-grid-and-statistics)
1. [References](#references)

---

## Structure

```
eegchtb/
  run_output.py     atomic, self-cleaning run outputs
  data/              loaders (BCI IV-2a/2b GDF, Galea BDF), EpochBundle, channel restriction, alignment
  preprocessing/     config-driven pipeline steps on continuous mne.Raw
  evaluation/        dataset adapter, leakage-safe splits, metrics, manifests
  models/            eegnet.py, conformer.py, grf_net.py
  training/          seed control, trainer, shared grid recipe
  experiments/       benchmark-grid enumeration/tracking
  experiment.py      single-run driver, exposed as `eegchtb-run`
configs/
  preprocessing/     minimal, standard, deep, maximum, galea_clean
  grid_*.yaml        grid matrices (2a, 2a-22ch, 2b, 2b-22ch)
scripts/
  validate_eegnet_iv2a.py / validate_conformer_iv2a.py / validate_grf_net_iv2a.py
  run_grid.py          resumable benchmark grid runner
  eval_cross_subject.py  cross-subject control on grid snapshots
  eval_galea.py          score grid snapshots against real Galea BDF recordings
notebooks/statistical_testing.ipynb   descriptive + omnibus/posthoc stats over grid results
runs/                committed per-run evidence (manifests, results, snapshots)
tests/               215 tests, see Testing
```

Third-party: NumPy, SciPy, scikit-learn, MNE-Python, PyTorch (pinned to a CUDA
build), BrainFlow. Versions declared as floors in `pyproject.toml`, pinned in
`requirements.txt`.

### Layers

| Layer         | Folder                      | Responsibility                                             |
| ------------- | ---------------------------- | ------------------------------------------------------------ |
| Data          | `eegchtb/data/`               | Trial container, loaders, channel restriction, alignment.    |
| Preprocessing | `eegchtb/preprocessing/`      | Config-driven pipeline steps on the continuous signal.       |
| Evaluation    | `eegchtb/evaluation/`         | Dataset adapter, leakage-safe splits, metrics, manifests.    |
| Models        | `eegchtb/models/`              | EEGNet, EEG-Conformer, GRF-Net.                               |
| Training      | `eegchtb/training/`            | Seed control, deterministic trainer, shared grid recipe.     |
| Experiments   | `eegchtb/experiments/`         | Benchmark-grid enumeration, validation, tracking.             |
| Runner        | `eegchtb/experiment.py`        | Drives one config end to end; exposed as `eegchtb-run`.       |

## Install

### Prerequisites

| Requirement | Notes                                                             |
| ----------- | ------------------------------------------------------------------ |
| Python      | 3.11+.                                                              |
| git         | To clone the repository.                                           |
| Datasets    | BCI IV GDF files, gitignored - see the layout below.               |

### Windows

```powershell
git clone https://olympus.ntu.ac.uk/N1077433/EEGCHTB-MScThesis.git
cd EEGCHTB-MScThesis
py -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

### Linux / macOS

```bash
git clone https://olympus.ntu.ac.uk/N1077433/EEGCHTB-MScThesis.git
cd EEGCHTB-MScThesis
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` ends with `-e .`, so `eegchtb`, `eegchtb-run` and the tests work
from any directory. On Linux/macOS every command below becomes
`./env/bin/python <script>` with forward-slash paths (or just `python <script>`
once the venv is activated).

**Datasets** (gitignored) go under `eegchtb/data/`, or point `BCI_IV_2A_DIR` /
`BCI_IV_2B_DIR` at them:

```
eegchtb/data/bci_iv_2a/{train,test,truelabels}/  A01T.gdf ... A01E.mat ...
eegchtb/data/bci_iv_2b/{train,test,truelabels}/  B0101T.gdf ... B0104E.mat ...
```

### Docker

The `Dockerfile` builds a CUDA 13.0 / Python 3.11 image with `requirements.txt`
installed and `eegchtb-run` as the entrypoint. Datasets and run outputs are not
baked into the image - bind-mount them at run time.

Build the image from the repo root:

```bash
docker build -t eegchtb .
```

Run one experiment, mounting the BCI IV datasets and a `runs/` output directory
(use `` ` `` instead of `\` to continue lines in PowerShell):

```bash
docker run --rm --gpus all \
  -v /path/to/bci_iv_2a:/data/2a -e BCI_IV_2A_DIR=/data/2a \
  -v /path/to/bci_iv_2b:/data/2b -e BCI_IV_2B_DIR=/data/2b \
  -v "$(pwd)/runs:/workspace/runs" \
  eegchtb --dataset bci_iv_2a --preprocessing standard --protocol subject_specific --subject 1 --output-root runs
```

```powershell
docker run --rm --gpus all `
  -v C:\path\to\bci_iv_2a:/data/2a -e BCI_IV_2A_DIR=/data/2a `
  -v C:\path\to\bci_iv_2b:/data/2b -e BCI_IV_2B_DIR=/data/2b `
  -v ${PWD}\runs:/workspace/runs `
  eegchtb --dataset bci_iv_2a --preprocessing standard --protocol subject_specific --subject 1 --output-root runs
```

Drop `--gpus all` for a CPU-only run. To run a different entry point (a script
under `scripts/`, or `pytest`), override the entrypoint:

```bash
docker run --rm --entrypoint python eegchtb scripts/run_grid.py configs/grid_iv2a.yaml runs/grid-iv2a
docker run --rm --entrypoint python eegchtb -m pytest -q
```

`docker-compose.yml` wraps the same image. Set `BCI_IV_2A_DIR` / `BCI_IV_2B_DIR`
to your dataset paths (defaults to `eegchtb/data/bci_iv_2a`/`2b`) and run:

```bash
docker compose run --rm eegchtb
```

Edit the `command:` list in `docker-compose.yml` to change the default
experiment, or override it inline: `docker compose run --rm eegchtb --dataset bci_iv_2b ...`.

## Usage

### Python API

```python
from eegchtb.data import BCICompetitionIV2aLoader, EpochWindow
from eegchtb.evaluation import EpochBundleDataset, compute_metrics
from eegchtb.models import EEGNet
from eegchtb.training import TrainConfig, set_seed, train_model, predict

loader = BCICompetitionIV2aLoader(root="eegchtb/data", window=EpochWindow(0.0, 4.0), preprocessing="standard")
bundle = loader.load_session(subject=1, session="T")
dataset = EpochBundleDataset(bundle)

set_seed(0)
model = EEGNet(n_channels=bundle.n_channels, n_times=bundle.n_times, n_classes=4)
train_model(model, dataset, dataset, TrainConfig(epochs=50, seed=0))
print(compute_metrics(bundle.labels, predict(model, dataset)))
```

### One experiment

Same command on both platforms once the venv is activated:

```powershell
eegchtb-run --dataset bci_iv_2a --preprocessing standard --protocol subject_specific --subject 1 --output-root runs
```

### Baseline validation

Per model, writes `runs/<name>/`:

```powershell
.\env\Scripts\python.exe scripts\validate_eegnet_iv2a.py runs\eegnet-iv2a
.\env\Scripts\python.exe scripts\validate_conformer_iv2a.py runs\conformer-iv2a
.\env\Scripts\python.exe scripts\validate_grf_net_iv2a.py runs\grf_net-iv2a --channels all --seeds 0 1 2 3 4
```

```bash
./env/bin/python scripts/validate_eegnet_iv2a.py runs/eegnet-iv2a
./env/bin/python scripts/validate_conformer_iv2a.py runs/conformer-iv2a
./env/bin/python scripts/validate_grf_net_iv2a.py runs/grf_net-iv2a --channels all --seeds 0 1 2 3 4
```

### Benchmark grid

Dry run enumerates/validates; `--execute` trains, resumable:

```powershell
.\env\Scripts\python.exe scripts\run_grid.py configs\grid_iv2a.yaml runs\grid-iv2a
.\env\Scripts\python.exe scripts\run_grid.py configs\grid_iv2a.yaml runs\grid-iv2a --execute
```

```bash
./env/bin/python scripts/run_grid.py configs/grid_iv2a.yaml runs/grid-iv2a
./env/bin/python scripts/run_grid.py configs/grid_iv2a.yaml runs/grid-iv2a --execute
```

### Cross-subject and Galea transfer

Using trained grid snapshots:

```powershell
.\env\Scripts\python.exe scripts\eval_cross_subject.py --grid-root runs\grid-iv2a --out runs\cross_subject_2a
.\env\Scripts\python.exe scripts\eval_galea.py --grid-root runs\grid-iv2a --dataset bci_iv_2a --preprocessing minimal --model eegnet --seeds 0 1 2 3 4
```

```bash
./env/bin/python scripts/eval_cross_subject.py --grid-root runs/grid-iv2a --out runs/cross_subject_2a
./env/bin/python scripts/eval_galea.py --grid-root runs/grid-iv2a --dataset bci_iv_2a --preprocessing minimal --model eegnet --seeds 0 1 2 3 4
```

Every run writes an atomic, self-contained directory under `runs/` (manifest,
result, history, `.run_complete`); a failed or interrupted run leaves nothing
behind. Run outputs are committed to the repository.

## Testing

```powershell
.\env\Scripts\python.exe -m pytest
```

```bash
./env/bin/python -m pytest
```

215 tests: hermetic unit/logic tests plus dependency-gated ones (`requires_mne`,
`requires_torch`, dataset-directory fixtures) that skip cleanly when MNE, PyTorch
or the BCI IV datasets are absent.

## Results

Each model is first validated against its own reference protocol (not a fair
head-to-head - that comparison is the grid, below). 5 seeds unless noted;
figures are mean (std) over 9 subjects. Full per-run evidence is under
`runs/`.

### EEGNet baseline

Subject-specific (train T, test E), `standard` preprocessing, 750 epochs,
final-epoch model.

| Montage          | Accuracy      | Cohen kappa   |
| ----------------- | ------------- | ------------- |
| 22-channel         | 0.709 (0.009) | 0.612 (0.012) |
| Galea 4-channel    | 0.588 (0.007) | 0.451 (0.009) |

### EEG-Conformer baseline

Same split, the published Conformer recipe (Adam 2e-4, 2000 epochs, S&R
augmentation), best-test-epoch selection (reference protocol, optimistic).

| Montage          | Accuracy      | Cohen kappa   |
| ----------------- | ------------- | ------------- |
| 22-channel         | 0.806 (0.003) | 0.742 (0.004) |
| Galea 4-channel    | 0.518 (0.008) | 0.358 (0.011) |

### GRF-Net baseline

MiniRocket's protocol: both sessions pooled, trial-level 5-fold CV,
Euclidean Alignment.

| Montage                    | Accuracy      | Cohen kappa   |
| --------------------------- | ------------- | ------------- |
| 22-channel (5 seeds)         | 0.781 (0.002) | 0.708 (0.003) |
| Galea 4-channel (2 seeds)    | 0.652 (0.001) | 0.537 (0.001) |

### Benchmark grid and statistics

The full model x preprocessing x montage x seed grid (`runs/grid-*`),
analysed in `notebooks/statistical_testing.ipynb` (outputs in
`runs/statistics/`):

- **Montage effect dominates**: dropping from 22/3-channel to the Galea
  montage costs 5-15 accuracy points in almost every model/preprocessing
  cell, and is statistically significant (Wilcoxon, Holm-corrected) in most
  of them - the project's central "shortcut learning" question.
- **No consistent winning architecture**: a Friedman test across the three
  models per dataset/montage/preprocessing cell is significant in only
  4 of 12 cells (e.g. IV-2b galea `standard`, `minimal`); elsewhere the
  models are statistically indistinguishable.
- **Preprocessing effect is model- and montage-dependent**: significant for
  GRF-Net and EEGNet on IV-2b galea and IV-2a's 22-channel montage, not
  significant for EEGNet on IV-2a galea.
- All models beat chance (0.25 for 4-class IV-2a, 0.5 for 2-class IV-2b) in
  every tested cell.

## References

- BCI Competition IV, 2008. Data set 2a/2b: motor imagery (Graz). https://www.bbci.de/competition/iv/
- BrainFlow developers, 2025. BrainFlow documentation. https://brainflow.readthedocs.io/
- Dempster, A., Schmidt, D.F. and Webb, G.I., 2021. MiniRocket. KDD 2021, pp.248-257. DOI: 10.1145/3447548.3467231.
- Gramfort, A. et al., 2013. MNE-Python. Frontiers in Neuroscience, 7, 267. DOI: 10.3389/fnins.2013.00267.
- Harris, C.R. et al., 2020. NumPy. Nature, 585, pp.357-362. DOI: 10.1038/s41586-020-2649-2.
- He, H. and Wu, D., 2020. Euclidean Alignment. IEEE TBME, 67(2), pp.399-410. DOI: 10.1109/TBME.2019.2913914.
- Hjorth, B., 1975. Orthogonal source derivations. EEG Clin. Neurophysiology, 39(5), pp.526-530. DOI: 10.1016/0013-4694(75)90056-5.
- Lawhern, V.J. et al., 2018. EEGNet. J. Neural Eng., 15(5), 056013. DOI: 10.1088/1741-2552/aace8c. https://github.com/vlawhern/arl-eegmodels
- Paszke, A. et al., 2019. PyTorch. NeurIPS 32. https://pytorch.org
- Pedregosa, F. et al., 2011. Scikit-learn. JMLR, 12, pp.2825-2830. https://scikit-learn.org
- Perrin, F. et al., 1989. Spherical splines for scalp potential mapping. EEG Clin. Neurophysiology, 72(2), pp.184-187. DOI: 10.1016/0013-4694(89)90180-6.
- Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG Conformer. IEEE TNSRE, 31, pp.710-719. DOI: 10.1109/TNSRE.2022.3230250. https://github.com/eeyhsong/EEG-Conformer
