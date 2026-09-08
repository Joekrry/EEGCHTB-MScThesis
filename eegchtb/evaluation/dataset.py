"""PyTorch adapter exposing an EpochBundle as an indexable tensor dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.epochs import EpochBundle


class EpochBundleDataset(Dataset):
    """Indexable torch Dataset over an EpochBundle; __getitem__ yields (signal float32 (n_channels, n_times), label long). Tensors are copied at construction; subjects/trial_ids kept per trial for leakage-safe splits."""

    def __init__(self, bundle: EpochBundle) -> None:
        self.bundle = bundle
        self.class_names = dict(bundle.class_names)

        self.x = torch.tensor(bundle.data, dtype=torch.float32)
        self.y = torch.tensor(bundle.labels, dtype=torch.long)
        self.subjects = torch.full((len(self.y),), int(bundle.subject), dtype=torch.long)
        if bundle.trial_ids is None:
            trial_ids = np.arange(len(self.y), dtype=np.int64)
        else:
            trial_ids = np.asarray(bundle.trial_ids, dtype=np.int64)
        self.trial_ids = torch.tensor(trial_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole bundle batched: (n_trials, n_channels, n_times) and (n_trials,)."""
        return self.x, self.y
