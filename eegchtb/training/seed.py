"""Central seed control for reproducible training."""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) and put cuDNN in deterministic mode."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
