"""EEGNet (Lawhern et al. 2018) ported to PyTorch.

Reference:
  V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for
  EEG-based brain-computer interfaces", J. Neural Eng. 15(5), 2018.
  Official Keras/TensorFlow implementation: https://github.com/vlawhern/arl-eegmodels
  (Used for: layer order and the two-block temporal/depthwise/separable design,
   the F1/D/F2 channel scheme, kernel and pooling sizes, ELU activation, and the
   max-norm constraints on the depthwise and classifier weights.)

Deviations from the Keras reference, kept explicit:
  - BatchNorm uses momentum=0.01, eps=1e-3 to match Keras' defaults
    (Keras momentum 0.99 == PyTorch 0.01).
  - Keras applies the max-norm kernel constraints continuously; here they are a
    method (apply_max_norm_constraints) the trainer calls after each optimiser
    step, implemented with torch.renorm_ (PyTorch's max-norm constraint).
  - forward emits logits (no softmax); pair with nn.CrossEntropyLoss.
"""

from __future__ import annotations

import torch
from torch import nn


def _max_norm_(weight: torch.Tensor, max_value: float) -> None:
    """Constrain each output unit's weight vector in place to L2 norm <= max_value via torch.renorm_ (PyTorch's built-in max-norm constraint), matching Keras' max_norm kernel constraint."""
    with torch.no_grad():
        weight.renorm_(2, 0, max_value)


class EEGNet(nn.Module):
    """EEGNet-F1,D for motor imagery; input (batch, n_channels, n_times), output logits (batch, n_classes)."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        *,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        F2 = F2 if F2 is not None else F1 * D
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        sep_kernel = 16

        # Block 1: temporal conv, then a depthwise spatial conv over all channels.
        # Explicit "same" padding (extra on the right, per the TF/Keras convention)
        self.pad_temporal = nn.ZeroPad2d(((kernel_length - 1) // 2, kernel_length // 2, 0, 0))
        self.temporal = nn.Conv2d(1, F1, (1, kernel_length), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise then pointwise)
        self.pad_sep = nn.ZeroPad2d(((sep_kernel - 1) // 2, sep_kernel // 2, 0, 0))
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        self.elu = nn.ELU()

        # size the classifier from a dummy pass in eval mode, so the shape probe
        # never updates BatchNorm running statistics
        self.eval()
        with torch.no_grad():
            n_features = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.train()
        self.classify = nn.Linear(n_features, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.temporal(self.pad_temporal(x)))
        x = self.drop1(self.pool1(self.elu(self.bn2(self.depthwise(x)))))
        x = self.sep_point(self.sep_depth(self.pad_sep(x)))
        x = self.drop2(self.pool2(self.elu(self.bn3(x))))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels; got {x.shape[1]}"
            )
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)
        return self.classify(self._features(x))

    def apply_max_norm_constraints(self) -> None:
        """Clamp the depthwise (max_norm=1) and classifier (max_norm=0.25) weights; call after each optimiser step."""
        _max_norm_(self.depthwise.weight, 1.0)
        _max_norm_(self.classify.weight, 0.25)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
