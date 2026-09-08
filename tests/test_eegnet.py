"""Tests for the EEGNet PyTorch implementation."""

from __future__ import annotations

import pytest

from conftest import requires_torch


@requires_torch()
@pytest.mark.parametrize("n_channels", [3, 22])  # IV-2b and IV-2a
@pytest.mark.parametrize("batch", [1, 16])
def test_forward_output_shape(n_channels, batch):
    import torch

    from eegchtb.models import EEGNet

    model = EEGNet(n_channels=n_channels, n_times=1000, n_classes=4)
    out = model(torch.zeros(batch, n_channels, 1000))
    assert out.shape == (batch, 4)


@requires_torch()
def test_backward_produces_gradients():
    import torch

    from eegchtb.models import EEGNet

    model = EEGNet(n_channels=3, n_times=1000, n_classes=2)
    x = torch.randn(8, 3, 1000)
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)


@requires_torch()
def test_rejects_invalid_input():
    import torch

    from eegchtb.models import EEGNet

    model = EEGNet(n_channels=3, n_times=1000, n_classes=2)
    with pytest.raises(ValueError, match="batch, n_channels, n_times"):
        model(torch.zeros(3, 1000))  # missing batch dim
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(5, 4, 1000))  # wrong channel count


@requires_torch()
def test_parameter_count_matches_reference():
    from eegchtb.models import EEGNet

    # C=3, T=1000, 2 classes, defaults F1=8 D=2 F2=16 kernel_length=64
    model = EEGNet(n_channels=3, n_times=1000, n_classes=2)
    assert model.count_parameters() == 2146
