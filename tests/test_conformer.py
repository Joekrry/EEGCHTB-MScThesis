"""Tests for the EEG-Conformer PyTorch implementation."""

from __future__ import annotations

import pytest

from conftest import requires_torch


@requires_torch()
@pytest.mark.parametrize("n_channels", [3, 22])  # IV-2b and IV-2a
@pytest.mark.parametrize("batch", [1, 16])
def test_forward_output_shape(n_channels, batch):
    import torch

    from eegchtb.models import EEGConformer

    model = EEGConformer(n_channels=n_channels, n_times=1000, n_classes=4)
    out = model(torch.zeros(batch, n_channels, 1000))
    assert out.shape == (batch, 4)


@requires_torch()
def test_backward_produces_gradients():
    import torch

    from eegchtb.models import EEGConformer

    model = EEGConformer(n_channels=3, n_times=1000, n_classes=2)
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

    from eegchtb.models import EEGConformer

    model = EEGConformer(n_channels=3, n_times=1000, n_classes=2)
    with pytest.raises(ValueError, match="batch, n_channels, n_times"):
        model(torch.zeros(3, 1000))  # missing batch dim
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(5, 4, 1000))  # wrong channel count


@requires_torch()
def test_emb_size_must_divide_heads():
    from eegchtb.models import EEGConformer

    with pytest.raises(ValueError, match="divisible"):
        EEGConformer(n_channels=3, n_times=1000, n_classes=2, emb_size=40, num_heads=7)


@requires_torch()
def test_parameter_count_matches_reference():
    from eegchtb.models import EEGConformer

    # Reference BCI IV-2a config: 22 channels, 4 s @ 250 Hz = 1000 samples, 4
    # classes, published defaults. The patch stem yields 61 tokens of size 40,
    # so the head's first linear is Linear(2440, 256), matching the official
    # implementation's hardcoded 2440.
    model = EEGConformer(n_channels=22, n_times=1000, n_classes=4)
    assert model.count_parameters() == 789572


@requires_torch()
def test_attention_mask_excludes_masked_keys():
    """A masked key position must not influence the output of any other query."""
    import torch

    from eegchtb.models.conformer import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(emb_size=8, num_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 5, 8)

    # Keep every key except the last one.
    mask = torch.tensor([[True, True, True, True, False]])
    with torch.no_grad():
        out = attn(x, mask)
        # Drastically change the masked position's features; as a masked-out key
        # it must not affect the other queries' outputs. (Its own query row does
        # change, since perturbing x[:, 4] also alters that query vector, so the
        # comparison is restricted to the unmasked query positions 0..3.)
        x_perturbed = x.clone()
        x_perturbed[:, 4, :] += 100.0
        out_perturbed = attn(x_perturbed, mask)

    assert torch.allclose(out[:, :4], out_perturbed[:, :4], atol=1e-5)


@requires_torch()
def test_attention_full_mask_matches_no_mask():
    """An all-true mask must be a no-op relative to passing no mask."""
    import torch

    from eegchtb.models.conformer import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(emb_size=8, num_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(attn(x, None), attn(x, mask), atol=1e-6)
