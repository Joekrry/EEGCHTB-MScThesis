from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch

MONTAGES = [3, 4, 10, 22]
RATES = [(125.0, 500), (250.0, 1000)]


def _coords(n_channels, seed=0):
    return np.random.default_rng(seed).standard_normal((n_channels, 2))


@requires_torch()
@pytest.mark.parametrize("n_channels", MONTAGES)
@pytest.mark.parametrize("sfreq,n_times", RATES)
@pytest.mark.parametrize("batch", [1, 16])
def test_forward_shape(n_channels, sfreq, n_times, batch):
    import torch

    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=n_channels, n_times=n_times, n_classes=4, sfreq=sfreq)
    assert model(torch.zeros(batch, n_channels, n_times)).shape == (batch, 4)


@requires_torch()
def test_backward_produces_gradients():
    import torch

    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0)
    x = torch.randn(8, 22, 1000)
    loss = torch.nn.functional.cross_entropy(model(x), torch.randint(0, 4, (8,)))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)


@requires_torch()
def test_rejects_invalid_input():
    import torch

    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=3, n_times=1000, n_classes=2)
    with pytest.raises(ValueError, match="batch, n_channels, n_times"):
        model(torch.zeros(3, 1000))
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(5, 4, 1000))


@requires_torch()
def test_parameter_count_anchor():
    from eegchtb.models import GRFNet

    # counts include the learnable per-kernel PPV bias (n_features params): 192 for
    # the default bank (3655 + 192), 128 for the fixed-only bank (2183 + 128).
    assert GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0).count_parameters() == 3847
    assert GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0, learnable_per_group=0).count_parameters() == 2311
    # PPV bias can be disabled to recover the original head
    assert GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0, ppv_bias=False).count_parameters() == 3655


@requires_torch()
def test_fixed_bank_is_frozen():
    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0)
    assert all(not conv.weight.requires_grad for conv in model.bank.fixed)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert not any(name.startswith("bank.fixed.") for name in trainable)
    assert any(name.startswith("bank.learnable.") for name in trainable)


@requires_torch()
def test_statistics_control_head_width():
    from eegchtb.models import GRFNet

    full = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0)
    ppv_only = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0, statistics=("ppv",))
    assert full.classify.weight.shape[1] == 3 * full.n_features
    assert ppv_only.classify.weight.shape[1] == full.n_features


@requires_torch()
def test_rate_invariant_feature_count():
    from eegchtb.models import GRFNet

    a = GRFNet(n_channels=10, n_times=500, n_classes=4, sfreq=125.0).n_features
    b = GRFNet(n_channels=10, n_times=1000, n_classes=4, sfreq=250.0).n_features
    assert a == b


@requires_torch()
def test_deterministic_init():
    import torch

    from eegchtb.models import GRFNet
    from eegchtb.training import set_seed

    set_seed(0)
    a = GRFNet(n_channels=10, n_times=500, n_classes=4, sfreq=125.0)
    set_seed(0)
    b = GRFNet(n_channels=10, n_times=500, n_classes=4, sfreq=125.0)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


@requires_torch()
def test_rejects_unknown_spatial_init():
    from eegchtb.models import GRFNet

    with pytest.raises(ValueError, match="spatial_init"):
        GRFNet(n_channels=4, n_times=1000, n_classes=2, spatial_init="laplacian")


@requires_torch()
def test_geometry_requires_and_validates_coords():
    from eegchtb.models import GRFNet

    with pytest.raises(ValueError, match="requires channel_names or channel_coords"):
        GRFNet(n_channels=4, n_times=1000, n_classes=2, spatial_init="geometry")
    with pytest.raises(ValueError, match=r"channel_coords must be \(4, 2\)"):
        GRFNet(n_channels=4, n_times=1000, n_classes=2, spatial_init="geometry", channel_coords=_coords(5))
    # names are resolved against the montage template, so a wrong count or an
    # electrode the template does not carry must fail loudly rather than mis-seed
    with pytest.raises(ValueError, match="channel_names must have 4 entries"):
        GRFNet(n_channels=4, n_times=1000, n_classes=2, channel_names=["C3", "Cz", "C4"])
    with pytest.raises(ValueError, match="absent from the standard_1020 template"):
        GRFNet(n_channels=2, n_times=1000, n_classes=2, channel_names=["C3", "NotAnElectrode"])


@requires_torch()
def test_channel_names_select_geometry_and_match_explicit_coords():
    from eegchtb.data.channels import channel_positions
    from eegchtb.models import GRFNet

    names = ["C3", "Cz", "C4", "Pz"]
    # supplying a montage implies geometry seeding without asking for it
    by_name = GRFNet(n_channels=4, n_times=1000, n_classes=2, channel_names=names)
    assert by_name.spatial.spatial_init == "geometry"
    # and resolving names must give exactly what passing the coordinates gives
    by_coords = GRFNet(n_channels=4, n_times=1000, n_classes=2,
                       channel_coords=channel_positions(names))
    assert np.allclose(by_name.spatial.conv.weight.detach().numpy(),
                       by_coords.spatial.conv.weight.detach().numpy())
    # geometry is data-driven, so it does not vary with the torch RNG
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    again = GRFNet(n_channels=4, n_times=1000, n_classes=2, channel_names=names)
    assert np.allclose(by_name.spatial.conv.weight.detach().numpy(),
                       again.spatial.conv.weight.detach().numpy())


@requires_torch()
def test_defaults_to_random_when_no_montage_given():
    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=4, n_times=1000, n_classes=2)
    assert model.spatial.spatial_init == "random"


@requires_torch()
def test_degenerate_geometry_pattern_is_skipped_not_seeded_dead():
    from eegchtb.models import GRFNet

    # collinear electrodes flatten the x*y basis pattern; every seeded component
    # must still be a live unit-norm filter rather than an all-zero row
    coords = np.stack([np.linspace(-1.0, 1.0, 4), np.zeros(4)], axis=1)
    model = GRFNet(n_channels=4, n_times=1000, n_classes=2,
                   spatial_init="geometry", channel_coords=coords)
    rows = model.spatial.conv.weight.detach().numpy()[:, :, 0]
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)


@requires_torch()
def test_geometry_seed_matches_coordinate_basis():
    from eegchtb.models import GRFNet

    coords = _coords(4)
    model = GRFNet(n_channels=4, n_times=1000, n_classes=2, spatial_init="geometry", channel_coords=coords)
    weight = model.spatial.conv.weight.detach().numpy()

    def normed(v):
        v = v - v.mean()
        return v / np.linalg.norm(v)

    assert np.allclose(weight[0, :, 0], normed(coords[:, 0]), atol=1e-5)
    assert np.allclose(weight[1, :, 0], normed(coords[:, 1]), atol=1e-5)


@requires_torch()
def test_geometry_forward_and_backward():
    import torch

    from eegchtb.models import GRFNet

    model = GRFNet(n_channels=10, n_times=1000, n_classes=4, sfreq=250.0, spatial_init="geometry", channel_coords=_coords(10))
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 10, 1000)), torch.randint(0, 4, (8,)))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def _learnable_bundle(n=48, n_ch=3, n_t=128, sfreq=128.0, seed=0):
    from eegchtb.data import EpochBundle

    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    data = (rng.standard_normal((n, n_ch, n_t)) * 0.1).astype(np.float32)
    burst = np.sin(2 * np.pi * 10 * np.linspace(0, 1, n_t)).astype(np.float32)
    data[labels == 1] += burst
    return EpochBundle(
        data=data, labels=labels, dataset="synthetic", subject=1, session="T",
        sfreq=sfreq, channel_names=[f"c{i}" for i in range(n_ch)],
        class_names={0: "a", 1: "b"}, trial_ids=np.arange(n),
    )


def _train_once():
    from eegchtb.evaluation import EpochBundleDataset
    from eegchtb.models import GRFNet
    from eegchtb.training import TrainConfig, predict, set_seed, train_model

    ds = EpochBundleDataset(_learnable_bundle())
    set_seed(0)
    model = GRFNet(n_channels=3, n_times=128, n_classes=2, sfreq=128.0)
    result = train_model(model, ds, ds, TrainConfig(epochs=5, batch_size=8, patience=10, seed=0))
    return predict(model, ds), result


@requires_torch()
def test_trains_through_shared_trainer_deterministically():
    preds_a, res_a = _train_once()
    preds_b, res_b = _train_once()
    assert np.array_equal(preds_a, preds_b)
    assert res_a.history["train_loss"] == res_b.history["train_loss"]
    assert np.isfinite(res_a.history["train_loss"]).all()
