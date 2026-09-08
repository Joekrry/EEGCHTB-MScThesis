"""GRF-Net (Galea Rocket-Fusion Network)

Not a recreation of ROCKET/MiniRocket. It keeps their low-overfitting recipe but
adds three EEG/motor-imagery/Galea-aware elements:
        a geometry-aware learnable spatial front-end (ROCKET has no spatial filter)

        a multi-statistic fusion pooling combining PPV with log-variance band-power
        and mean (ROCKET pools PPV only)

        an end-to-end trainable compact fusion head with per-statistic gating
        (ROCKET uses a ridge classifier)

References

  A. Dempster, F. Petitjean, G. I. Webb, "ROCKET: exceptionally fast and accurate
    time series classification using random convolutional kernels", DMKD 34, 2020.
    (PPV pooling; random, mean-centred convolutional kernels.)

  A. Dempster, D. F. Schmidt, G. I. Webb, "MiniRocket", KDD 2021. (Small fixed
    dilated kernel bank as a low-variance feature extractor.)

  K. K. Ang et al., "Filter Bank Common Spatial Pattern (FBCSP)", IJCNN 2008.
    (log-variance of spatially filtered band power as the motor-imagery feature.)
    
  R. T. Schirrmeister et al., "Deep learning with convolutional neural networks
    for EEG decoding and visualization", HBM 38(11), 2017. (square/mean/log band
    power readout after a learned spatial filter, in ShallowConvNet.)

"""

from __future__ import annotations

import torch
from torch import nn

_STATISTICS = ("ppv", "bandpower", "mean")


def _rocket_init_(weight: torch.Tensor) -> None:
    """Seed a conv weight (out, 1, K) ROCKET-style: random kernels, each mean-centred over its taps (so a constant input yields no response)."""
    with torch.no_grad():
        weight.normal_()
        weight.sub_(weight.mean(dim=-1, keepdim=True))


def _channel_coords_from_names(channel_names, n_channels: int):
    """2-D montage coordinates for named electrodes, via the project's 10-20 lookup."""
    from ..data.channels import channel_positions

    names = list(channel_names)
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names must have {n_channels} entries to match n_channels; got {len(names)}"
        )
    return channel_positions(names)


def _geometry_seed_spatial_(weight: torch.Tensor, coords) -> None:
    """Seed a spatial-conv weight (spatial_components, n_channels, 1) with smooth spatial basis patterns (x, y gradients, then quadratics) built from electrode coordinates and cycled across the outputs."""
    n_components, n_channels, _ = weight.shape
    coords = torch.as_tensor(coords, dtype=weight.dtype)
    if coords.shape != (n_channels, 2):
        raise ValueError(f"channel_coords must be ({n_channels}, 2); got {tuple(coords.shape)}")
    x, y = coords[:, 0], coords[:, 1]
    family = [x, y, x * x + y * y, x * y, x * x - y * y]
    basis = []
    for feature in family:
        feature = feature - feature.mean()  # a contrast, so no pattern is common-mode
        norm = feature.norm()
        # a montage can make a pattern degenerate (collinear electrodes flatten x*y);
        # such a pattern would seed a dead all-zero component, so drop it
        if norm > 1e-6:
            basis.append(feature / norm)
    if not basis:
        return  # no usable geometry (e.g. a single channel); leave the random init
    with torch.no_grad():
        for out_channel in range(n_components):
            weight[out_channel, :, 0] = basis[out_channel % len(basis)]


class SpatialFrontEnd(nn.Module):
    """Learnable spatial mixer: a 1x1 conv over the montage giving spatial_components CSP-like components, then BatchNorm. Optionally geometry-seeded from electrode coordinates."""

    def __init__(
        self,
        n_channels: int,
        spatial_components: int,
        *,
        spatial_init: str | None = None,
        channel_names=None,
        channel_coords=None,
    ) -> None:
        super().__init__()
        # geometry is used when a montage is supplied and nothing else is asked for,
        # so callers that know their electrodes get it without a second argument,
        # while spatial_init='random' stays an explicit opt-out
        if spatial_init is None:
            has_montage = channel_names is not None or channel_coords is not None
            spatial_init = "geometry" if has_montage else "random"
        if spatial_init not in ("random", "geometry"):
            raise ValueError(f"spatial_init must be 'random' or 'geometry'; got {spatial_init!r}")
        self.conv = nn.Conv1d(n_channels, spatial_components, 1, bias=False)
        self.bn = nn.BatchNorm1d(spatial_components, momentum=0.01, eps=1e-3)
        self.spatial_init = spatial_init
        if spatial_init == "geometry":
            if channel_coords is None:
                if channel_names is None:
                    raise ValueError(
                        "spatial_init='geometry' requires channel_names or channel_coords"
                    )
                channel_coords = _channel_coords_from_names(channel_names, n_channels)
            _geometry_seed_spatial_(self.conv.weight, channel_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))  # (batch, spatial_components, n_times)


class RocketFeatureBank(nn.Module):
    """Multi-dilation depthwise conv bank over the spatial components: a frozen ROCKET-style kernel set (low variance) plus an optional few learnable band-adaptive kernels. Dilations scale with the sampling rate so the receptive fields cover a fixed span in seconds."""

    def __init__(
        self,
        spatial_components: int,
        *,
        sfreq: float,
        kernel_length: int,
        base_dilations,
        fixed_per_group: int,
        learnable_per_group: int,
    ) -> None:
        super().__init__()
        if kernel_length % 2 == 0:
            raise ValueError(f"kernel_length must be odd for symmetric padding; got {kernel_length}")
        if fixed_per_group < 1:
            raise ValueError(f"fixed_per_group must be >= 1; got {fixed_per_group}")
        if learnable_per_group < 0:
            raise ValueError(f"learnable_per_group must be >= 0; got {learnable_per_group}")
        # Rate-adaptive dilations: a fixed base set at 250 Hz, rescaled by sfreq so a
        # 4 s window keeps the same dilation count (hence the same feature count) at
        # 125 and 250 Hz. Duplicates are kept, not deduplicated, to hold the count fixed.
        self.dilations = [max(1, round(b * sfreq / 250.0)) for b in base_dilations]
        s = spatial_components

        def _conv(out_per_group: int, dilation: int) -> nn.Conv1d:
            # groups=s makes it depthwise: each spatial component gets its own kernels.
            return nn.Conv1d(
                s,
                s * out_per_group,
                kernel_length,
                padding=dilation * (kernel_length - 1) // 2,  # "same" length for odd kernels
                dilation=dilation,
                groups=s,
                bias=False,
            )

        self.fixed = nn.ModuleList()
        for dilation in self.dilations:
            conv = _conv(fixed_per_group, dilation)
            _rocket_init_(conv.weight)
            conv.weight.requires_grad_(False)  # frozen ROCKET bank: the low-overfitting core
            self.fixed.append(conv)

        self.learnable = (
            nn.ModuleList(_conv(learnable_per_group, d) for d in self.dilations)
            if learnable_per_group > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = []
        for i, fixed in enumerate(self.fixed):
            maps.append(fixed(x))
            if self.learnable is not None:
                maps.append(self.learnable[i](x))
        return torch.cat(maps, dim=1)  # (batch, n_features, n_times)


class GRFNet(nn.Module):
    """GRF-Net for motor imagery: geometry-aware learnable spatial front-end -> ROCKET-style multi-dilation conv bank -> PPV/band-power/mean fusion pooling -> gated compact linear head. statistics selects the fusion streams; learnable_per_group=0 is the pure-fixed-bank ablation."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        *,
        sfreq: float = 250.0,
        spatial_components: int = 16,
        spatial_init: str | None = None,
        channel_names=None,
        channel_coords=None,
        kernel_length: int = 9,
        base_dilations=(1, 2, 4, 8),
        fixed_per_group: int = 2,
        learnable_per_group: int = 1,
        statistics=_STATISTICS,
        ppv_tau: float = 0.5,
        ppv_bias: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        statistics = tuple(statistics)
        unknown = [s for s in statistics if s not in _STATISTICS]
        if unknown:
            raise ValueError(f"statistics must be drawn from {_STATISTICS}; got {unknown}")
        if len(set(statistics)) != len(statistics):
            raise ValueError(f"statistics must not repeat; got {statistics}")
        if not statistics:
            raise ValueError("statistics must name at least one fusion stream")
        if ppv_tau <= 0:
            raise ValueError(f"ppv_tau must be positive; got {ppv_tau}")

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.statistics = statistics
        self.ppv_tau = ppv_tau

        self.spatial = SpatialFrontEnd(
            n_channels,
            spatial_components,
            spatial_init=spatial_init,
            channel_names=channel_names,
            channel_coords=channel_coords,
        )
        self.bank = RocketFeatureBank(
            spatial_components,
            sfreq=sfreq,
            kernel_length=kernel_length,
            base_dilations=base_dilations,
            fixed_per_group=fixed_per_group,
            learnable_per_group=learnable_per_group,
        )

        n_features = len(self.bank.dilations) * spatial_components * (fixed_per_group + learnable_per_group)
        self.n_features = n_features
        pooled_dim = len(statistics) * n_features
        self.feature_norm = nn.BatchNorm1d(n_features, momentum=0.01, eps=1e-3)
        # Learnable per-kernel PPV threshold. ROCKET/MiniRocket make PPV informative
        # by counting values above a per-kernel bias sampled from the convolution
        # output; without it, the mean-centred kernels leave PPV ~= 0.5 for every
        # trial once features are batch-normalised. A learnable bias is the
        # end-to-end trainable equivalent of that sampled threshold.
        self.ppv_bias = (
            nn.Parameter(torch.zeros(n_features)) if (ppv_bias and "ppv" in statistics) else None
        )
        self.stream_gate = nn.Parameter(torch.zeros(len(statistics)))
        self.drop = nn.Dropout(dropout)
        self.classify = nn.Linear(pooled_dim, n_classes)

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        """Fuse complementary statistics over the time axis into (batch, len(statistics) * n_features)."""
        streams = []
        for name in self.statistics:
            if name == "ppv":
                z = feats if self.ppv_bias is None else feats - self.ppv_bias[None, :, None]
                streams.append(torch.sigmoid(z / self.ppv_tau).mean(dim=-1))
            elif name == "bandpower":
                # log-variance over time: the FBCSP/ShallowConvNet mu/beta ERD-ERS feature.
                streams.append(torch.log(feats.var(dim=-1, unbiased=False) + 1e-6))
            else:
                streams.append(feats.mean(dim=-1))
        return torch.cat(streams, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        components = self.spatial(x)  # (batch, spatial_components, n_times)
        feats = self.feature_norm(self.bank(components))  # (batch, n_features, n_times)
        pooled = self._pool(feats)
        # Soft per-stream gating over the pooled statistic blocks.
        gate = torch.softmax(self.stream_gate, dim=0)
        blocks = [gate[i] * pooled[:, i * self.n_features : (i + 1) * self.n_features] for i in range(len(self.statistics))]
        return self.classify(self.drop(torch.cat(blocks, dim=1)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
