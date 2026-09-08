"""EEG-Conformer (Song et al. 2022) ported to PyTorch.

Reference:
  Y. Song, Q. Zheng, B. Liu and X. Gao, "EEG Conformer: Convolutional
  Transformer for EEG Decoding and Visualization", IEEE Trans. Neural Syst.
  Rehabil. Eng., vol. 31, pp. 710-719, 2022.
  Official PyTorch implementation: https://github.com/eeyhsong/EEG-Conformer
  Cross-checked against braindecode's EEGConformer.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Shallow temporal/spatial conv stem turning EEG (batch, 1, n_channels, n_times) into tokens (batch, n_tokens, emb_size)."""

    def __init__(
        self,
        n_channels: int,
        *,
        emb_size: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        # Temporal conv (learned filter bank), then a spatial conv over all
        # electrodes that collapses the channel axis into one virtual channel.
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, filter_time_length), (1, 1)),
            nn.Conv2d(emb_size, emb_size, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            # average pooling acts as slicing to obtain 'patches' along time
            nn.AvgPool2d((1, pool_time_length), (1, pool_time_stride)),
            nn.Dropout(drop_prob),
        )
        # 1x1 conv projection (the reference notes it slightly aids fitting).
        self.projection = nn.Conv2d(emb_size, emb_size, (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        # (b, emb, h, w) -> (b, h*w, emb); h == 1 after the spatial conv.
        b, e, h, w = x.shape
        return x.reshape(b, e, h * w).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention; scales by emb_size ** 0.5 (full embedding), matching the reference rather than the canonical head_dim ** 0.5."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})"
            )
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        head_dim = self.emb_size // self.num_heads
        # (b, n, emb) -> (b, heads, n, head_dim)
        return x.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        queries = self._split_heads(self.queries(x))
        keys = self._split_heads(self.keys(x))
        values = self._split_heads(self.values(x))

        energy = torch.matmul(queries, keys.transpose(-2, -1))  # (b, heads, n, n)
        if mask is not None:
            # mask is (b, n) boolean over key positions; True keeps the key.
            keep = mask[:, None, None, :]
            energy = energy.masked_fill(~keep, torch.finfo(energy.dtype).min)
        att = torch.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)

        out = torch.matmul(att, values)  # (b, heads, n, head_dim)
        b, _, n, _ = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.emb_size)
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    """Position-wise MLP with GELU and an expansion factor, as in ViT."""

    def __init__(self, emb_size: int, expansion: int, drop_prob: float) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    """Pre-norm attention and feed-forward sub-blocks, each with a residual add."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_prob: float,
        *,
        forward_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, drop_prob)
        self.attn_drop = nn.Dropout(drop_prob)
        self.ff_norm = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlock(emb_size, forward_expansion, drop_prob)
        self.ff_drop = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.attn_norm(x), mask))
        x = x + self.ff_drop(self.ff(self.ff_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of identical transformer-encoder blocks."""

    def __init__(
        self, depth: int, emb_size: int, num_heads: int, drop_prob: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(emb_size, num_heads, drop_prob)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class ClassificationHead(nn.Module):
    """Flatten the token sequence and read out class logits through a small MLP."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        *,
        hidden1: int = 256,
        hidden2: int = 32,
        drop1: float = 0.5,
        drop2: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ELU(),
            nn.Dropout(drop1),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
            nn.Dropout(drop2),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))


class EEGConformer(nn.Module):
    """EEG-Conformer for motor imagery; input (batch, n_channels, n_times), output logits (batch, n_classes)."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        *,
        emb_size: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        depth: int = 6,
        num_heads: int = 10,
        drop_prob: float = 0.5,
        att_drop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        self.patch_embedding = PatchEmbedding(
            n_channels,
            emb_size=emb_size,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = TransformerEncoder(depth, emb_size, num_heads, att_drop_prob)

        # size the head from a dummy pass in eval mode, so the shape probe never
        # updates BatchNorm running statistics (as EEGNet does here too)
        self.eval()
        with torch.no_grad():
            tokens = self.patch_embedding(torch.zeros(1, 1, n_channels, n_times))
        self.train()
        if tokens.shape[1] < 1:
            raise ValueError(
                f"n_times={n_times} is too short for the conv/pool stem; it yields no tokens"
            )
        self.classify = ClassificationHead(tokens.shape[1] * emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.classify(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
