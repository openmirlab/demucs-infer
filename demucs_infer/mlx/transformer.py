"""Vendored+adapted MLX `CrossTransformerEncoder` -- htdemucs.py's bottleneck.

Ports `transformer.py`'s `CrossTransformerEncoder` (the transformer bottleneck
`HTDemucs` uses in place of `HDemucs`'s LSTM/local-attention branch) for the
standard configuration every shipped htdemucs checkpoint in this package's
registry actually uses: dense (non-sparse) attention, `emb="sin"`,
`norm_first=True`, plain `LayerNorm` for `norm1`/`norm2`/`norm3`
(`t_group_norm=False`). `HTDemucsMLX` (`htdemucs.py`) refuses to construct
with any other combination rather than silently building a module tree that
doesn't match the checkpoint -- see its module docstring.

**`t_norm_out` (`_NormOutTC`) is ported, not skipped** -- it was initially
assumed inert and refused, until loading the real `htdemucs` checkpoint's
captured `_init_args_kwargs` showed `t_norm_out=True` (the shipped default),
and `load_converted_weights` then caught the resulting silent mismatch
(392 unconverted parameters at `crosstransformer.layers*.norm_out.*` and a
layer-parity divergence it exposed alongside it) before this port ever ran
against real weights. See `_NormOutTC`'s docstring for what it actually
computes -- not `nn.LayerNorm`.

**Not ported**: sparse attention (`t_sparse_self_attn`/`t_sparse_cross_attn`,
which need `xformers`), `t_emb="cape"`/`"scaled"` positional embeddings, and
`t_group_norm=True` (the `norm1`/`norm2`/`norm3` GroupNorm variant -- distinct
from `t_norm_out`, which *is* ported). None of this package's registry
checkpoints use them (`sin` embedding, dense attention) -- confirmed by
reading `_init_args_kwargs` off the real `htdemucs` checkpoint this port's
parity test loads, and by the `msst_htdemucs_vocals` registry entry's explicit
architecture kwargs in `config/checkpoints.toml`, which also carry
`t_group_norm = false`.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/demucs_mlx/mlx_transformer.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.
    Adapted: sparse-attention and `t_group_norm=True` branches removed (this
    package's HTDemucsMLX refuses those configurations before construction
    rather than building an untested path); `norm_out` reimplemented directly
    against `htdemucs.py`'s own `MyGroupNorm` semantics rather than carried
    over from upstream (upstream's `mlx_transformer.py` wraps a composed
    `nn.GroupNorm` as `self.gn`, needing a `convert.py` key remap this port
    doesn't need -- see `_NormOutTC`); positional-embedding cache dropped
    (this package's chunked seam calls the model once per chunk with a stable
    shape, so the cache upstream added for repeated-shape workloads buys
    nothing here and is one fewer piece of hidden state to keep correct).

Reads: mlx.core, mlx.nn, .layers (LayerScale)
"""

from __future__ import annotations

import math
import random

import mlx.core as mx
from mlx import nn

from .layers import LayerScale


def create_sin_embedding(length: int, dim: int, shift: int = 0, max_period: float = 10000.0) -> mx.array:
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pos = shift + mx.arange(length).reshape(-1, 1, 1)
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


def create_2d_sin_embedding(d_model: int, height: int, width: int, max_period: float = 10000.0) -> mx.array:
    """Matches `transformer.create_2d_sin_embedding`'s interleaved sin/cos
    layout: width-embedding occupies channels `[0:half]` (sin at even
    offsets, cos at odd), height-embedding occupies `[half:d_model]` the same
    way. Built via stack+reshape (alternating sin/cos along a fresh axis,
    then flattened) rather than strided assignment, which is the
    interleaving `stack([sin, cos], axis=1).reshape(-1, ...)` always
    produces -- e.g. `stack([[a0,a1],[b0,b1]], axis=1).reshape(-1) ==
    [a0,b0,a1,b1]` for two channels, matching torch's `pe[0::2]=sin,
    pe[1::2]=cos`."""
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4")
    half = d_model // 2
    div_term = mx.exp(mx.arange(0.0, half, 2) * -(math.log(max_period) / half))

    pos_w = mx.arange(0.0, width).reshape(-1, 1)
    pos_h = mx.arange(0.0, height).reshape(-1, 1)
    n = half // 2

    sin_w = mx.broadcast_to(mx.sin(pos_w * div_term).transpose(1, 0)[:, None, :], (n, height, width))
    cos_w = mx.broadcast_to(mx.cos(pos_w * div_term).transpose(1, 0)[:, None, :], (n, height, width))
    pe_w = mx.stack([sin_w, cos_w], axis=1).reshape(half, height, width)

    sin_h = mx.broadcast_to(mx.sin(pos_h * div_term).transpose(1, 0)[:, :, None], (n, height, width))
    cos_h = mx.broadcast_to(mx.cos(pos_h * div_term).transpose(1, 0)[:, :, None], (n, height, width))
    pe_h = mx.stack([sin_h, cos_h], axis=1).reshape(d_model - half, height, width)

    pe = mx.concatenate([pe_w, pe_h], axis=0)
    return pe[None, :]


class _NormOutTC(nn.Module):
    """`transformer.MyGroupNorm(num_groups=int(norm_out), d_model)` applied
    within a layer's `norm_out` slot, for a `(B, T, C)` tensor.

    Every real checkpoint's `t_norm_out` is the bool `True`, and Torch's
    `int(True) == 1` -- a *single* group spanning every channel -- so this
    normalizes jointly over `(T, C)` per batch element (not per-channel, and
    not per-timestep the way `nn.LayerNorm(d_model)` would), then applies a
    per-channel affine. `MyGroupNorm` subclasses `nn.GroupNorm` directly
    (no wrapper submodule), so its Torch state dict already has flat
    `norm_out.weight`/`norm_out.bias` keys -- this module mirrors that
    naming exactly, no `convert.py` remap needed.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_channels,), dtype=mx.float32)
        self.bias = mx.zeros((num_channels,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return x * self.weight + self.bias


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 norm_first, layer_scale=False, init_values=1e-4, norm_out=False):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = activation
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.norm_out = _NormOutTC(d_model) if (norm_first and norm_out) else None

    def _g1(self, x):
        return self.gamma_1(x) if self.gamma_1 is not None else x

    def _g2(self, x):
        return self.gamma_2(x) if self.gamma_2 is not None else x

    def __call__(self, x: mx.array) -> mx.array:
        if self.norm_first:
            xn = self.norm1(x)
            x = x + self._g1(self.attn(xn, xn, xn))
            x = x + self._g2(self.linear2(self.activation(self.linear1(self.norm2(x)))))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(x + self._g1(self.attn(x, x, x)))
            x = self.norm2(x + self._g2(self.linear2(self.activation(self.linear1(x)))))
        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 norm_first, layer_scale=False, init_values=1e-4, norm_out=False):
        super().__init__()
        self.cross_attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = activation
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.norm_out = _NormOutTC(d_model) if (norm_first and norm_out) else None

    def _g1(self, x):
        return self.gamma_1(x) if self.gamma_1 is not None else x

    def _g2(self, x):
        return self.gamma_2(x) if self.gamma_2 is not None else x

    def __call__(self, q: mx.array, k: mx.array) -> mx.array:
        if self.norm_first:
            qn, kn = self.norm1(q), self.norm2(k)
            x = q + self._g1(self.cross_attn(qn, kn, kn))
            x = x + self._g2(self.linear2(self.activation(self.linear1(self.norm3(x)))))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self._g1(self.cross_attn(q, k, k)))
            x = self.norm2(x + self._g2(self.linear2(self.activation(self.linear1(x)))))
        return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, emb="sin", hidden_scale=4.0, num_heads=8, num_layers=6,
                 cross_first=False, dropout=0.0, max_positions=1000, norm_in=True,
                 norm_in_group=False, group_norm=False, norm_first=False, norm_out=False,
                 max_period=10000.0, weight_pos_embed=1.0, layer_scale=False, gelu=True,
                 sin_random_shift=0, cape_mean_normalize=True, cape_augment=True,
                 cape_glob_loc_scale=[5000.0, 1.0, 1.4], sparse_self_attn=False,
                 sparse_cross_attn=False, **kwargs):
        super().__init__()
        if sparse_self_attn or sparse_cross_attn:
            raise ValueError("sparse attention is not ported to the MLX backend")
        if emb != "sin":
            raise ValueError(f"only emb='sin' is ported to the MLX backend, got {emb!r}")
        if group_norm:
            raise ValueError("t_group_norm=True is not ported to the MLX backend")
        if norm_out and int(norm_out) != 1:
            # `_NormOutTC` only implements the num_groups=1 case, the only
            # value `t_norm_out: bool` can ever produce in practice.
            raise ValueError(
                f"t_norm_out={norm_out!r} (num_groups={int(norm_out)}) is not ported; "
                "only the bool True (num_groups=1) case is"
            )
        if not norm_in:
            raise ValueError("norm_in=False is not ported to the MLX backend")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        activation = nn.gelu if gelu else (lambda t: mx.maximum(t, 0))

        self.norm_in = nn.LayerNorm(dim)
        self.norm_in_t = nn.LayerNorm(dim)

        self.layers = []
        self.layers_t = []
        for idx in range(num_layers):
            classic = idx % 2 == self.classic_parity
            kwargs_layer = dict(
                d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,
                activation=activation, norm_first=norm_first, layer_scale=layer_scale,
                norm_out=bool(norm_out),
            )
            cls = TransformerEncoderLayer if classic else CrossTransformerEncoderLayer
            self.layers.append(cls(**kwargs_layer))
            self.layers_t.append(cls(**kwargs_layer))

    def _get_pos_embedding(self, T, C):
        shift = random.randrange(self.sin_random_shift + 1)
        return create_sin_embedding(T, C, shift=shift, max_period=self.max_period)

    def __call__(self, x: mx.array, xt: mx.array):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, self.max_period)
        pos_emb_2d = mx.broadcast_to(pos_emb_2d.reshape(1, C, Fr, T1), (B, C, Fr, T1))
        pos_emb_2d = pos_emb_2d.transpose(0, 3, 2, 1).reshape(B, T1 * Fr, C)

        x = x.transpose(0, 3, 2, 1).reshape(B, T1 * Fr, C)
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B2, C2, T2 = xt.shape
        xt = xt.transpose(0, 2, 1)
        pos_emb = self._get_pos_embedding(T2, C2)
        pos_emb = pos_emb.transpose(1, 0, 2)
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.reshape(B, T1, Fr, C).transpose(0, 3, 2, 1)
        xt = xt.transpose(0, 2, 1)
        return x, xt
