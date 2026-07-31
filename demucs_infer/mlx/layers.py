"""Vendored MLX layer primitives -- Conv/GroupNorm/BLSTM/LocalState/DConv/LayerScale.

Reads: mlx.core, mlx.nn

Adapted from mlx-audio-separator's `demucs_mlx/mlx_layers.py` and
`demucs_mlx/mlx_demucs.py`, which implement the same layer stack hdemucs.py's
`HEncLayer`/`HDecLayer`/`DConv` and demucs.py's `BLSTM`/`LocalState`/`DConv`
use, in MLX's NLC/NHWC-native layout instead of Torch's NCL/NCHW. Every
Conv/GroupNorm class here is a thin transpose-wrapper so parameter names and
shapes still line up 1:1 with the Torch module tree that `convert.py` walks.

**Deliberate deviation from upstream**: upstream additionally ships
`FusedGroupNormGELU`/`FusedGroupNormGLU`, which replace GroupNorm+activation
with a custom Metal kernel (`metal_kernels.py`, not vendored here). This port
always takes the unfused path -- plain `GroupNorm` followed by a separate
GELU/GLU call. A fused kernel that is "numerically close" to the unfused
composition is exactly the class of divergence this org has already paid for
once (see `exact_zero_safe_rfft`'s history in the sibling `bs-roformer-infer`/
`mdxnet-infer` packages); the unfused path is the one this port's parity tests
actually exercise. Parameter names (`weight`/`bias`) are identical either way,
so this choice does not change `convert.py`'s key mapping.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    Files:    mlx_audio_separator/demucs_mlx/mlx_layers.py,
              mlx_audio_separator/demucs_mlx/mlx_demucs.py (BLSTM, LocalState,
              DConv, LayerScale, GroupNorm, resampling helpers not used here)
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.
"""

from __future__ import annotations

import math
import typing as tp
from functools import lru_cache

import mlx.core as mx
from mlx import nn


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Lambda(nn.Module):
    """Wraps a stateless function as a (parameter-free) module, so it still
    occupies a slot in an `nn.Sequential`'s index-addressed children -- this
    keeps Sequential child indices aligned with Torch's, which `convert.py`'s
    key mapping (and `load_converted_weights`'s key audit) depends on."""

    def __init__(self, fn: tp.Callable[[mx.array], mx.array]):
        super().__init__()
        self.fn = fn

    def __call__(self, x: mx.array) -> mx.array:
        return self.fn(x)


class Conv1dNCL(nn.Module):
    """Conv1d wrapper for NCL (Batch, Channels, Length) layout; MLX's native
    Conv1d expects NLC, so inputs/outputs are transposed at the boundary."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)
        y = self.conv(x)
        return y.transpose(0, 2, 1)


class ConvTranspose1dNCL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        output_padding=output_padding, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)
        y = self.conv(x)
        return y.transpose(0, 2, 1)


class Conv2dNCHW(nn.Module):
    """Conv2d wrapper for NCHW layout; MLX's native Conv2d expects NHWC."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)
        y = self.conv(x)
        return y.transpose(0, 3, 1, 2)


class ConvTranspose2dNCHW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        output_padding=output_padding, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)
        y = self.conv(x)
        return y.transpose(0, 3, 1, 2)


def _group_norm_nd(x: mx.array, num_groups: int, weight, bias, eps: float) -> mx.array:
    """Shared GroupNorm math for NCL/NCHW layouts (channel axis 1)."""
    B, C = x.shape[0], x.shape[1]
    G = num_groups
    if C % G != 0:
        raise ValueError(f"num_channels {C} not divisible by num_groups {G}")
    x_reshaped = x.reshape(B, G, C // G, *x.shape[2:])
    axes = tuple(range(2, x_reshaped.ndim))
    mean = x_reshaped.mean(axis=axes, keepdims=True)
    var = ((x_reshaped - mean) ** 2).mean(axis=axes, keepdims=True)
    x_norm = (x_reshaped - mean) * mx.rsqrt(var + eps)
    x_out = x_norm.reshape(x.shape)
    if weight is not None:
        shape = [1, C] + [1] * (x_out.ndim - 2)
        x_out = x_out * weight.reshape(shape) + bias.reshape(shape)
    return x_out


class GroupNormNCL(nn.Module):
    """GroupNorm over an NCL (Batch, Channels, Length) tensor, channel axis 1."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = mx.ones((num_channels,), dtype=mx.float32)
            self.bias = mx.zeros((num_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return _group_norm_nd(
            x, self.num_groups,
            self.weight if self.affine else None,
            self.bias if self.affine else None,
            self.eps,
        )


class GroupNormNCHW(nn.Module):
    """GroupNorm over an NCHW (Batch, Channels, Height, Width) tensor, channel axis 1."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = mx.ones((num_channels,), dtype=mx.float32)
            self.bias = mx.zeros((num_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return _group_norm_nd(
            x, self.num_groups,
            self.weight if self.affine else None,
            self.bias if self.affine else None,
            self.eps,
        )


class GLUNCL(nn.Module):
    """Gated Linear Unit splitting the channel axis (axis=1), matching torch.nn.GLU(dim=1)."""

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def __call__(self, x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=self.axis)
        return a * mx.sigmoid(b)


class GELUNCL(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(x)


class LayerScale(nn.Module):
    """Learned per-channel residual-branch scale (Touvron et al. 2021)."""

    def __init__(self, channels: int, init: float = 0.0, channel_last: bool = False):
        super().__init__()
        self.channel_last = bool(channel_last)
        self.scale = mx.zeros((channels,), dtype=mx.float32) + float(init)

    def __call__(self, x: mx.array) -> mx.array:
        if self.channel_last:
            return x * self.scale
        return x * self.scale[:, None]


def unfold(x: mx.array, kernel_size: int, stride: int) -> mx.array:
    """Frame the last axis into overlapping windows -- MLX equivalent of
    demucs_infer.utils.unfold, used by BLSTM's chunked mode."""
    *shape, length = x.shape
    n_frames = int(math.ceil(length / stride))
    tgt_length = (n_frames - 1) * stride + kernel_size
    pad = tgt_length - length
    if pad > 0:
        x = mx.pad(x, [(0, 0)] * len(shape) + [(0, pad)], mode="constant")
    x = mx.contiguous(x)
    current_strides = [1] * x.ndim
    for i in range(x.ndim - 2, -1, -1):
        current_strides[i] = current_strides[i + 1] * x.shape[i + 1]
    new_strides = current_strides[:-1] + [stride, 1]
    return mx.as_strided(x, shape=[*shape, n_frames, kernel_size], strides=new_strides)


class BLSTM(nn.Module):
    """Bidirectional LSTM with a linear projection back to `dim`, used inside
    `DConv`'s residual branch. Torch's single bidirectional `nn.LSTM` is split
    into separate forward/backward `nn.LSTM` instances here because MLX's
    `nn.LSTM` has no native bidirectional mode; `convert.py` remaps Torch's
    packed `weight_ih_l{N}[_reverse]` naming onto this split."""

    def __init__(self, dim: int, layers: int = 1, max_steps: tp.Optional[int] = None, skip: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.skip = skip
        self.layers = layers
        self.forward_lstms = [
            nn.LSTM(input_size=dim if i == 0 else 2 * dim, hidden_size=dim) for i in range(layers)
        ]
        self.backward_lstms = [
            nn.LSTM(input_size=dim if i == 0 else 2 * dim, hidden_size=dim) for i in range(layers)
        ]
        self.linear = nn.Linear(2 * dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.transpose(0, 2, 3, 1).reshape(-1, width, C)
        else:
            x = x.transpose(0, 2, 1)

        seq = x
        for lstm_f, lstm_b in zip(self.forward_lstms, self.backward_lstms):
            f_out, _ = lstm_f(seq)
            b_out, _ = lstm_b(seq[:, ::-1, :])
            b_out = b_out[:, ::-1, :]
            seq = mx.concatenate([f_out, b_out], axis=-1)

        x = self.linear(seq)
        x = x.transpose(0, 2, 1)

        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            x = mx.concatenate(out, axis=-1)[..., :T]

        if self.skip:
            x = x + y
        return x


@lru_cache(maxsize=32)
def _delta_eye_cached(T: int) -> tp.Tuple[mx.array, mx.array]:
    indexes = mx.arange(T, dtype=mx.float32)
    delta = indexes[:, None] - indexes[None, :]
    eye = mx.eye(T, dtype=mx.bool_)
    return delta, eye


class LocalState(nn.Module):
    """Local (data-only) attention with a learned time-decay penalty, used
    inside `DConv`'s residual branch when `attn=True`."""

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = Conv1dNCL(channels, channels, 1)
        self.query = Conv1dNCL(channels, channels, 1)
        self.key = Conv1dNCL(channels, channels, 1)
        if nfreqs:
            self.query_freqs = Conv1dNCL(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = Conv1dNCL(channels, heads * ndecay, 1)
        self.proj = Conv1dNCL(channels + heads * nfreqs, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T = x.shape
        heads = self.heads
        delta, eye = _delta_eye_cached(T)
        delta = delta.astype(x.dtype)

        queries = self.query(x).reshape(B, heads, -1, T)
        keys = self.key(x).reshape(B, heads, -1, T)
        dots = mx.matmul(keys.transpose(0, 1, 3, 2), queries) * (1.0 / math.sqrt(keys.shape[2]))

        freq_kernel = None
        if self.nfreqs:
            periods = mx.arange(1, self.nfreqs + 1, dtype=x.dtype)
            freq_kernel = mx.cos(2 * math.pi * delta / periods.reshape(-1, 1, 1))
            freq_q = self.query_freqs(x).reshape(B, heads, -1, T) / math.sqrt(self.nfreqs)
            dots = dots + mx.einsum("fts,bhfs->bhts", freq_kernel, freq_q)

        if self.ndecay:
            decays = mx.arange(1, self.ndecay + 1, dtype=x.dtype)
            decay_q = self.query_decay(x).reshape(B, heads, -1, T)
            decay_q = mx.sigmoid(decay_q) / 2
            coeff = (decay_q * decays.reshape(1, 1, -1, 1)).sum(axis=2)
            dots = dots - (
                mx.abs(delta).reshape(1, 1, T, T)
                * coeff.reshape(B, heads, 1, T)
                / math.sqrt(self.ndecay)
            )

        dots = mx.where(eye, mx.array(-100.0, dtype=dots.dtype), dots)
        weights = mx.softmax(dots, axis=2)

        content = self.content(x).reshape(B, heads, -1, T)
        # Torch: `einsum("bhts,bhct->bhcs", weights, content)` -- output free
        # index `s` is weights' *second* axis, summed index `t` is its
        # first. That's `content @ weights` (content's last axis `t` paired
        # against weights' `t` axis), not `weights @ content^T` (which was
        # tried first and pairs `s` against `t` instead -- same shapes, subtly
        # wrong result, caught by comparing this module in isolation against
        # Torch with real trained weights when the full model's parity
        # diverged sharply only from the first `dconv_attn`-eligible layer
        # onward).
        result = mx.matmul(content, weights)
        if self.nfreqs:
            time_sig = mx.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = mx.concatenate([result, time_sig], axis=2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class DConv(nn.Module):
    """Dilated-conv residual branch, optionally with a local-attention and/or
    BLSTM sub-branch -- shared by every `HEncLayer`/`HDecLayer` in both
    `HDemucs` and `HTDemucs`. Layer order (Conv, GroupNorm, act, [attn/lstm],
    Conv, GroupNorm, GLU, LayerScale) matches `demucs.py`'s `DConv.layers`
    `nn.Sequential` index-for-index, which `convert.py` depends on.

    Each depth's block is stored as a **plain Python list** (`self.layers`
    is a list of lists), not wrapped in an intermediate `nn.Module`/
    `nn.Sequential`-alike: MLX flattens nested plain lists by index directly
    (`layers.<d>.<i>.weight`, matching Torch's `nn.Sequential` numbering
    exactly), whereas an extra wrapper module would insert its own attribute
    name into the path (`layers.<d>.layers.<i>.weight`) and silently
    diverge from Torch's naming -- `load_converted_weights` caught exactly
    this on the first real-checkpoint conversion attempt.
    """

    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm: bool = True, attn: bool = False, heads: int = 4, ndecay: int = 4,
                 lstm: bool = False, gelu: bool = True, kernel: int = 3, dilate: bool = True):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError("kernel must be odd")
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        def norm_fn(d):
            return GroupNormNCL(1, d) if norm else Identity()

        hidden = int(channels / compress)
        act_fn = nn.gelu if gelu else (lambda t: mx.maximum(t, 0))

        self.layers = []
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            mods: tp.List[nn.Module] = [
                Conv1dNCL(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden),
                Lambda(act_fn),
                Conv1dNCL(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                Lambda(lambda t: GLUNCL(axis=1)(t)),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            self.layers.append(mods)

    def __call__(self, x: mx.array) -> mx.array:
        for mods in self.layers:
            y = x
            for m in mods:
                y = m(y)
            x = x + y
        return x
