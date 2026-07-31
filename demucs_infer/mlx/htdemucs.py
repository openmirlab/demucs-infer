"""Vendored+adapted MLX `HTDemucs` -- the `htdemucs`/`htdemucs_6s`/
`htdemucs_ft`/`msst_htdemucs_vocals` architecture (the default `htdemucs`
registry model).

Ports `htdemucs.py`'s forward pass (encoder/decoder from `hdemucs.py`,
transformer bottleneck from `transformer.py`) to MLX. Structure and
arithmetic are carried field-for-field from `htdemucs.py`; only tensor
layout and module wiring differ. See `hdemucs.py`'s module docstring for
what "vendored+adapted" means here (unfused norm/activation, Wiener path
dropped) -- the same choices apply.

**Whether `exact_zero_safe_rfft` is load-bearing here was measured, not
assumed.** HTDemucs has two properties the sibling packages' findings split
on: like `bs-roformer-infer`'s model, it has cross-attention over the time
axis that could spread one corrupted STFT frame across every position; like
`mdxnet-infer`'s model, its normalization epsilon (`1e-5`, see `__call__`
below, matching `htdemucs.py`'s `(1e-5 + std)`) is seven orders above the
~4.5e-07 rfft artifact rather than `bs-roformer-infer`'s `1e-12`, so the
artifact does not get renormalized into a full-scale spurious feature the way
it does there. Measured on the real `htdemucs` checkpoint through the public
`Separator.separate_audio_file()` API (`tests/test_mlx_parity.py`,
`test_rfft_guard_removed_stays_within_noise_floor`), worst-case max-abs
Torch-vs-MLX divergence on a 9.3s fixture with a zero-padded tail:

    with the guard:    1.937e-07
    guard removed:     2.533e-07

Both in the same ~1e-7 noise floor (the clean-signal case measured
5.411e-07 with the guard, 3.912e-07 without) -- i.e. inert here, closer to
`mdxnet-infer`'s result than `bs-roformer-infer`'s (which diverged by
1.455e-02 with the guard removed). Kept anyway (cheap, and the org's
standing policy after finding this class of bug twice); see `rfft_guard.py`.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/demucs_mlx/mlx_htdemucs.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.
    Adapted: fused Metal-kernel norm/activation and the Wiener path dropped
    (see hdemucs.py); sparse-attention configurations refused rather than
    silently built (see transformer.py).

Reads: mlx.core, mlx.nn, .hdemucs (HEncLayer, HDecLayer, MultiWrap,
ScaledEmbedding, pad1d), .transformer (CrossTransformerEncoder), .layers
(Conv1dNCL), .spec (spectro, ispectro)
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .hdemucs import HDecLayer, HEncLayer, MultiWrap, ScaledEmbedding, pad1d
from .layers import Conv1dNCL
from .spec import ispectro, spectro
from .transformer import CrossTransformerEncoder


def _center_trim(x: mx.array, reference_length: int) -> mx.array:
    delta = x.shape[-1] - reference_length
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        start = delta // 2
        x = x[..., start:x.shape[-1] - (delta - start)]
    return x


class HTDemucsMLX(nn.Module):
    def __init__(self, sources, audio_channels=2, channels=48, channels_time=None, growth=2,
                 nfft=4096, wiener_iters=0, end_iters=0, wiener_residual=False, cac=True,
                 depth=4, rewrite=True, multi_freqs=None, multi_freqs_depth=3, freq_emb=0.2,
                 emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2, stride=4,
                 context=1, context_enc=0, norm_starts=4, norm_groups=4, dconv_mode=1,
                 dconv_depth=2, dconv_comp=8, dconv_init=1e-3, bottom_channels=0,
                 t_layers=5, t_emb="sin", t_hidden_scale=4.0, t_heads=8, t_dropout=0.0,
                 t_max_positions=10000, t_norm_in=True, t_norm_in_group=False,
                 t_group_norm=False, t_norm_first=True, t_norm_out=True, t_max_period=10000.0,
                 t_weight_decay=0.0, t_lr=None, t_layer_scale=True, t_gelu=True,
                 t_weight_pos_embed=1.0, t_sin_random_shift=0, t_cape_mean_normalize=True,
                 t_cape_augment=True, t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
                 t_sparse_self_attn=False, t_sparse_cross_attn=False, t_mask_type="diag",
                 t_mask_random_seed=42, t_sparse_attn_window=500, t_global_window=100,
                 t_sparsity=0.95, t_auto_sparsity=False, t_cross_first=False, rescale=0.1,
                 samplerate=44100, segment=10, use_train_segment=True):
        super().__init__()
        if not cac:
            raise ValueError("HTDemucsMLX only supports cac=True; the Wiener path is not ported")
        if wiener_iters != end_iters:
            raise ValueError("HTDemucs requires wiener_iters == end_iters")

        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None

        self.encoder = []
        self.decoder = []
        self.tencoder = []
        self.tdecoder = []

        chin = audio_channels
        chin_z = chin * (2 if self.cac else 1)
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                ker = time_stride * 2
                stri = time_stride
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker, "stride": stri, "freq": freq, "pad": pad, "norm": norm,
                "rewrite": rewrite, "norm_groups": norm_groups,
                "dconv_kw": {"depth": dconv_depth, "compress": dconv_comp,
                             "init": dconv_init, "gelu": True},
            }
            kwt = dict(kw)
            kwt["freq"] = False
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, dconv=bool(dconv_mode & 1), context=context_enc, **kw)
            if freq:
                tenc = HEncLayer(chin, chout, dconv=bool(dconv_mode & 1), context=context_enc,
                                  empty=last_freq, **kwt)
                self.tencoder.append(tenc)
            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * (2 if self.cac else 1)

            dec = HDecLayer(chout_z, chin_z, dconv=bool(dconv_mode & 2), last=index == 0,
                             context=context, **kw_dec)
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(chout, chin, dconv=bool(dconv_mode & 2), empty=last_freq,
                                  last=index == 0, context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        transformer_channels = channels * (growth ** (depth - 1))
        self.bottom_channels = bottom_channels
        if bottom_channels:
            self.channel_upsampler = Conv1dNCL(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = Conv1dNCL(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = Conv1dNCL(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = Conv1dNCL(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels, emb=t_emb, hidden_scale=t_hidden_scale,
                num_heads=t_heads, num_layers=t_layers, cross_first=t_cross_first,
                dropout=t_dropout, max_positions=t_max_positions, norm_in=t_norm_in,
                norm_in_group=t_norm_in_group, group_norm=t_group_norm, norm_first=t_norm_first,
                norm_out=t_norm_out, max_period=t_max_period, weight_pos_embed=t_weight_pos_embed,
                layer_scale=t_layer_scale, gelu=t_gelu, sin_random_shift=t_sin_random_shift,
                cape_mean_normalize=t_cape_mean_normalize, cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale, sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = spectro(x, nfft, hl)[..., :-1, :]
        return z[..., 2:2 + le]

    def _ispec(self, z, length=None):
        hl = self.hop_length
        z = mx.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (0, 0)])
        z = mx.pad(z, [(0, 0)] * (z.ndim - 1) + [(2, 2)])
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        return x[..., pad:pad + length]

    def _magnitude(self, z):
        B, C, Fr, T = z.shape
        return mx.stack([mx.real(z), mx.imag(z)], axis=2).reshape(B, C * 2, Fr, T)

    def _mask(self, m):
        B, S, C, Fr, T = m.shape
        out = m.reshape(B, S, -1, 2, Fr, T).transpose(0, 1, 2, 4, 5, 3)
        return out[..., 0] + 1j * out[..., 1]

    def valid_length(self, length: int) -> int:
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                f"Given length {length} is longer than training length {training_length}")
        return training_length

    def __call__(self, mix: mx.array) -> mx.array:
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length_pre_pad)])

        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        # 1e-5 eps, matching htdemucs.py's `(1e-5 + std)` -- see this module's
        # docstring for why this is one of the two data points that made the
        # rfft-guard measurement here come out "inert" rather than assumed.
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.std(xt, axis=(1, 2), keepdims=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = mx.arange(x.shape[-2], dtype=mx.int32)
                emb = self.freq_emb(frs).transpose(1, 0)[None, :, :, None]
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = self.channel_upsampler(x)
                x = x.reshape(b, self.bottom_channels, f, t)
                xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt)
            if self.bottom_channels:
                x = x.reshape(b, self.bottom_channels, f * t)
                x = self.channel_downsampler(x)
                x = x.reshape(b, c, f, t)
                xt = self.channel_downsampler_t(xt)

        offset = self.depth - len(self.tdecoder)
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        if saved or lengths_t or saved_t:
            raise RuntimeError("skip connections not fully consumed")

        S = len(self.sources)
        x = x.reshape(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(x)
        target_length = training_length if self.use_train_segment else length
        x = self._ispec(zout, target_length)

        actual_length = xt.shape[-1]
        xt = xt.reshape(B, S, -1, actual_length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = _center_trim(x, xt.shape[-1])
        x = xt + x
        x = x[..., :target_length]
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
