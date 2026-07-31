"""MLX STFT/iSTFT wrappers matching spec.py's `spectro`/`ispectro` contract.

Torch's `spec.py` calls `torch.stft`/`torch.istft` with `normalized=True,
center=True, pad_mode='reflect'`, a Hann window sized `n_fft`, and reshapes
around a `(*other, freqs, frames)` batch-of-anything layout. This module
gets the same numbers from `mlx_spectro.get_transform_mlx()` (see
`exact_zero_safe_rfft` in `htdemucs.py` for the one deliberate numerical
deviation applied around every call here) and reproduces `spec.py`'s exact
reshape conventions for the 3-D `(B, C, T)` / 4-D `(B, C, F, N)` shapes
`hdemucs.py`/`htdemucs.py` actually use -- the 1-D/2-D/5-D cases the vendored
`mlx-audio-separator` project's own `spec_mlx.py` also handles are outside
what this package's registry needs and are not ported.

Depends on `mlx-spectro` (MIT, floors only -- see pyproject.toml's `[mlx]`
extra), the same dependency both sibling OpenMIRLab MLX ports use, rather than
vendoring STFT math: it is a real, permissively-licensed spectral library, not
a shim.

Reads: mlx.core, mlx_spectro, .rfft_guard (exact_zero_safe_rfft)
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx_spectro import get_transform_mlx, resolve_fft_params

from .rfft_guard import exact_zero_safe_rfft


def _transform(n_fft: int, hop_length: int, pad: int = 0):
    eff_n_fft, hop, win = resolve_fft_params(int(n_fft), hop_length, None, int(pad))
    return get_transform_mlx(
        n_fft=eff_n_fft,
        hop_length=hop,
        win_length=win,
        window_fn="hann",
        periodic=True,
        center=True,
        normalized=True,
        window=None,
    )


def spectro(x: mx.array, n_fft: int = 512, hop_length: Optional[int] = None, pad: int = 0) -> mx.array:
    """`(B, C, T) -> (B, C, F, N)` complex STFT, matching `spec.spectro`."""
    hop = hop_length if hop_length is not None else n_fft // 4
    transform = _transform(n_fft, hop, pad)
    B, C, T = x.shape
    x2 = mx.contiguous(x).reshape(B * C, T)
    with exact_zero_safe_rfft():
        spec2 = transform.stft(x2)
        mx.eval(spec2)
    return spec2.reshape(B, C, spec2.shape[1], spec2.shape[2])


def ispectro(z: mx.array, hop_length: Optional[int] = None, length: Optional[int] = None,
             pad: int = 0) -> mx.array:
    """`(B, C, F, N) -> (B, C, T)` inverse STFT, matching `spec.ispectro`."""
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    hop = hop_length if hop_length is not None else n_fft // 4
    transform = _transform(n_fft, hop, pad)
    z2 = mx.contiguous(z).reshape(-1, freqs, frames)
    with exact_zero_safe_rfft():
        wav2 = transform.istft(z2, length=length)
        mx.eval(wav2)
    return wav2.reshape(*other, wav2.shape[-1])
