"""`exact_zero_safe_rfft` -- routes `mx.fft.rfft` through the CPU stream.

MLX 0.31.2's Metal rfft kernel packs two real FFTs into one complex FFT; in
float32 that cancellation is not bit-exact, so a frame whose true value is
exactly zero comes back as roughly 4.5e-07 instead of 0. In the sibling
`bs-roformer-infer` package this was severely load-bearing (250,000x worse
without it: a zero-padded chunk diverged from Torch by 1.455e-02 max abs,
vs. 2.012e-07 with the guard) because that model's `L2Norm` divides by a
near-zero magnitude with `eps=1e-12` (five orders below the artifact, so the
clamp never engages) and then spreads the corrupted frame across every
time position via attention.

`mdxnet-infer` measured the *same* guard **inert** for its architecture: no
eps that small, no attention to spread a corrupted frame.

demucs-infer's HDemucs/HTDemucs sit in between those two data points and were
not assumed either way -- see `demucs_infer/mlx/htdemucs.py`'s module
docstring for how this was actually measured (both models normalize with
`eps=1e-5`, seven orders above the rfft artifact, and HTDemucs additionally
has cross-attention that could spread a corrupted frame the way the roformer
does). Measured on the real `htdemucs` checkpoint through the public API
with a zero-padded-tail fixture: 1.937e-07 with the guard, 2.533e-07 with it
removed -- inert here, matching `mdxnet-infer`'s result rather than
`bs-roformer-infer`'s. The guard is applied unconditionally regardless of
the measurement, per the org's standing policy of cheap insurance once a
real bug in this class has already been found.

Caveat, stated rather than hidden: this swaps a module-level attribute, so it
is not thread-safe. Inference here is single-threaded per session.

Delete this once MLX's rfft kernel is fixed upstream.

Reads: mlx.core
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def exact_zero_safe_rfft():
    """Context manager: `mx.fft.rfft` runs on the CPU stream while active."""
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original
