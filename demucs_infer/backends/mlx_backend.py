"""MLX backend -- native Apple Silicon execution behind the same seam.

Reads: numpy, torch (boundary tensor conversion only -- see
`_apply_mlx`'s `th.from_numpy` at the end of the MLX call), .base
(BackendUnavailable), ..apply (BagOfModels, for isinstance/shape checks only
-- never calls into Torch model code; imported lazily, like ..audio's
`convert_audio` and every `mlx`/`mlx_spectro` import below, per this
module's own lazy-import policy)

Builds the vendored MLX `HDemucsMLX`/`HTDemucsMLX` from a Torch model this
package's own `Separator._load_model()` already resolved, downloaded,
sha256-verified, and constructed -- so the package-owned checkpoint contract
is unchanged: no second catalog, no converted-weight cache this module
controls, and `checkpoint_runtime.py`/`checkpoint_catalog.py` stay the only
place a checkpoint is resolved. Mixed precision is forced off
(`MLX_ENABLE_AMP=0`) per the org's standing accuracy-affecting-optimizations-
are-opt-in rule, mirroring both sibling packages' MLX backends even though no
speed/accuracy measurement has been made for this port specifically.

Two refusals are deliberate and measured, not declared -- plus one narrow,
also-measured *exception* to the first that turned out to matter for most of
this package's own registry:

1. **Bags of two or more models are refused.** `apply.BagOfModels` can wrap
   an arbitrary number of sub-models with per-source weights (`htdemucs_ft`,
   `mdx`, `mdx_extra`, `cdx23_dnr`, `uvr_demucs_model_bag`, ... -- 10 of this
   package's 17 registry entries). Silently running one sub-model would
   produce plausible but wrong output; silently converting every sub-model
   and re-implementing the weighted-average combiner was in scope but was
   not the choice made here (see `CHANGELOG.md`'s entry for the reasoning)
   -- a genuine multi-model bag is refused outright, in favour of
   `backend='torch'`.

   **Exception, discovered empirically while wiring this up, not assumed:**
   `checkpoint_runtime.load_registered_model()` wraps *every*
   `demucs_package`-format recipe in a `BagOfModels`, including the 7
   single-component registry entries (`htdemucs`, `htdemucs_6s`,
   `hdemucs_mmi`, `drumsep`, `uvr_demucs_model_1`, `uvr_demucs_model_2`) --
   confirmed by loading `htdemucs` through the public `get_model()` path and
   observing a `BagOfModels` of length 1, not a bare `HTDemucs`. A
   one-submodel bag is mathematically identical to its submodel alone
   (`BagOfModels.weights` defaults to all-ones per source, and
   `apply_model()`'s per-source `estimates[:, k] /= totals[k]` divides by
   that same weight -- an identity for any nonzero weight), so `supports_model`
   unwraps a length-1 bag and measures support against its one submodel
   instead of refusing on sight. A bag of 2+ submodels is still refused.
2. **Architectures with no MLX port.** Only `HDemucs` and `HTDemucs` are
   ported (`demucs_infer/mlx/hdemucs.py`, `htdemucs.py`); the original
   time-domain-only `Demucs` class (used inside some *multi*-model bags'
   sub-models, e.g. `repro_mdx_a_time_only`'s components, but never as a
   single-component registry entry) has no MLX module here, and multi-model
   bags are refused per (1) regardless. Support is measured from
   `type(model).__name__` against what's actually implemented, never
   declared, so an unported architecture cannot be advertised and then fail
   deep inside conversion.
3. **A checkpoint that would exercise the Wiener path.** `cac=False` or
   `wiener_iters != 0` routes `_mask()` through `_wiener()`
   (`hdemucs.py`/`htdemucs.py`), which is not ported here -- see
   `demucs_infer/mlx/hdemucs.py`'s module docstring for how this was measured
   (both `htdemucs` and `hdemucs_mmi`, the two single-model checkpoints this
   backend was verified against, ship `cac=True, wiener_iters=0`).

`separate()` mirrors `apply.apply_model()`'s own shift/split/segment
arithmetic (see this module's `_apply_mlx`) rather than the sibling packages'
seam signature -- see `backends/base.py`'s module docstring for why.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch as th

from .base import BackendUnavailable

_SUPPORTED_CLASSES = ("HDemucs", "HTDemucs")


def _unwrap_single_model_bag(model):
    """A length-1 `BagOfModels` is identical to its one submodel (see this
    module's docstring); anything else (bare model, or a bag of 2+) passes
    through unchanged so the caller's own checks decide."""
    from ..apply import BagOfModels

    if isinstance(model, BagOfModels) and len(model.models) == 1:
        return model.models[0]
    return model


def supports_model(model) -> bool:
    """Whether the MLX backend can convert and run `model` -- measured from
    the model's own class/attributes, never declared in advance."""
    from ..apply import BagOfModels

    model = _unwrap_single_model_bag(model)
    if isinstance(model, BagOfModels):
        return False
    if type(model).__name__ not in _SUPPORTED_CLASSES:
        return False
    if not getattr(model, "cac", True):
        return False
    return getattr(model, "wiener_iters", 0) == 0


def _explain_unsupported(model) -> str:
    from ..apply import BagOfModels

    original = model
    model = _unwrap_single_model_bag(model)
    if isinstance(model, BagOfModels):
        return (
            "backend 'mlx' does not support a BagOfModels of 2+ sub-models (this "
            f"checkpoint assembles {len(model.models)}); use backend='torch' for this model"
        )
    class_name = type(model).__name__
    if class_name not in _SUPPORTED_CLASSES:
        return (
            f"backend 'mlx' has no port of {class_name!r} (only {_SUPPORTED_CLASSES} "
            "are ported); use backend='torch' for this model"
        )
    if not getattr(model, "cac", True) or getattr(model, "wiener_iters", 0) != 0:
        return (
            "backend 'mlx' does not port the Wiener-filtering path "
            f"(cac={getattr(model, 'cac', None)!r}, "
            f"wiener_iters={getattr(model, 'wiener_iters', None)!r}); "
            "use backend='torch' for this model"
        )
    del original
    return f"backend 'mlx' cannot run this {class_name}"


class MLXBackend:
    """Runs `HDemucs`/`HTDemucs` natively on Apple Silicon through MLX."""

    name = "mlx"

    def __init__(self, mlx_model, sources, samplerate: int, audio_channels: int, device: str = "mps"):
        self._model = mlx_model
        self._sources = sources
        self._samplerate = samplerate
        self._audio_channels = audio_channels
        self._device = device

    # ------------------------------------------------------------- availability

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_spectro  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def _require(cls) -> None:
        if not cls.is_available():
            raise BackendUnavailable(
                "the MLX backend needs the optional extra: "
                "pip install 'demucs-infer[mlx]' (Apple Silicon)"
            )

    @staticmethod
    def supports_model(model) -> bool:
        return supports_model(model)

    @staticmethod
    def assert_supports_model(model) -> None:
        if supports_model(model):
            return
        raise BackendUnavailable(_explain_unsupported(model))

    # ------------------------------------------------------------- construction

    @classmethod
    def from_torch_model(cls, torch_model, device: str = "mps") -> MLXBackend:
        """Convert an already-loaded, already-verified Torch model in place.

        This is the only constructor: it never touches a checkpoint path, a
        URL, or `checkpoint_runtime.py`'s resolver -- `torch_model` is
        whatever `Separator._load_model()` already produced (a bare
        `HDemucs`/`HTDemucs`, or a `BagOfModels` -- unwrapped here if it has
        exactly one submodel, see this module's docstring).
        """
        cls._require()
        cls.assert_supports_model(torch_model)
        device = cls._select_device(device)
        torch_model = _unwrap_single_model_bag(torch_model)

        os.environ.setdefault("MLX_ENABLE_AMP", "0")

        from ..mlx import (
            HDemucsMLX,
            HTDemucsMLX,
            convert_state_dict,
            load_converted_weights,
        )

        if not hasattr(torch_model, "_init_args_kwargs"):
            raise BackendUnavailable(
                f"{type(torch_model).__name__} was not constructed with capture_init; "
                "cannot recover its constructor kwargs for MLX conversion"
            )
        _args, kwargs = torch_model._init_args_kwargs
        mlx_class = HTDemucsMLX if type(torch_model).__name__ == "HTDemucs" else HDemucsMLX
        mlx_model = mlx_class(**kwargs)

        weights = convert_state_dict(torch_model)
        load_converted_weights(mlx_model, weights)
        # `BagOfModels.__init__` can mutate a submodel's `.segment` in place
        # (its own `segment=` override); `_init_args_kwargs` only captured
        # the value at construction time, so the *current* attribute -- not
        # the captured kwarg -- is this checkpoint's real chunking segment.
        mlx_model.segment = torch_model.segment
        mlx_model.eval()

        return cls(
            mlx_model,
            sources=torch_model.sources,
            samplerate=torch_model.samplerate,
            audio_channels=torch_model.audio_channels,
            device=device,
        )

    @staticmethod
    def _select_device(device):
        """MLX owns its own execution target; a Torch device string is refused."""
        if device in (None, "auto", "mps"):
            return "mps"
        raise BackendUnavailable(
            f"backend 'mlx' cannot honour device {device!r}; it executes on Apple "
            f"Silicon and accepts None, 'auto', or 'mps'. Use backend='torch' to "
            f"select a Torch device."
        )

    # ----------------------------------------------------------------- protocol

    @property
    def resolved_device(self) -> str:
        return self._device

    @property
    def sources(self):
        return self._sources

    @property
    def model(self):
        """The resident MLX model, for callers composing the advanced path directly."""
        return self._model

    def release(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    def separate(
        self,
        wav: th.Tensor,
        sr: Optional[int] = None,
        *,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[float] = None,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """Resample + reference-normalize + chunked-apply, mirroring
        `Separator.separate_tensor()`'s own contract (see `backends/base.py`).
        """
        from ..audio import convert_audio

        if sr is not None and sr != self._samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        ref = wav.mean(0)
        wav = wav - ref.mean()
        wav = wav / (ref.std() + 1e-8)

        import mlx.core as mx

        mix = mx.array(np.ascontiguousarray(wav.detach().cpu().numpy(), dtype=np.float32))[None]
        out = _apply_mlx(
            self._model,
            mix,
            shifts=shifts,
            split=split,
            overlap=overlap,
            segment=segment,
            samplerate=self._samplerate,
            num_sources=len(self._sources),
            progress=progress,
        )
        mx.eval(out)
        out_th = th.from_numpy(np.array(out, dtype=np.float32))

        out_th = out_th * (ref.std() + 1e-8)
        out_th = out_th + ref.mean()
        wav = wav * (ref.std() + 1e-8)
        wav = wav + ref.mean()
        return wav, out_th


def _apply_mlx(model, mix, *, shifts, split, overlap, segment, samplerate, num_sources,
               transition_power: float = 1.0, progress: bool = False):
    """Entry point: `mix` is the whole (already normalized) mixture, offset 0."""
    return _apply_chunk(
        model, mix, 0, mix.shape[-1],
        shifts=shifts, split=split, overlap=overlap, segment=segment,
        samplerate=samplerate, num_sources=num_sources,
        transition_power=transition_power, progress=progress,
    )


def _padded_from_root(root, offset: int, length: int, target_length: int):
    """MLX mirror of `apply.TensorChunk.padded()`: center `target_length` on
    the physical window `[offset, offset + length)` inside `root`, pulling
    real neighbouring samples from `root` where available and zero-padding
    only what falls outside `root`'s true bounds. This is *not* a symmetric
    zero-pad of the chunk in isolation -- a naive "pad this chunk's own
    edges with zeros" implementation would silently disagree with Torch on
    every chunk that needs padding (in practice, a track's final chunk on
    every run), which is exactly the class of silence-adjacent divergence
    `tests/test_mlx_parity.py`'s zero-padded-tail case exists to catch.
    """
    import mlx.core as mx

    total_length = root.shape[-1]
    delta = target_length - length
    start = offset - delta // 2
    end = start + target_length
    correct_start = max(0, start)
    correct_end = min(total_length, end)
    pad_left = correct_start - start
    pad_right = end - correct_end
    out = root[..., correct_start:correct_end]
    if pad_left or pad_right:
        out = mx.pad(out, [(0, 0)] * (out.ndim - 1) + [(pad_left, pad_right)])
    return out


def _apply_chunk(model, root, offset: int, length: int, *, shifts, split, overlap, segment,
                  samplerate, num_sources, transition_power: float = 1.0, progress: bool = False):
    """MLX mirror of `apply.apply_model()`'s shift/split/segment recursion.

    `root` is the physical array every `offset` is relative to -- carried
    unchanged through the split recursion within one shift iteration
    (mirroring `apply.TensorChunk` composing offsets against one underlying
    tensor rather than re-slicing on every recursive call), and replaced by
    a freshly materialized padded array only when `shifts` itself pads.

    Deliberately not thread/process-parallelized across segments (unlike
    Torch's `ThreadPoolExecutor`/`pool` machinery): each MLX forward already
    runs on the GPU's own async command queue, and a serial Python loop
    produces identical numbers with far less code to keep in parity.
    `shifts>0` draws from the same unseeded stdlib `random` module
    `apply_model()` does (see its header's KNOWN QUIRK), so interleaving
    Torch and MLX calls in one process still consumes the shared RNG stream
    exactly as today.
    """
    import mlx.core as mx

    if shifts:
        max_shift = int(0.5 * samplerate)
        padded = _padded_from_root(root, offset, length, length + 2 * max_shift)
        total = 0.0
        for _ in range(shifts):
            shift_offset = random.randint(0, max_shift)
            shift_length = length + max_shift - shift_offset
            res = _apply_chunk(
                model, padded, shift_offset, shift_length,
                shifts=0, split=split, overlap=overlap, segment=segment,
                samplerate=samplerate, num_sources=num_sources,
                transition_power=transition_power, progress=progress,
            )
            total = total + res[..., max_shift - shift_offset:]
        return total / shifts

    if split:
        batch, channels = root.shape[0], root.shape[1]
        if segment is None:
            segment = model.segment
        segment_length = int(samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = list(range(0, length, stride))

        weight_np = np.concatenate([
            np.arange(1, segment_length // 2 + 1, dtype=np.float32),
            np.arange(segment_length - segment_length // 2, 0, -1, dtype=np.float32),
        ])
        weight_np = (weight_np / weight_np.max()) ** transition_power
        weight = mx.array(weight_np)

        out = mx.zeros((batch, num_sources, channels, length), dtype=mx.float32)
        sum_weight = mx.zeros((length,), dtype=mx.float32)

        iterator = offsets
        if progress:
            import tqdm

            iterator = tqdm.tqdm(offsets, ncols=120, unit="chunk")

        for rel_offset in iterator:
            chunk_len = min(segment_length, length - rel_offset)
            chunk_out = _apply_chunk(
                model, root, offset + rel_offset, chunk_len,
                shifts=0, split=False, overlap=overlap, segment=segment,
                samplerate=samplerate, num_sources=num_sources,
                transition_power=transition_power, progress=False,
            )
            out_len = chunk_out.shape[-1]
            w = weight[:out_len]
            pad_right = length - rel_offset - out_len
            # `out.at[..., span].add(...)` -- MLX 0.31.2's indexed-update
            # scatter-add -- was measured to silently corrupt large 4-D
            # updates at this scale (reproduced in isolation: an all-ones
            # update over a ~344k-of-353k-sample span landed 4x too small at
            # some positions and not at all at others). Padding the
            # already-weighted chunk out to the full length and adding it as
            # a dense elementwise op sidesteps that path entirely and is the
            # one that's actually been measured correct end to end (see
            # `tests/test_mlx_parity.py`).
            out = out + mx.pad(w * chunk_out, [(0, 0)] * 3 + [(rel_offset, pad_right)])
            sum_weight = sum_weight + mx.pad(w, [(rel_offset, pad_right)])

        if bool(mx.any(sum_weight <= 0)):
            raise RuntimeError("zero accumulation weight while chunking; segment/overlap mismatch")
        return out / sum_weight

    # Whole-chunk forward, mirroring apply_model()'s non-split branch.
    if hasattr(model, "valid_length"):
        valid_length = model.valid_length(length)
    else:
        valid_length = length
    padded = _padded_from_root(root, offset, length, valid_length)
    out = model(padded)
    return _center_trim(out, length)


def _center_trim(x, reference_length: int):
    delta = x.shape[-1] - reference_length
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        start = delta // 2
        x = x[..., start:x.shape[-1] - (delta - start)]
    return x
