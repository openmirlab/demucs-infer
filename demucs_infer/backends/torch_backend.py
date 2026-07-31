"""PyTorch backend -- the shipped path, moved behind the seam unchanged.

`separate()` is `Separator.separate_tensor()`'s pre-existing body (resample via
`audio.convert_audio`, reference-normalize, `apply.apply_model()`, denormalize)
verbatim -- a wrapper, never a second implementation, so the Torch path cannot
drift from the advanced composition API (`apply_model`, `get_model`) callers
still import directly.

Reads: ..apply (apply_model), ..audio (convert_audio), .base (SeparationBackend), torch
"""

from __future__ import annotations

from typing import Callable, Optional

import torch as th

from ..apply import apply_model
from ..audio import convert_audio


class TorchBackend:
    """Wraps an already-loaded Demucs model (or `BagOfModels`) as a `SeparationBackend`."""

    name = "torch"

    def __init__(self, model, device: str):
        self._model = model
        self._device = device

    @classmethod
    def is_available(cls) -> bool:
        return True

    @property
    def resolved_device(self) -> str:
        return str(self._device)

    @property
    def sources(self):
        return self._model.sources

    @property
    def model(self):
        """The resident model, for callers composing the advanced path directly."""
        return self._model

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
        samplerate = self._model.samplerate
        audio_channels = self._model.audio_channels
        if sr is not None and sr != samplerate:
            wav = convert_audio(wav, sr, samplerate, audio_channels)
        ref = wav.mean(0)
        wav = wav - ref.mean()
        wav = wav / (ref.std() + 1e-8)
        out = apply_model(
            self._model,
            wav[None],
            segment=segment,
            shifts=shifts,
            split=split,
            overlap=overlap,
            device=self._device,
            num_workers=jobs,
            callback=callback,
            callback_arg=callback_arg,
            progress=progress,
        )
        if out is None:
            raise KeyboardInterrupt
        out = out * (ref.std() + 1e-8)
        out = out + ref.mean()
        wav = wav * (ref.std() + 1e-8)
        wav = wav + ref.mean()
        return wav, out

    def release(self) -> None:
        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        if th.cuda.is_available():
            th.cuda.empty_cache()
        mps = getattr(th.backends, "mps", None)
        if mps is not None and mps.is_available():
            th.mps.empty_cache()
