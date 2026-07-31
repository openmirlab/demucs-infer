"""The backend seam -- one narrow protocol every compute backend implements.

A backend owns everything framework-specific: model construction, checkpoint
weights, tensor layout, device placement, and the whole conflated
resample + reference-normalize + chunked-apply that ``Separator.separate_tensor()``
has always done in one method. That conflation is deliberate and mirrors it,
rather than the narrower "one whole mixture, already normalized" seam a sibling
OpenMIRLab package (``bs-roformer-infer``) uses: this package's own
``separate_tensor()`` -- the method this whole seam exists to keep
behavior-identical to -- already resamples and reference-normalizes inline
before calling ``apply.apply_model()``, exactly the shape the sibling
``mdxnet-infer`` package's seam was built to mirror (see its
``backends/base.py`` module docstring for the same reasoning). Duplicating a
narrower "already-normalized mixture in" seam on top would just be a second
shape convention for the same concept.

Chunking (the shift trick + segment/overlap split) accumulates on-device for
speed, so the seam sits above it: each backend owns its whole chunked
inference, mirroring ``apply.apply_model()``'s own recursive shift/split
structure rather than forking a per-chunk seam that would drag every
accumulator back to the host.

Above the seam nothing knows a tensor layout, a dtype, or which chip is busy:
folder iteration, stem naming, file writing, and the CLI stay in
separate.py/api.py exactly as today.

Reads: torch (boundary tensor type only)
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import torch as th


@runtime_checkable
class SeparationBackend(Protocol):
    """Turns one loaded waveform into stems, hiding how and where it computed them."""

    #: Stable identifier, matching the `backend=` argument that selects it.
    name: str

    @classmethod
    def is_available(cls) -> bool:
        """True when this backend can actually run on this machine right now."""

    @property
    def resolved_device(self) -> str:
        """The concrete target chosen, after any `auto`/`None` sentinel was resolved."""

    @property
    def sources(self):
        """Source names, mirroring the wrapped model's `.sources`."""

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
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Resample + reference-normalize + chunked-apply, mirroring
        ``Separator.separate_tensor()``'s own contract exactly.

        Returns `(wav, out)`: `wav` is the (possibly resampled) input at the
        model's own sample rate, and `out` is a `(1, sources, channels,
        length)` tensor -- the same shape `apply_model()` returns -- so
        `Separator.separate_tensor()` can zip it against `self._model.sources`
        unchanged regardless of which backend ran.
        """

    def release(self) -> None:
        """Drop resident model and device memory. Disk checkpoints stay."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend is requested by name but cannot run here, or is
    asked to run a model it cannot honour (a bag of models, an unported
    architecture, a Wiener-filtering configuration, etc.).

    Always raised, never swallowed into a fallback: silently substituting a
    different backend -- or silently running one sub-model of a bag -- discards
    what the caller explicitly asked for, and would only be discovered by
    noticing the wrong hardware was busy or the wrong numbers came out.
    """
