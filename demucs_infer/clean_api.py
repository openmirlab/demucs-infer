"""Task-level source-separation facade and session lifecycle.

This module is a small front door over :class:`demucs_infer.api.Separator`.
It owns input-boundary validation and helper lifecycle, while
``checkpoint_runtime`` owns checkpoint resolution/loading and the existing
separator owns audio decoding and separation. The module deliberately avoids
importing torch (or the heavier API module) until a separation call is made.

Reads: checkpoint_runtime, api.Separator through the internal runtime.
"""

from pathlib import Path
from threading import RLock

from .checkpoint_runtime import CheckpointRuntime


class DemucsSeparator:
    """Reusable, lazily initialized Demucs source-separation helper.

    The underlying ``Separator`` instance (and therefore its loaded model) is
    cached for this helper's lifetime. A helper is serialized by a small lock
    so concurrent calls do not race model initialization or mutable separator
    state. One-shot functions create a fresh helper and do not share caches.
    """

    def __init__(self, model="htdemucs", *, checkpoint_path=None,
                 checkpoint_url=None, checkpoint_sha256=None, cache_dir=None,
                 **separator_options):
        self.model_name = model
        self.separator_options = dict(separator_options)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._separator = None
        self._call_lock = RLock()
        self._status = "new"

    @property
    def status(self):
        return self._status

    @property
    def samplerate(self):
        if self._separator is None:
            raise RuntimeError("session must be loaded before samplerate is available")
        return self._separator.samplerate

    @property
    def sources(self):
        if self._separator is None:
            raise RuntimeError("session must be loaded before sources are available")
        return tuple(self._separator.model.sources)

    def _checkpoint_runtime(self):
        return CheckpointRuntime(
            self.model_name,
            checkpoint_path=self.checkpoint_path,
            checkpoint_url=self.checkpoint_url,
            checkpoint_sha256=self.checkpoint_sha256,
            cache_dir=self.cache_dir,
        )

    def _materialize_checkpoint(self):
        return self._checkpoint_runtime().materialize_override()

    def _get_separator(self):
        if self._status == "closed":
            raise RuntimeError("cannot load a closed DemucsSession")
        if self._separator is None:
            self._status = "loading"
            try:
                self._separator = self._checkpoint_runtime().load_separator(self.separator_options)
                self._status = "ready"
            except Exception:
                self._status = "failed"
                raise
        return self._separator

    def load(self):
        with self._call_lock:
            self._get_separator()
        return self

    def infer(self, audio, *, sample_rate=None):
        """Run inference on an explicitly loaded session.

        ``infer`` is the strict lifecycle entry point: callers must invoke
        :meth:`load` (or enter the context manager) first.  The legacy
        callable surface, :meth:`__call__`, intentionally keeps its lazy
        initialization behavior for backward compatibility.
        """
        with self._call_lock:
            if self.status != "ready" or self._separator is None:
                raise RuntimeError(
                    "session must be loaded and ready before infer(); "
                    f"current status is {self.status!r}"
                )
            return self._infer_loaded(audio, sample_rate=sample_rate)

    def _infer_loaded(self, audio, *, sample_rate=None):
        """Dispatch inference assuming the session is already ready."""
        separator = self._separator
        if isinstance(audio, (str, Path)):
            if sample_rate is not None:
                raise ValueError("sample_rate is not accepted for path inputs")
            return separator.separate_audio_file(Path(audio))
        if sample_rate is None:
            raise ValueError("sample_rate is required for tensor audio")
        return separator.separate_tensor(audio, sr=sample_rate)

    def release(self):
        with self._call_lock:
            if self._status == "closed":
                return self
            if self._separator is not None:
                model = getattr(self._separator, "model", None)
                if model is not None and hasattr(model, "cpu"):
                    model.cpu()
                self._separator = None
            self._status = "released"
        return self

    def close(self):
        with self._call_lock:
            if self._status != "closed":
                self.release()
                self._status = "closed"
        return self

    def cache_info(self):
        return self._checkpoint_runtime().cache_info(
            loaded=self._separator is not None,
            status=self.status,
        )

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __call__(self, audio, *, sample_rate=None):
        """Separate a path or tensor, returning ``(mixture, stems)``.

        Paths are decoded by the existing ``Separator`` audio loader. Tensor
        inputs must provide their source sample rate explicitly; resampling,
        channel conversion, and normalization remain delegated to
        ``Separator.separate_tensor``.
        """
        with self._call_lock:
            self._get_separator()
            return self._infer_loaded(audio, sample_rate=sample_rate)

    def separate(self, audio, *, sample_rate=None):
        """Named alias for :meth:`__call__`."""
        return self(audio, sample_rate=sample_rate)


def separate(audio, *, model="htdemucs", sample_rate=None, **separator_options):
    """One-shot source separation with a fresh model helper.

    ``separator_options`` are the existing ``Separator`` options (for
    example ``device``, ``shifts``, ``overlap``, and ``split``). No model or
    processor state is retained after this call.
    """
    return DemucsSeparator(model=model, **separator_options)(
        audio, sample_rate=sample_rate
    )


def separate_file(path, *, model="htdemucs", **separator_options):
    """One-shot separation for an audio file path."""
    return separate(path, model=model, **separator_options)


def separate_tensor(wav, *, sample_rate, model="htdemucs", **separator_options):
    """One-shot separation for a tensor and its explicit source sample rate."""
    return separate(wav, model=model, sample_rate=sample_rate,
                    **separator_options)


DemucsSession = DemucsSeparator

__all__ = ["DemucsSeparator", "DemucsSession", "separate", "separate_file", "separate_tensor"]
