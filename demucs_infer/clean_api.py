"""Task-level source-separation facade.

This module is a small front door over :class:`demucs_infer.api.Separator`.
It owns input-boundary validation and helper lifecycle, while the existing
separator continues to own model loading, audio decoding, and separation.
The module deliberately avoids importing torch (or the heavier API module)
until a separation call is made.
"""

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from threading import RLock
from urllib.request import urlopen

from .checkpoint_catalog import checkpoint_catalog, get_checkpoint_metadata


_IO_CHUNK_SIZE = 1024 * 1024


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

    def _verify(self, path, expected):
        digest = sha256()
        with path.open("rb") as checkpoint:
            while chunk := checkpoint.read(_IO_CHUNK_SIZE):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")

    def _cache_root(self, *, create=True):
        root = self.cache_dir or (Path.home() / ".cache" / "demucs-infer")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _download_checkpoint(self, url, path, expected):
        if path.exists():
            try:
                self._verify(path, expected)
                return path
            except ValueError:
                path.unlink()

        temporary_path = None
        try:
            with urlopen(url) as source:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as target:
                    temporary_path = Path(target.name)
                    while chunk := source.read(_IO_CHUNK_SIZE):
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
            self._verify(temporary_path, expected)
            os.replace(temporary_path, path)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return path

    def _materialize_named_checkpoint(self, metadata):
        path = self._cache_root() / metadata["path"]
        return self._download_checkpoint(metadata["url"], path, metadata["sha256"])

    def _materialize_checkpoint(self):
        path = self.checkpoint_path
        if path is None and self.checkpoint_url:
            if not self.checkpoint_sha256:
                raise ValueError("checkpoint_sha256 is required with checkpoint_url")
            root = self._cache_root()
            path = root / Path(self.checkpoint_url).name
            self._download_checkpoint(self.checkpoint_url, path, self.checkpoint_sha256)
            return path, path.stem.split("-", 1)[0]
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        signature = path.stem.split("-", 1)[0]
        expected = self.checkpoint_sha256 or (get_checkpoint_metadata(signature) or {}).get("sha256")
        if expected:
            self._verify(path, expected)
        return path, signature

    def _materialize_named_model_repo(self):
        from .pretrained import REMOTE_ROOT
        import yaml

        root = self._cache_root()
        catalog = checkpoint_catalog()
        bag_path = REMOTE_ROOT / f"{self.model_name}.yaml"

        if bag_path.is_file():
            bag = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
            signatures = bag.get("models", [])
            if signatures and all(signature in catalog for signature in signatures):
                for signature in signatures:
                    self._materialize_named_checkpoint(catalog[signature])
                (root / bag_path.name).write_text(
                    bag_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                return root, self.model_name

        metadata = get_checkpoint_metadata(self.model_name)
        if metadata is None:
            return None
        self._materialize_named_checkpoint(metadata)
        return root, metadata["signature"]

    def _resolved_named_checkpoints(self):
        """Return the local artifacts the named-model loader would use.

        This is deliberately the read-only counterpart of
        :meth:`_materialize_named_model_repo`: it reads the packaged bag
        recipe but never downloads a checkpoint or copies that recipe.
        """
        from .pretrained import REMOTE_ROOT
        import yaml

        catalog = checkpoint_catalog()
        bag_path = REMOTE_ROOT / f"{self.model_name}.yaml"
        if bag_path.is_file():
            bag = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
            signatures = bag.get("models", [])
            if signatures and all(signature in catalog for signature in signatures):
                return [catalog[signature] for signature in signatures]
        metadata = get_checkpoint_metadata(self.model_name)
        return [metadata] if metadata is not None else []

    def _get_separator(self):
        if self._status == "closed":
            raise RuntimeError("cannot load a closed DemucsSession")
        if self._separator is None:
            self._status = "loading"
            try:
                from .api import Separator
                override = self._materialize_checkpoint()
                if override:
                    path, signature = override
                    from .states import load_model
                    separator = Separator.__new__(Separator)
                    separator._name = signature
                    separator._repo = path.parent
                    separator._model = load_model(path)
                    separator._audio_channels = separator._model.audio_channels
                    separator._samplerate = separator._model.samplerate
                    separator.update_parameter(**self.separator_options)
                    self._separator = separator
                else:
                    pinned = self._materialize_named_model_repo()
                    if pinned is not None:
                        repo_root, resolved_name = pinned
                        self._separator = Separator(model=resolved_name,
                                                    repo=repo_root,
                                                    **self.separator_options)
                    else:
                        self._separator = Separator(model=self.model_name,
                                                    **self.separator_options)
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
        metadata = get_checkpoint_metadata(self.model_name)
        if self.checkpoint_path is not None:
            paths = [self.checkpoint_path]
        elif self.checkpoint_url:
            paths = [self._cache_root(create=False) / Path(self.checkpoint_url).name]
        else:
            entries = self._resolved_named_checkpoints()
            paths = [self._cache_root(create=False) / entry["path"] for entry in entries]
        path = paths[0] if paths else None
        return {"model": self.model_name,
                "checkpoint_path": str(path) if path is not None else None,
                "checkpoint_paths": [str(item) for item in paths],
                "checkpoint_url": self.checkpoint_url or (metadata or {}).get("url"),
                "sha256": self.checkpoint_sha256 or (metadata or {}).get("sha256"),
                "cached": bool(paths) and all(item.is_file() for item in paths),
                "loaded": self._separator is not None, "status": self.status}

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
