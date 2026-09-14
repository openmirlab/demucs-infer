"""Backend registry -- the only name callers use to pick a compute path.

Resolution is by name and nothing more: asking for a backend that cannot run
here raises rather than quietly substituting another. Only `torch` is
registered now that the optional MLX backend has been removed (2026-09-14 --
MLX/MPS were never released, see CHANGELOG's `[Unreleased]` "Removed"
entry); this module stays as a seam rather than being inlined away
immediately, so a future accelerated backend has one place to register.

Reads: .base (SeparationBackend, BackendUnavailable), .torch_backend
(lazily)
"""

from __future__ import annotations

from typing import Optional

from .base import BackendUnavailable, SeparationBackend

#: Every selectable backend name.
BACKEND_NAMES = ("torch",)
DEFAULT_BACKEND = "torch"


def _load(name: str):
    """Import a backend module on demand. Importing must not require its framework."""
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend
    raise BackendUnavailable(f"no backend registered under {name!r}")


def resolve_backend_name(requested: Optional[str], *, model=None) -> str:
    """Resolve a requested backend name, honouring it exactly or raising.

    `None` and `"torch"` both mean the shipped Torch path -- the only
    registered backend, so this never has anything to fall back from.
    """
    if requested is None or requested == DEFAULT_BACKEND:
        return DEFAULT_BACKEND
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"backend must be None or one of {BACKEND_NAMES}; got {requested!r}"
        )
    backend = _load(requested)
    if not backend.is_available():
        raise BackendUnavailable(f"backend {requested!r} is unavailable on this machine")
    return requested


def get_backend(name: str):
    """Return the backend class registered under `name`."""
    return _load(name)


__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_BACKEND",
    "BackendUnavailable",
    "SeparationBackend",
    "get_backend",
    "resolve_backend_name",
]
