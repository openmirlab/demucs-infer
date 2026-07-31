"""Backend registry -- the only name callers use to pick a compute path.

Resolution is by name and nothing more: asking for a backend that cannot run
here (or cannot run the given model) raises rather than quietly substituting
another -- a silent substitution is discovered only by noticing the wrong
hardware was busy. Backend modules are imported lazily so `import
demucs_infer` never drags an optional framework into the default import path.

Both backends are built from an *already-loaded* Torch model (see
`torch_backend.TorchBackend` and `mlx_backend.MLXBackend.from_torch_model`):
`Separator._load_model()` keeps sole ownership of checkpoint resolution,
sha256 verification, format dispatch, and `BagOfModels` assembly (`api.py`,
`checkpoint_runtime.py`), and the MLX path converts that resident, already-
verified model object rather than re-deriving any of it from a checkpoint
path -- no second catalog, no converted-weight cache this module controls.

Reads: .base (SeparationBackend, BackendUnavailable), .torch_backend
(lazily), .mlx_backend (lazily)
"""

from __future__ import annotations

from typing import Optional

from .base import BackendUnavailable, SeparationBackend

#: Every selectable backend name, in the order `auto` prefers them.
BACKEND_NAMES = ("mlx", "torch")
DEFAULT_BACKEND = "torch"


def _load(name: str):
    """Import a backend module on demand. Importing must not require its framework."""
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend
    raise BackendUnavailable(f"no backend registered under {name!r}")


def resolve_backend_name(requested: Optional[str], *, model=None) -> str:
    """Resolve a requested backend name, honouring it exactly or raising.

    `None` and `"torch"` both mean the shipped Torch path -- the default never
    moves on its own. `"auto"` prefers an accelerated backend when one is
    genuinely importable *and* can run this model, falling back to Torch
    otherwise; that is the one place a fallback is what the caller asked for.

    `model` is the already-loaded Torch model (or `BagOfModels`), when known.
    Passing it matters: without it, `auto` can settle on a backend that then
    fails at conversion (a bag, an unported architecture, a Wiener-filtering
    config), where Torch would simply have worked. An *explicit* backend still
    raises for a model it cannot run -- that request is honoured or refused,
    never downgraded.
    """
    if requested is None or requested == DEFAULT_BACKEND:
        return DEFAULT_BACKEND
    if requested == "auto":
        for name in BACKEND_NAMES:
            try:
                backend = _load(name)
            except BackendUnavailable:
                continue
            if backend.is_available() and _supports(backend, model):
                return name
        return DEFAULT_BACKEND
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"backend must be None, 'auto', or one of {BACKEND_NAMES}; got {requested!r}"
        )
    backend = _load(requested)
    if not backend.is_available():
        raise BackendUnavailable(
            f"backend {requested!r} is unavailable on this machine; "
            f"it may need an optional extra (pip install 'demucs-infer[{requested}]')"
        )
    if model is not None and not _supports(backend, model):
        backend.assert_supports_model(model)
    return requested


def _supports(backend, model) -> bool:
    """Whether `backend` can run `model`."""
    check = getattr(backend, "supports_model", None)
    return True if check is None or model is None else bool(check(model))


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
