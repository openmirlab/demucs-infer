"""Thin barrel for the vendored MLX HDemucs/HTDemucs models -- lazy on purpose.

`demucs_infer.mlx` is only ever imported on demand by the (caller-owned) MLX
compute backend (`backends/mlx_backend.py`), never by the package's default
import path, so `import demucs_infer` stays MLX-free even with this
subpackage present. Attribute access is deferred via `__getattr__` so merely
importing this package doesn't eagerly import `mlx.core`/`mlx.nn` -- the cost
of that import is paid only when a name is actually used.

Reads: .hdemucs (HDemucsMLX, lazily), .htdemucs (HTDemucsMLX, lazily),
.convert (convert_state_dict, load_converted_weights, lazily)
"""

from __future__ import annotations

__all__ = [
    "HDemucsMLX",
    "HTDemucsMLX",
    "convert_state_dict",
    "load_converted_weights",
]


def __getattr__(name: str):
    if name == "HDemucsMLX":
        from .hdemucs import HDemucsMLX

        return HDemucsMLX
    if name == "HTDemucsMLX":
        from .htdemucs import HTDemucsMLX

        return HTDemucsMLX
    if name in ("convert_state_dict", "load_converted_weights"):
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
