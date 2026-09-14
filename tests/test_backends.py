"""Backend seam contract: resolution and import purity.

These are offline and hardware-independent. They guard the properties the
seam exists to protect: a requested backend is honoured or refused, never
silently swapped; the default import path stays free of any optional
compute framework. Only `torch` is registered now that the optional MLX
backend has been removed (2026-09-14, never released -- see CHANGELOG's
`[Unreleased]` "Removed" entry).
"""

import subprocess
import sys

import pytest

from demucs_infer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    get_backend,
    resolve_backend_name,
)


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("onnx")


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_backend_names_only_lists_torch():
    assert BACKEND_NAMES == ("torch",)


# -------------------------------------------------------------------- purity


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install demucs-infer` must stay import-clean of any optional
    compute framework. Run in a fresh subprocess for isolation."""
    code = (
        "import sys\n"
        "import demucs_infer\n"
        "optional = {'mlx', 'mlx_spectro'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import demucs_infer pulled in optional frameworks: {leaked}"


def test_importing_backends_package_does_not_pull_in_mlx():
    """Even importing the seam itself must not import mlx. Subprocess-isolated
    for the same reason as the test above."""
    code = (
        "import sys\n"
        "import demucs_infer.backends\n"
        "optional = {'mlx', 'mlx_spectro'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import demucs_infer.backends pulled in: {leaked}"
