"""Backend seam contract: resolution, import purity, and measured MLX support.

These are offline and hardware-independent. They guard the properties the
seam exists to protect: a requested backend is honoured or refused, never
silently swapped or downgraded; the default import path stays free of the
optional `mlx`/`mlx_spectro` frameworks; and MLX support is *measured* from
a model's own class/attributes (never a name-based allowlist), so a
`BagOfModels` of 2+ sub-models, an unported architecture, or a
Wiener-filtering configuration is refused explicitly rather than silently
mis-run.
"""

import subprocess
import sys

import pytest
import torch as th

from demucs_infer.apply import BagOfModels
from demucs_infer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    get_backend,
    resolve_backend_name,
)
from demucs_infer.backends.mlx_backend import supports_model


def _fake_model(class_name, *, sources=("a", "b"), samplerate=44100,
                 audio_channels=2, segment=10.0, cac=True, wiener_iters=0):
    """A minimal `nn.Module` shaped like `HDemucs`/`HTDemucs` enough to drive
    `supports_model`'s checks (class name + `cac`/`wiener_iters`) without
    constructing a real, full-sized model or needing a checkpoint."""

    def _init(self):
        th.nn.Module.__init__(self)
        self.sources = list(sources)
        self.samplerate = samplerate
        self.audio_channels = audio_channels
        self.segment = segment
        self.cac = cac
        self.wiener_iters = wiener_iters

    return type(class_name, (th.nn.Module,), {"__init__": _init})()


# --------------------------------------------------------------------- resolve


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_resolves_to_a_registered_backend():
    """`auto` is the one place a fallback is what the caller asked for."""
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("onnx")


def test_unavailable_backend_raises_rather_than_substituting(monkeypatch):
    """An explicit request is honoured or fails loudly -- never downgraded."""
    from demucs_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_auto_falls_back_to_torch_when_mlx_is_unavailable(monkeypatch):
    from demucs_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    assert resolve_backend_name("auto") == "torch"


def test_auto_prefers_mlx_when_available_and_model_supported(monkeypatch):
    from demucs_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    model = _fake_model("HTDemucs")
    assert resolve_backend_name("auto", model=model) == "mlx"


def test_auto_skips_mlx_for_an_unsupported_model_even_if_available(monkeypatch):
    """Without the model, `auto` could pick a backend that then fails at
    conversion; passing it lets `auto` fall back the way an explicit
    `backend='torch'` request never would (see `resolve_backend_name`'s
    docstring)."""
    from demucs_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    bag = BagOfModels([_fake_model("HTDemucs"), _fake_model("HTDemucs")])
    assert resolve_backend_name("auto", model=bag) == "torch"


def test_explicit_mlx_raises_for_unsupported_model(monkeypatch):
    from demucs_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    bag = BagOfModels([_fake_model("HTDemucs"), _fake_model("HTDemucs")])
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx", model=bag)


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_backend_declares_the_protocol_surface():
    backend = get_backend("mlx")
    assert backend.name == "mlx"
    for method in ("separate", "release", "from_torch_model", "is_available", "supports_model"):
        assert hasattr(backend, method), f"MLXBackend is missing {method}"


# ------------------------------------------------------------- measured support


def test_supports_bare_htdemucs_and_hdemucs():
    assert supports_model(_fake_model("HTDemucs")) is True
    assert supports_model(_fake_model("HDemucs")) is True


def test_refuses_unported_architecture():
    """Only HDemucs/HTDemucs are ported; the original time-only `Demucs`
    (used inside some *multi*-model bags, never as a single registry entry)
    has no MLX module -- measured from the class name, not declared."""
    assert supports_model(_fake_model("Demucs")) is False


def test_refuses_wiener_configuration():
    assert supports_model(_fake_model("HDemucs", cac=False)) is False
    assert supports_model(_fake_model("HDemucs", wiener_iters=2)) is False


def test_refuses_bag_of_two_or_more_models():
    bag = BagOfModels([_fake_model("HTDemucs"), _fake_model("HTDemucs")])
    assert supports_model(bag) is False


def test_unwraps_a_single_model_bag_and_measures_its_submodel():
    """`checkpoint_runtime.load_registered_model()` wraps *every*
    `demucs_package`-format recipe in a `BagOfModels`, including this
    package's 7 single-component registry entries -- a length-1 bag must not
    be refused on sight, or every one of those checkpoints would be
    unusable with backend='mlx'."""
    bag = BagOfModels([_fake_model("HTDemucs")])
    assert supports_model(bag) is True

    unsupported_bag = BagOfModels([_fake_model("Demucs")])
    assert supports_model(unsupported_bag) is False


def test_mlx_backend_assert_supports_model_names_the_reason():
    bag = BagOfModels([_fake_model("HTDemucs"), _fake_model("HTDemucs")])
    with pytest.raises(BackendUnavailable, match="2\\+ sub-models"):
        get_backend("mlx").assert_supports_model(bag)

    with pytest.raises(BackendUnavailable, match="Demucs"):
        get_backend("mlx").assert_supports_model(_fake_model("Demucs"))

    with pytest.raises(BackendUnavailable, match="Wiener"):
        get_backend("mlx").assert_supports_model(_fake_model("HDemucs", cac=False))


# -------------------------------------------------------------------- purity


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install demucs-infer` must stay MLX-free and import-clean.

    Run in a fresh subprocess rather than this test process: on a machine
    that actually has the ``[mlx]`` extra installed, an *earlier* test in
    this same file legitimately imports real ``mlx`` as a side effect of
    calling the real (unmonkeypatched) ``MLXBackend.is_available()`` -- that
    is correct behaviour for `resolve_backend_name("auto")`, not a leak, and
    asserting against this process's already-polluted `sys.modules` would
    make the test's pass/fail depend on execution order instead of on what
    `import demucs_infer` itself does.
    """
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
    """Even importing the seam itself must not import mlx -- only requesting
    the mlx backend by name should. Subprocess-isolated for the same reason
    as the test above."""
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
