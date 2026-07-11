"""Smoke test: every public-facing module must import cleanly with only
this package's declared dependencies installed (torch, torchaudio, einops,
julius, numpy, pyyaml, tqdm -- no openunmix, see ADOPT P1)."""
import importlib

import pytest

PUBLIC_MODULES = [
    "demucs_infer",
    "demucs_infer.api",
    "demucs_infer.apply",
    "demucs_infer.audio",
    "demucs_infer.community",
    "demucs_infer.compat",
    "demucs_infer.demucs",
    "demucs_infer.hdemucs",
    "demucs_infer.htdemucs",
    "demucs_infer.log",
    "demucs_infer.model_info",
    "demucs_infer.pretrained",
    "demucs_infer.repo",
    "demucs_infer.separate",
    "demucs_infer.spec",
    "demucs_infer.states",
    "demucs_infer.transformer",
    "demucs_infer.utils",
    "demucs_infer.wdemucs",
    "demucs_infer.wiener",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_openunmix_not_installed_or_not_required():
    """Vendoring (P1) means openunmix must not be required -- if it happens
    to be installed anyway (e.g. a dev's local venv), that's fine, but every
    module above must already have imported successfully without it being
    a declared dependency."""
    import demucs_infer.wiener as vendored
    assert hasattr(vendored, "wiener")


def test_version_is_single_sourced():
    import demucs_infer
    assert isinstance(demucs_infer.__version__, str)
    assert demucs_infer.__version__
