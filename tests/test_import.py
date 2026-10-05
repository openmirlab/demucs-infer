"""Smoke test: public modules import with the declared inference dependencies.

TorchAudio and TorchCodec are outside this package's install and import path.
"""
import importlib
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib

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


def test_public_audio_api_imports_when_torchaudio_is_unavailable():
    """A fresh package import must not transitively require torchaudio."""
    code = """
import importlib.abc
import sys

class BlockTorchAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torchaudio' or fullname.startswith('torchaudio.'):
            raise RuntimeError('torchaudio imported')

sys.meta_path.insert(0, BlockTorchAudio())
from demucs_infer import DemucsSession
from demucs_infer.api import Separator
from demucs_infer.audio import AudioFile, save_audio
assert callable(DemucsSession) and callable(Separator)
assert callable(AudioFile) and callable(save_audio)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_package_metadata_does_not_pull_torchaudio_or_torchcodec():
    with (Path(__file__).resolve().parent.parent / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    assert all(not requirement.lower().startswith(("torchaudio", "torchcodec"))
               for requirement in requirements)
