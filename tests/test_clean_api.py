"""Focused contract tests for the additive Demucs task facade."""

from pathlib import Path

import pytest

from demucs_infer.clean_api import DemucsSeparator, DemucsSession, separate
from demucs_infer.checkpoint_catalog import CHECKPOINT_CATALOG, get_checkpoint_metadata


class _FakeSeparator:
    instances = []

    def __init__(self, model="htdemucs", **options):
        self.model = model
        self.options = options
        self.calls = []
        self.__class__.instances.append(self)

    def separate_audio_file(self, path):
        self.calls.append(("file", path))
        return ("mixture", {"vocals": "stem"})

    def separate_tensor(self, wav, sr=None):
        self.calls.append(("tensor", wav, sr))
        return (wav, {"vocals": "stem"})


def test_existing_separator_import_and_facade_delegation(monkeypatch, tmp_path):
    import demucs_infer.api as api

    monkeypatch.setattr(api, "Separator", _FakeSeparator)
    _FakeSeparator.instances.clear()
    path = tmp_path / "song.wav"
    path.touch()

    helper = DemucsSeparator(model="mdx", device="cpu")
    assert helper(path) == ("mixture", {"vocals": "stem"})
    assert helper("song.wav") == ("mixture", {"vocals": "stem"})
    assert len(_FakeSeparator.instances) == 1
    assert _FakeSeparator.instances[0].calls[0][0] == "file"


def test_tensor_requires_explicit_sample_rate(monkeypatch):
    import demucs_infer.api as api

    monkeypatch.setattr(api, "Separator", _FakeSeparator)
    helper = DemucsSeparator()
    with pytest.raises(ValueError, match="sample_rate is required"):
        helper(object())


def test_tensor_delegates_sample_rate_and_one_shot_is_fresh(monkeypatch):
    import demucs_infer.api as api

    monkeypatch.setattr(api, "Separator", _FakeSeparator)
    _FakeSeparator.instances.clear()
    wav = object()
    assert separate(wav, sample_rate=22050)[1] == {"vocals": "stem"}
    assert separate(wav, sample_rate=22050)[1] == {"vocals": "stem"}
    assert len(_FakeSeparator.instances) == 2
    assert _FakeSeparator.instances[0].calls[0] == ("tensor", wav, 22050)


def test_advanced_separator_surface_remains_importable():
    from demucs_infer.api import Separator

    assert callable(Separator.separate_tensor)


def test_session_lifecycle_and_context_manager(monkeypatch):
    import demucs_infer.api as api
    monkeypatch.setattr(api, "Separator", _FakeSeparator)
    session = DemucsSession(device="cpu")
    assert session.status == "new"
    with pytest.raises(RuntimeError, match="must be loaded and ready"):
        session.infer("song.wav")
    assert session.load() is session
    assert session.status == "ready"
    assert session.infer("song.wav")[0] == "mixture"
    assert session.cache_info()["loaded"] is True
    session.release()
    assert session.status == "released"
    assert session.cache_info()["loaded"] is False
    with pytest.raises(RuntimeError, match="must be loaded and ready"):
        session.infer("song.wav")
    with DemucsSeparator() as scoped:
        assert scoped.status == "ready"
    assert scoped.status == "released"


def test_session_infer_rejects_failed_state(monkeypatch):
    import demucs_infer.api as api

    def fail_load(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "Separator", fail_load)
    session = DemucsSession()
    with pytest.raises(RuntimeError, match="boom"):
        session.load()
    assert session.status == "failed"
    with pytest.raises(RuntimeError, match="must be loaded and ready"):
        session.infer("song.wav")


def test_checkpoint_catalog_is_pinned_and_copyable():
    metadata = get_checkpoint_metadata("htdemucs")
    assert metadata["url"].startswith("https://")
    assert len(metadata["sha256"]) == 64
    metadata["url"] = "override"
    assert CHECKPOINT_CATALOG["955717e8"]["url"] != "override"


def test_checkpoint_url_requires_pinned_hash(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        DemucsSeparator(checkpoint_url="https://example.invalid/model.th").load()


def test_default_session_uses_package_checkpoint_url_and_hash_loader(monkeypatch):
    import demucs_infer.api as api
    import demucs_infer.repo as repo_module

    calls = []

    def fake_download(url, *, map_location, check_hash):
        calls.append((url, map_location, check_hash))
        return {"fake": "weights"}

    class _FakeModel:
        audio_channels = 2
        samplerate = 44100

    monkeypatch.setattr(repo_module.torch.hub, "load_state_dict_from_url", fake_download)
    monkeypatch.setattr(repo_module, "load_model", lambda payload: _FakeModel())

    class _LoaderProbe:
        def __init__(self, model="htdemucs", repo=None, **options):
            self.model = model
            self.repo = repo
            self.options = options
            self.model_obj = repo.get_model(model)

    monkeypatch.setattr(api, "Separator", _LoaderProbe)
    session = DemucsSession(device="cpu").load()
    metadata = get_checkpoint_metadata("htdemucs")
    assert session._separator.model == metadata["signature"]
    assert calls == [(metadata["url"], "cpu", True)]
