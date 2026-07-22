"""Offline lifecycle and cache-resolver contract tests for DemucsSession."""

import pytest

from demucs_infer.clean_api import DemucsSession
from demucs_infer.checkpoint_catalog import get_checkpoint_metadata


class _Model:
    sources = ("drums", "bass", "other", "vocals")

    def __init__(self):
        self.released = False

    def cpu(self):
        self.released = True
        return self


class _Separator:
    samplerate = 44100

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = _Model()

    def separate_audio_file(self, path):
        return "mixture", {"drums": path}


def test_session_load_is_idempotent_release_reloads_and_close_is_terminal(monkeypatch):
    from demucs_infer.checkpoint_runtime import CheckpointRuntime

    built = []
    def build(runtime, kwargs):
        separator = _Separator(**kwargs)
        built.append(separator)
        return separator

    monkeypatch.setattr(CheckpointRuntime, "load_separator", build)
    session = DemucsSession(device="cuda:1")
    with pytest.raises(RuntimeError, match="ready"):
        session.infer("song.wav")
    session.load()
    session.load()
    assert len(built) == 1
    assert built[0].kwargs["device"] == "cuda:1"
    session.release()
    assert built[0].model.released
    session.load()
    assert len(built) == 2
    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="closed"):
        session("song.wav")


def test_cache_info_uses_non_downloading_default_and_custom_resolvers(monkeypatch, tmp_path):
    import demucs_infer.checkpoint_runtime as checkpoint_runtime

    monkeypatch.setattr(checkpoint_runtime, "urlopen", lambda *args, **kwargs: pytest.fail("download attempted"))
    metadata = get_checkpoint_metadata("htdemucs")
    cache_root = tmp_path / "empty-cache"
    session = DemucsSession(model="htdemucs", cache_dir=cache_root)
    info = session.cache_info()
    expected = cache_root / metadata["path"]
    assert info["checkpoint_path"] == str(expected)
    assert info["cached"] is False
    assert not cache_root.exists()
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"present")
    assert session.cache_info()["cached"] is True

    custom = tmp_path / "private.th"
    custom.write_bytes(b"present")
    custom_info = DemucsSession(checkpoint_path=custom).cache_info()
    assert custom_info["checkpoint_path"] == str(custom)
    assert custom_info["sha256"] == metadata["sha256"]
    assert custom_info["cached"] is True

    custom_url = "https://example.invalid/checkpoints/custom.th"
    url_info = DemucsSession(
        checkpoint_url=custom_url,
        checkpoint_sha256="0" * 64,
        cache_dir=cache_root,
    ).cache_info()
    assert url_info["checkpoint_path"] == str(cache_root / "custom.th")
    assert url_info["checkpoint_url"] == custom_url
    assert url_info["cached"] is False


def test_session_delegates_loading_and_cache_reporting_to_checkpoint_runtime(monkeypatch):
    import demucs_infer.clean_api as clean_api

    calls = []
    separator = _Separator()

    class Runtime:
        def __init__(self, model_name, **options):
            calls.append(("init", model_name, options))

        def load_separator(self, separator_options):
            calls.append(("load", separator_options))
            return separator

        def cache_info(self, *, loaded, status):
            calls.append(("cache_info", loaded, status))
            return {"loaded": loaded, "status": status}

    monkeypatch.setattr(clean_api, "CheckpointRuntime", Runtime)
    session = DemucsSession(model="custom", cache_dir="cache", device="cpu")

    assert session.cache_info() == {"loaded": False, "status": "new"}
    session.load()
    assert session.cache_info() == {"loaded": True, "status": "ready"}
    assert calls == [
        ("init", "custom", {
            "checkpoint_path": None,
            "checkpoint_url": None,
            "checkpoint_sha256": None,
            "cache_dir": session.cache_dir,
        }),
        ("cache_info", False, "new"),
        ("init", "custom", {
            "checkpoint_path": None,
            "checkpoint_url": None,
            "checkpoint_sha256": None,
            "cache_dir": session.cache_dir,
        }),
        ("load", {"device": "cpu"}),
        ("init", "custom", {
            "checkpoint_path": None,
            "checkpoint_url": None,
            "checkpoint_sha256": None,
            "cache_dir": session.cache_dir,
        }),
        ("cache_info", True, "ready"),
    ]
