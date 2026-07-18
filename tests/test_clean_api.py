"""Focused contract tests for the additive Demucs task facade."""

from hashlib import sha256
from pathlib import Path

import pytest
from torch import nn

from demucs_infer.clean_api import DemucsSeparator, DemucsSession, separate
from demucs_infer.apply import BagOfModels
from demucs_infer.checkpoint_catalog import CHECKPOINT_CATALOG, get_checkpoint_metadata


class _ChunkedResponse:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        self.read_sizes.append(size)
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class _FakeLoadedModel(nn.Module):
    def __init__(self, signature="fake-signature"):
        super().__init__()
        self.signature = signature
        self.audio_channels = 2
        self.samplerate = 44100
        self.sources = ["drums", "bass", "other", "vocals"]
        self.segment = 4.0

    def forward(self, wav):
        return wav


class _FakeSeparator:
    instances = []

    def __init__(self, model="htdemucs", **options):
        self.model_name = model
        self.model = _FakeLoadedModel(signature=model)
        self.samplerate = self.model.samplerate
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


def test_demucs_separator_auto_device_resolves_through_real_separator(monkeypatch):
    """`device="auto"` reaches the real `Separator.update_parameter`
    (unlike the `_FakeSeparator`-mocked tests above, which stub out
    `api.Separator` entirely and would only prove pass-through, not
    resolution). Only model loading is faked here -- via `api.get_model`,
    the seam `Separator._load_model` already calls through -- so the real
    device-resolution choke point runs unmodified."""
    import torch as th
    import demucs_infer.api as api

    monkeypatch.setattr(api, "get_model", lambda name, repo: _FakeLoadedModel(signature=name))

    helper = DemucsSeparator(model="mdx", device="auto")
    helper.load()
    expected = "cuda" if th.cuda.is_available() else "cpu"
    assert helper._separator._device == expected

    default_helper = DemucsSeparator(model="mdx")
    default_helper.load()
    assert default_helper._separator._device == helper._separator._device


def test_session_lifecycle_and_context_manager(monkeypatch):
    import demucs_infer.api as api
    monkeypatch.setattr(api, "Separator", _FakeSeparator)
    session = DemucsSession(device="cpu")
    assert session.status == "new"
    with pytest.raises(RuntimeError, match="must be loaded and ready"):
        session.infer("song.wav")
    assert session.load() is session
    assert session.status == "ready"
    assert session.samplerate == 44100
    assert session.sources == ("drums", "bass", "other", "vocals")
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


def test_session_properties_require_loaded_session():
    session = DemucsSession()
    with pytest.raises(RuntimeError, match="session must be loaded"):
        _ = session.samplerate
    with pytest.raises(RuntimeError, match="session must be loaded"):
        _ = session.sources


def test_default_session_materializes_package_pinned_checkpoint(monkeypatch, tmp_path):
    import demucs_infer.clean_api as clean_api
    import demucs_infer.repo as repo_module

    downloads = []
    verified = []
    checksums = []

    def fake_urlopen(url):
        downloads.append(url)
        return _ChunkedResponse(f"payload:{Path(url).name}".encode("ascii"))

    def fake_verify(self, path, expected):
        verified.append((path.name, expected))

    def fake_check_checksum(path, checksum):
        checksums.append((path.name, checksum))

    monkeypatch.setattr(clean_api, "urlopen", fake_urlopen)
    monkeypatch.setattr(DemucsSeparator, "_verify", fake_verify)
    monkeypatch.setattr(repo_module, "check_checksum", fake_check_checksum)
    monkeypatch.setattr(repo_module, "load_model",
                        lambda path: _FakeLoadedModel(signature=Path(path).stem.split("-", 1)[0]))

    session = DemucsSession(device="cpu", cache_dir=tmp_path).load()
    metadata = get_checkpoint_metadata("htdemucs")
    checkpoint = tmp_path / metadata["path"]
    assert checkpoint.exists()
    assert checkpoint.read_bytes() == f"payload:{checkpoint.name}".encode("ascii")
    assert isinstance(session._separator.model, BagOfModels)
    assert [model.signature for model in session._separator.model.models] == [metadata["signature"]]
    assert downloads == [metadata["url"]]
    assert [expected for _, expected in verified] == [metadata["sha256"]]
    assert checksums == [(metadata["path"], metadata["sha256"][:8])]


def test_htdemucs_ft_session_loads_pinned_four_model_bag(monkeypatch, tmp_path):
    import demucs_infer.clean_api as clean_api
    import demucs_infer.repo as repo_module

    downloads = []
    verified = []
    checksums = []

    def fake_urlopen(url):
        downloads.append(url)
        return _ChunkedResponse(f"payload:{Path(url).name}".encode("ascii"))

    def fake_verify(self, path, expected):
        verified.append((path.name, expected))

    def fake_check_checksum(path, checksum):
        checksums.append((path.name, checksum))

    monkeypatch.setattr(clean_api, "urlopen", fake_urlopen)
    monkeypatch.setattr(DemucsSeparator, "_verify", fake_verify)
    monkeypatch.setattr(repo_module, "check_checksum", fake_check_checksum)
    monkeypatch.setattr(repo_module, "load_model",
                        lambda path: _FakeLoadedModel(signature=Path(path).stem.split("-", 1)[0]))

    session = DemucsSession(model="htdemucs_ft", device="cpu", cache_dir=tmp_path).load()
    signatures = ["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"]
    expected = [get_checkpoint_metadata(signature) for signature in signatures]

    assert isinstance(session._separator.model, BagOfModels)
    assert len(session._separator.model.models) == 4
    assert [model.signature for model in session._separator.model.models] == signatures
    assert session.samplerate == 44100
    assert session.sources == ("drums", "bass", "other", "vocals")
    assert sorted(path.name for path in tmp_path.glob("*.th")) == sorted(item["path"] for item in expected)
    assert (tmp_path / "htdemucs_ft.yaml").exists()
    assert downloads == [item["url"] for item in expected]
    assert [digest for _, digest in verified] == [item["sha256"] for item in expected]
    assert checksums == [(item["path"], item["sha256"][:8]) for item in expected]


def test_checkpoint_download_reads_fixed_size_chunks(monkeypatch, tmp_path):
    import demucs_infer.clean_api as clean_api

    payload = b"checkpoint-payload"
    response = _ChunkedResponse(payload)
    monkeypatch.setattr(clean_api, "_IO_CHUNK_SIZE", 4)
    monkeypatch.setattr(clean_api, "urlopen", lambda url: response)

    session = DemucsSeparator(
        checkpoint_url="https://example.test/model.th",
        checkpoint_sha256=sha256(payload).hexdigest(),
        cache_dir=tmp_path,
    )
    path, _ = session._materialize_checkpoint()

    assert path.read_bytes() == payload
    assert response.read_sizes == [4, 4, 4, 4, 4, 4]
    assert not list(tmp_path.glob(".model.th.*.tmp"))


def test_failed_checkpoint_download_leaves_no_files(monkeypatch, tmp_path):
    import demucs_infer.clean_api as clean_api

    class FailingResponse(_ChunkedResponse):
        def read(self, size):
            if self.offset:
                raise OSError("connection lost")
            return super().read(size)

    monkeypatch.setattr(clean_api, "_IO_CHUNK_SIZE", 4)
    monkeypatch.setattr(clean_api, "urlopen", lambda url: FailingResponse(b"partial"))
    session = DemucsSeparator(
        checkpoint_url="https://example.test/model.th",
        checkpoint_sha256=sha256(b"complete").hexdigest(),
        cache_dir=tmp_path,
    )

    with pytest.raises(OSError, match="connection lost"):
        session._materialize_checkpoint()

    assert not (tmp_path / "model.th").exists()
    assert not list(tmp_path.glob(".model.th.*.tmp"))


def test_corrupted_cached_checkpoint_is_redownloaded_once(monkeypatch, tmp_path):
    import demucs_infer.clean_api as clean_api

    payload = b"valid-checkpoint"
    cached = tmp_path / "model.th"
    cached.write_bytes(b"corrupt")
    responses = []

    def fake_urlopen(url):
        response = _ChunkedResponse(payload)
        responses.append(response)
        return response

    monkeypatch.setattr(clean_api, "urlopen", fake_urlopen)
    session = DemucsSeparator(
        checkpoint_url="https://example.test/model.th",
        checkpoint_sha256=sha256(payload).hexdigest(),
        cache_dir=tmp_path,
    )

    path, _ = session._materialize_checkpoint()

    assert path.read_bytes() == payload
    assert len(responses) == 1
    assert not list(tmp_path.glob(".model.th.*.tmp"))
