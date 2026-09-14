"""Offline contract tests for native HTDemucs safetensors checkpoints."""

import hashlib
import json
import sys
from fractions import Fraction

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from demucs_infer import checkpoint_runtime
from demucs_infer import safetensors as safetensors_loader
from demucs_infer.checkpoint_runtime import CheckpointRuntime


class _FakeHTDemucs(nn.Module):
    def __init__(self, marker="default", *, segment=1, nested=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.marker = marker
        self.segment = segment
        self.nested = nested


class _RuntimeModel(nn.Module):
    audio_channels = 2
    samplerate = 44100
    sources = ("drums", "bass", "other", "vocals")
    segment = 10


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def _metadata(klass="demucs.htdemucs.HTDemucs"):
    fraction = {"_type": "fraction", "numerator": 39, "denominator": 5}
    return {
        "klass": klass,
        "args": json.dumps(["native"]),
        "kwargs": json.dumps({"segment": fraction, "nested": [fraction]}),
    }


def _write_checkpoint(
    tmp_path, *, metadata=None, tensors=None, name="model.safetensors"
):
    path = tmp_path / name
    save_file(
        tensors if tensors is not None else {"weight": torch.tensor([1.0, 2.0])},
        path,
        metadata=metadata,
    )
    return path


@pytest.mark.parametrize(
    "klass",
    ["demucs.htdemucs.HTDemucs", "demucs_infer.htdemucs.HTDemucs"],
)
def test_loads_both_native_class_names_and_nested_fractions(
    monkeypatch, tmp_path, klass
):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)
    model = safetensors_loader.load_safetensors_model(
        _write_checkpoint(tmp_path, metadata=_metadata(klass))
    )

    assert isinstance(model, _FakeHTDemucs)
    assert model.marker == "native"
    assert model.segment == Fraction(39, 5)
    assert model.nested == [Fraction(39, 5)]
    assert torch.equal(model.weight, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (None, "metadata is required"),
        ({**_metadata(), "structure": "{}"}, "metadata fields"),
        ({**_metadata(), "klass": "other.Model"}, "model class"),
        ({**_metadata(), "args": "{"}, "args metadata is not valid JSON"),
        ({**_metadata(), "args": "{}"}, "args metadata must be a JSON list"),
        ({**_metadata(), "kwargs": "[]"}, "kwargs metadata must be a JSON object"),
    ],
)
def test_rejects_missing_malformed_or_unsupported_metadata(
    monkeypatch, tmp_path, metadata, message
):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)
    path = _write_checkpoint(tmp_path, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        safetensors_loader.load_safetensors_model(path)


@pytest.mark.parametrize(
    "fraction",
    [
        {"_type": "class", "name": "anything"},
        {"_type": "fraction", "numerator": 1, "denominator": 0},
        {"_type": "fraction", "numerator": True, "denominator": 2},
        {"_type": "fraction", "numerator": 1.5, "denominator": 2},
        {"_type": "fraction", "numerator": 1, "denominator": 2, "extra": 3},
    ],
)
def test_rejects_invalid_structured_fraction_values(monkeypatch, tmp_path, fraction):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)
    metadata = _metadata()
    metadata["kwargs"] = json.dumps({"segment": fraction})

    with pytest.raises(ValueError, match="structured value|fraction"):
        safetensors_loader.load_safetensors_model(
            _write_checkpoint(tmp_path, metadata=metadata)
        )


def test_keeps_constructor_arguments_strict(monkeypatch, tmp_path):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)
    metadata = _metadata()
    metadata["kwargs"] = '{"unexpected": true}'

    with pytest.raises(TypeError, match="unexpected"):
        safetensors_loader.load_safetensors_model(
            _write_checkpoint(tmp_path, metadata=metadata)
        )


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        ({"unexpected": torch.ones(2)}, r"(?s)Missing key.*Unexpected key"),
        ({"weight": torch.ones(3)}, "size mismatch"),
    ],
)
def test_keeps_tensor_keys_and_shapes_strict(monkeypatch, tmp_path, tensors, message):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)

    with pytest.raises(RuntimeError, match=message):
        safetensors_loader.load_safetensors_model(
            _write_checkpoint(tmp_path, metadata=_metadata(), tensors=tensors)
        )


def test_rejects_truncated_safetensors(monkeypatch, tmp_path):
    monkeypatch.setattr(safetensors_loader, "HTDemucs", _FakeHTDemucs)
    path = _write_checkpoint(tmp_path, metadata=_metadata())
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(Exception, match="incomplete|invalid|header|offset|buffer"):
        safetensors_loader.load_safetensors_model(path)


def test_round_trips_actual_tiny_htdemucs_state(tmp_path):
    from demucs_infer.htdemucs import HTDemucs

    kwargs = {
        "sources": ["one", "two"],
        "audio_channels": 1,
        "channels": 2,
        "depth": 1,
        "nfft": 64,
        "t_layers": 0,
        "dconv_depth": 1,
        "dconv_comp": 1,
        "segment": Fraction(1, 1),
    }
    expected = HTDemucs(**kwargs)
    metadata_kwargs = dict(kwargs)
    metadata_kwargs["segment"] = {
        "_type": "fraction",
        "numerator": 1,
        "denominator": 1,
    }
    state = {key: value.contiguous() for key, value in expected.state_dict().items()}
    path = _write_checkpoint(
        tmp_path,
        metadata={
            "klass": "demucs_infer.htdemucs.HTDemucs",
            "args": "[]",
            "kwargs": json.dumps(metadata_kwargs),
        },
        tensors=state,
    )

    actual = safetensors_loader.load_safetensors_model(path)

    assert type(actual) is HTDemucs
    assert actual.segment == Fraction(1, 1)
    assert actual.training is True
    assert all(
        torch.equal(actual.state_dict()[key], value)
        for key, value in expected.state_dict().items()
    )


def test_safe_local_signature_collision_uses_only_explicit_checksum(
    monkeypatch, tmp_path
):
    path = _write_checkpoint(
        tmp_path, metadata=_metadata(), name="955717e8.safetensors"
    )
    runtime = CheckpointRuntime(checkpoint_path=path)
    resolution = runtime.resolve()

    assert resolution.expected_sha256 is None
    assert runtime.cache_info(loaded=False, status="new")["sha256"] is None
    monkeypatch.setattr(
        CheckpointRuntime,
        "_verify",
        lambda *args: pytest.fail("registry checksum inferred for safetensors"),
    )
    assert runtime.materialize_override(resolution)[0] == path


def test_safe_local_dispatch_never_uses_legacy_loaders(monkeypatch, tmp_path):
    path = _write_checkpoint(tmp_path, metadata=_metadata())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = _RuntimeModel()
    monkeypatch.setattr(
        safetensors_loader, "load_safetensors_model", lambda item: loaded
    )
    monkeypatch.setattr(
        "demucs_infer.states.load_model",
        lambda *args: pytest.fail("legacy states.load_model used for safetensors"),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.load used for safetensors"),
    )

    separator = CheckpointRuntime(
        checkpoint_path=path, checkpoint_sha256=digest
    ).load_separator({"device": "cpu"})

    assert separator.model is loaded
    assert separator._device == "cpu"


def test_safe_url_uses_url_pathname_and_verifies_before_loading(
    monkeypatch, tmp_path
):
    payload = b"verified safetensors payload"
    digest = hashlib.sha256(payload).hexdigest()
    events = []
    original_verify = CheckpointRuntime._verify

    def verify(runtime, path, expected):
        original_verify(runtime, path, expected)
        events.append("verified")

    def load(path):
        events.append("loaded")
        assert path == tmp_path / "model.safetensors"
        return _RuntimeModel()

    monkeypatch.setattr(checkpoint_runtime, "urlopen", lambda url: _Response(payload))
    monkeypatch.setattr(CheckpointRuntime, "_verify", verify)
    monkeypatch.setattr(safetensors_loader, "load_safetensors_model", load)
    monkeypatch.setattr(
        "demucs_infer.states.load_model",
        lambda *args: pytest.fail("legacy loader used for safetensors URL"),
    )
    runtime = CheckpointRuntime(
        checkpoint_url="https://example.test/model.safetensors?download=1",
        checkpoint_sha256=digest,
        cache_dir=tmp_path,
    )

    separator = runtime.load_separator({"device": "cpu"})

    assert separator.model.sources == _RuntimeModel.sources
    assert events == ["verified", "loaded"]


def test_safe_url_hash_failure_happens_before_model_construction(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        checkpoint_runtime, "urlopen", lambda url: _Response(b"wrong payload")
    )
    monkeypatch.setattr(
        safetensors_loader,
        "load_safetensors_model",
        lambda path: pytest.fail("model constructed before hash verification"),
    )
    runtime = CheckpointRuntime(
        checkpoint_url="https://example.test/model.safetensors",
        checkpoint_sha256="0" * 64,
        cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.load_separator({"device": "cpu"})


def test_safe_runtime_reports_actionable_missing_extra(monkeypatch, tmp_path):
    path = _write_checkpoint(tmp_path, metadata=_metadata())
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ModuleNotFoundError, match=r"demucs-infer\[safetensors\]"):
        CheckpointRuntime(checkpoint_path=path).load_separator({"device": "cpu"})


def test_legacy_th_override_keeps_existing_loader(monkeypatch, tmp_path):
    path = tmp_path / "custom.th"
    path.write_bytes(b"legacy")
    loaded = _RuntimeModel()
    calls = []
    monkeypatch.setattr(
        "demucs_infer.states.load_model",
        lambda item: calls.append(item) or loaded,
    )
    monkeypatch.setattr(
        safetensors_loader,
        "load_safetensors_model",
        lambda *args: pytest.fail("safetensors loader used for .th"),
    )

    separator = CheckpointRuntime(checkpoint_path=path).load_separator(
        {"device": "cpu"}
    )

    assert separator.model is loaded
    assert calls == [path]
