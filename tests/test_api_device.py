"""Regression coverage for the `"auto"` device-string sentinel.

`Separator.update_parameter` is the single choke point every device value
flows through -- from `__init__`'s default/explicit arg and from any later
direct `update_parameter(device=...)` call alike. These tests exercise that
choke point directly (no model loading needed, since `_resolve_device` and
`update_parameter`'s device branch don't touch `self._model`), plus one
existing-behavior check that explicit device strings still pass through
unchanged.
"""

import inspect

import pytest
import torch as th

from demucs_infer.api import Separator, _resolve_device


def _bare_separator():
    """A `Separator` with none of `__init__`'s side effects (no model load).

    Mirrors the pattern `clean_api.py`'s `_get_separator` already uses for
    checkpoint-override construction (`Separator.__new__(Separator)`).
    """
    return Separator.__new__(Separator)


def test_resolve_device_auto_matches_cuda_availability():
    expected = "cuda" if th.cuda.is_available() else "cpu"
    assert _resolve_device("auto") == expected


def test_resolve_device_validates_explicit_values_and_preserves_cuda_index(monkeypatch):
    monkeypatch.setattr(th.cuda, "is_available", lambda: True)
    monkeypatch.setattr(th.cuda, "device_count", lambda: 2)
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_accepts_torch_cpu_device():
    assert _resolve_device(th.device("cpu")) == "cpu"


def test_resolve_device_rejects_invalid_or_unavailable_requests(monkeypatch):
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        _resolve_device("cuda")
    with pytest.raises(ValueError):
        _resolve_device("cuda:-1")
    with pytest.raises(ValueError):
        _resolve_device("metal")

    monkeypatch.setattr(th.cuda, "is_available", lambda: True)
    monkeypatch.setattr(th.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="index 1"):
        _resolve_device("cuda:1")


def test_resolve_device_rejects_unavailable_mps(monkeypatch):
    monkeypatch.setattr(th.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        _resolve_device("mps")


def test_default_unset_device_matches_auto_resolution():
    """The unset-device default (baked in at class-definition time) must
    resolve identically to the explicit `"auto"` sentinel."""
    default_device = inspect.signature(Separator.__init__).parameters["device"].default
    assert default_device == _resolve_device("auto")


def test_update_parameter_resolves_auto_device():
    separator = _bare_separator()
    separator.update_parameter(device="auto")
    expected = "cuda" if th.cuda.is_available() else "cpu"
    assert separator._device == expected
    assert separator._device != "auto"


def test_update_parameter_explicit_device_strings_pass_through_unchanged():
    separator = _bare_separator()
    from unittest.mock import patch

    with patch.object(th.cuda, "is_available", return_value=True), patch.object(th.cuda, "device_count", return_value=2):
        for value in ("cpu", "cuda", "cuda:0", "cuda:1"):
            separator.update_parameter(device=value)
            assert separator._device == value
