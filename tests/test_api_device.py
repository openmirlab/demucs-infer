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


def test_resolve_device_passes_through_non_auto_values():
    for value in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        assert _resolve_device(value) == value


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
    for value in ("cpu", "cuda", "cuda:0", "cuda:1"):
        separator.update_parameter(device=value)
        assert separator._device == value
