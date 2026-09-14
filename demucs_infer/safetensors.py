# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Load native HTDemucs checkpoints stored as safetensors.

Adapted from ``demucs/hf.py`` in the MIT-licensed Demucs repository at
revision 2883f3db65617d6d178c6ed10d869dc14e44e59b. This loader accepts only
the flat HTDemucs metadata shape and never imports classes from checkpoint
metadata or falls back to pickle deserialization.

Reads: htdemucs.HTDemucs.
"""

from __future__ import annotations

import json
import typing as tp
from fractions import Fraction
from pathlib import Path

from .htdemucs import HTDemucs

_MODEL_CLASS_TAGS = frozenset(
    {
        "demucs.htdemucs.HTDemucs",
        "demucs_infer.htdemucs.HTDemucs",
    }
)
_METADATA_KEYS = frozenset({"klass", "args", "kwargs"})


def load_safetensors_model(path: tp.Union[str, Path]) -> HTDemucs:
    """Load a native HTDemucs safetensors checkpoint from ``path``."""
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        if error.name != "safetensors":
            raise
        raise ModuleNotFoundError(
            "Safetensors checkpoint support requires the optional extra; "
            "install it with `pip install 'demucs-infer[safetensors]'`."
        ) from error

    with safe_open(str(path), framework="pt") as file:
        metadata = file.metadata()
        tensors = {key: file.get_tensor(key) for key in file.keys()}

    if metadata is None:
        raise ValueError("Demucs safetensors metadata is required.")
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("Unsupported Demucs safetensors metadata fields.")
    if metadata["klass"] not in _MODEL_CLASS_TAGS:
        raise ValueError(
            f"Unsupported Demucs safetensors model class: {metadata['klass']}"
        )

    args = _decode_metadata_json(metadata["args"], "args")
    kwargs = _decode_metadata_json(metadata["kwargs"], "kwargs")
    if not isinstance(args, list):
        raise ValueError("Demucs safetensors args metadata must be a JSON list.")
    if not isinstance(kwargs, dict):
        raise ValueError("Demucs safetensors kwargs metadata must be a JSON object.")

    model = HTDemucs(*args, **kwargs)
    model.load_state_dict(tensors, strict=True)
    return model


def _decode_metadata_json(value: str, field: str) -> tp.Any:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"Demucs safetensors {field} metadata is not valid JSON."
        ) from error
    return _decode_json(decoded)


def _decode_json(value: tp.Any) -> tp.Any:
    if isinstance(value, dict):
        if "_type" in value:
            if (
                set(value) != {"_type", "numerator", "denominator"}
                or value["_type"] != "fraction"
            ):
                raise ValueError("Unsupported Demucs safetensors structured value.")
            numerator = value["numerator"]
            denominator = value["denominator"]
            if (
                not isinstance(numerator, int)
                or isinstance(numerator, bool)
                or not isinstance(denominator, int)
                or isinstance(denominator, bool)
                or denominator == 0
            ):
                raise ValueError(
                    "Demucs safetensors fraction must contain valid integer values."
                )
            return Fraction(numerator, denominator)
        return {key: _decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


__all__ = ["load_safetensors_model"]
