# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities to save and load models.

`load_model`/`set_state` are the inference-time checkpoint loader every
Repo (repo.py, community.py) calls after fetching a `.th` file: reconstructs
the model class from the pickled `klass`/`args`/`kwargs`, then loads its
state dict (transparently handling diffq-quantized states via
`restore_quantized_state`). `capture_init` is the decorator model
`__init__`s use to record those args/kwargs at construction time so they
round-trip through pickling. Training-only counterparts (quantizer setup
with an optimizer, checkpoint-saving/EMA-state-swapping helpers) were
removed as dead code in the ADOPT campaign's P2 -- this file only retains
the inference-time load path.

Reads: log (fatal)
"""
import functools
import inspect
from pathlib import Path
import warnings

from .log import fatal
import torch


def _check_diffq():
    try:
        import diffq  # noqa
    except ImportError:
        fatal('Trying to use DiffQ, but diffq is not installed.\n'
              'On Windows run: python.exe -m pip install diffq \n'
              'On Linux/Mac, run: python3 -m pip install diffq')


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            # Use weights_only=False for PyTorch 2.6+ compatibility
            # Models serialized with class references (e.g., drumsep, standard demucs models)
            # require this to load properly
            package = torch.load(path, 'cpu', weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get('__quantized'):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state['quantized'])
        else:
            _check_diffq()
            from diffq import restore_quantized_state
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
