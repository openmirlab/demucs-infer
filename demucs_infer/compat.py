# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint-unpickling compatibility shim -- import this before loading any
pretrained model.

Reads: hdemucs, htdemucs, demucs, states, spec, apply, repo, pretrained,
audio, utils (imported solely to populate the sys.modules aliases below)

This module's real, load-bearing job is the `sys.modules` aliasing below, not
the small `get_torch_arange` helper that follows it. Old checkpoints
(including community ones like drumsep) were pickled with class references
under the `demucs.*` module path. This module registers `demucs` and
`demucs.{hdemucs,htdemucs,demucs,states,spec,apply,repo,pretrained,audio,
utils}` as aliases into `sys.modules` at IMPORT TIME (a side effect, not a
function call) so `torch.load(..., weights_only=False)` can still resolve
those classes under this renamed, inference-only package. Both __init__.py
and separate.py import this module first for exactly that ordering reason --
removing or reordering the import breaks old-checkpoint loading silently (it
only surfaces as an unpickling error).

The explicit 10-module alias list below is an eagerness/convenience list, not
the sole resolution mechanism: because `sys.modules['demucs']` is aliased to
`demucs_infer` itself, `import demucs.<anything>` also resolves via the
parent package's `__path__` even for submodules not listed here (empirically
verified). Listing the common ones up front just avoids the extra import
step at unpickling time for the modules old checkpoints actually reference.
"""

import sys
import torch

# Module aliasing for backward compatibility with pretrained models
# Models saved with 'demucs' module name need this aliasing to load correctly
# This includes models like drumsep that were trained with the original demucs package
sys.modules['demucs'] = sys.modules['demucs_infer']

# Import and alias submodules for models that reference specific demucs submodules
# (e.g., drumsep uses demucs.hdemucs.HDemucs)
from . import hdemucs, htdemucs, demucs, states, spec, apply, repo, pretrained, audio, utils  # noqa: E402

sys.modules['demucs.hdemucs'] = hdemucs
sys.modules['demucs.htdemucs'] = htdemucs
sys.modules['demucs.demucs'] = demucs
sys.modules['demucs.states'] = states
sys.modules['demucs.spec'] = spec
sys.modules['demucs.apply'] = apply
sys.modules['demucs.repo'] = repo
sys.modules['demucs.pretrained'] = pretrained
sys.modules['demucs.audio'] = audio
sys.modules['demucs.utils'] = utils


def get_torch_arange(*args, **kwargs):
    """
    Wrapper for torch.arange that handles device parameter correctly across PyTorch versions.
    """
    return torch.arange(*args, **kwargs)
