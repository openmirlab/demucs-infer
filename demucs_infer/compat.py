# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch 2.x compatibility utilities.
Provides helper functions for torch operations that work across different PyTorch versions.
Also provides module aliasing for backward compatibility with pretrained models.
"""

import sys
import torch

# Module aliasing for backward compatibility with pretrained models
# Models saved with 'demucs' module name need this aliasing to load correctly
# This includes models like drumsep that were trained with the original demucs package
sys.modules['demucs'] = sys.modules['demucs_infer']

# Import and alias submodules for models that reference specific demucs submodules
# (e.g., drumsep uses demucs.hdemucs.HDemucs)
from . import hdemucs, htdemucs, demucs, states, spec, apply, repo, pretrained, audio, utils

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


def get_torch_empty(*args, **kwargs):
    """
    Wrapper for torch.empty that handles device parameter correctly across PyTorch versions.
    """
    return torch.empty(*args, **kwargs)


def get_torch_tensor(*args, **kwargs):
    """
    Wrapper for torch.tensor that handles device parameter correctly across PyTorch versions.
    """
    return torch.tensor(*args, **kwargs)


def get_device_tensor(device):
    """
    Get a small tensor on the specified device for compatibility checks.
    """
    return torch.zeros(1, device=device)


def get_cuda_current_device():
    """
    Get current CUDA device, compatible across PyTorch versions.
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None
