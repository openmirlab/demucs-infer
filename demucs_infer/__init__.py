# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# demucs-infer: Inference-only distribution of Demucs for PyTorch 2.x.
# Package entry point -- primarily exists to import `compat` first (its
# sys.modules aliasing must run before any pretrained-model unpickling) and
# to expose __version__.
#
# NOTE: __version__ below is a second source of truth alongside
# pyproject.toml's `version` field; they have drifted before (see ADOPT P5).
# Original Demucs version: 4.1.0

__version__ = "4.1.3"

# Import compatibility module to ensure demucs module is available
from . import compat

# Reads: compat (import side effect only, see compat.py's header)
