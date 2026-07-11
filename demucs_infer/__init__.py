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
# __version__ is single-sourced from __about__.py (ADOPT campaign P5;
# pyproject.toml's `version` is `dynamic` and reads from the same file via
# hatchling -- see [tool.hatch.version] there). Previously duplicated here
# as a literal, which drifted from pyproject.toml.
# Original Demucs version: 4.1.0

from .__about__ import __version__  # noqa: F401

# Import compatibility module to ensure demucs module is available
from . import compat

# Reads: __about__ (__version__), compat (import side effect only, see
# compat.py's header)
