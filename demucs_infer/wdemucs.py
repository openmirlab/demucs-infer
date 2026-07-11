# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Backward-compat alias for old pickled checkpoints (intentionally a 9-line
# file, not dead code): checkpoints trained/saved before HDemucs absorbed
# WDemucs pickle their model class as `demucs.wdemucs.WDemucs`. states.load_model
# resolves that class by reading this module, so `WDemucs = HDemucs` must
# keep existing even though nothing in this codebase constructs a WDemucs
# directly. Do not remove or merge into hdemucs.py.
#
# For compat
from .hdemucs import HDemucs

WDemucs = HDemucs
