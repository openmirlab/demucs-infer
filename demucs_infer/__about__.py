# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Single source of truth for the package version (ADOPT campaign P5).

Before this, `version` was duplicated in pyproject.toml and
demucs_infer/__init__.py's `__version__` -- they drifted (pyproject.toml
read 4.2.0 while __init__.py still read 4.1.3 at the start of this
campaign, because scripts/bump_version.sh's sed-both-files approach missed
a manual bump somewhere along the way). hatchling now reads the version
straight from this file via `[tool.hatch.version] path =
"demucs_infer/__about__.py"` in pyproject.toml, and __init__.py imports it
from here instead of hardcoding its own copy. Bump only this file.

Reads: (nothing internal)
"""

__version__ = "4.2.1"
