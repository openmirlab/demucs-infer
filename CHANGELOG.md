# Changelog

All notable changes to demucs-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No changes yet.

## [4.1.2] - 2026-01-14

### Added
- **Google Colab notebook** (`notebooks/quickstart_colab.ipynb`)
  - 7-section workflow: install, download model, upload audio, run separation, check outputs, preview results, download results
  - Support for both CLI and Python API methods
  - Colab badge and "Try it in Colab" section in README
- **Soundfile fallback** for audio I/O operations
  - `save_audio()` falls back to soundfile when torchaudio fails
  - `_load_audio()` now has 3 fallback levels: ffmpeg → torchaudio → soundfile

### Fixed
- **Python 3.12+ compatibility**: Fixed torchaudio/torchcodec dependency issue in newer Python environments (e.g., Google Colab)

## [4.1.1] - 2025-11-27

### Added
- **Comprehensive test suite** with pytest
  - Unit tests for logging, API, and model loading
  - Integration tests (marked with `@pytest.mark.slow`)
  - GitHub Actions CI workflow for automated testing on every push/PR
  - Test coverage validation in CI pipeline
- **Pytest configuration** in `pyproject.toml` with markers and test paths
- **Module aliasing** in `compat.py` for backward compatibility with pretrained models

### Changed
- **CI/CD Pipeline**: Updated GitHub Actions workflow to use UV and Python 3.10 only
- **Developer Documentation**: Updated PRINCIPLES.md and MAINTENANCE.md with pytest testing instructions
- **README.md**: Added testing section with examples and CI badge

### Fixed
- **CLI naming**: Fixed argument parser name from "demucsfix.separate" to "demucs-infer" in `separate.py`
- **Documentation cleanup**: Removed redundant files (docs/IMPLEMENTATION_NOTES.md, docs/dev/SETUP_COMPLETE.md)
- Fixed YAML config files inclusion in package distribution
- Updated GitHub Actions workflow to use release-based publishing
- Updated build configuration to explicitly include YAML and TXT files from remote/ directory
- Improved PyPI publishing workflow with package validation step

## [4.1.0] - 2025-10-03

### Overview

**demucs-infer** is an inference-only fork of the original [Demucs](https://github.com/facebookresearch/demucs) by Alexandre Défossez and Meta AI Research. This package was created to provide ongoing maintenance and PyTorch 2.x compatibility for Demucs inference capabilities, as the original repository is no longer actively maintained.

### Added

- **PyTorch 2.x Support**: Full compatibility with PyTorch 2.0+ and modern torchaudio versions
  - Removed `torchaudio<2.1` version restriction
  - Updated all dependencies for PyTorch 2.x compatibility

- **Inference-Only Packaging**: Streamlined package focusing solely on inference
  - Removed training code (~50% package size reduction)
  - Removed evaluation scripts
  - Kept all inference models and algorithms (100% unchanged)

- **Modern Dependency Management**:
  - UV package manager support with fast installation
  - Minimal core dependencies (7 packages vs 15+ in original)
  - Optional dependency groups: `[mp3]`, `[quantized]`, `[dev]`

- **Enhanced Documentation**:
  - Comprehensive README.md with installation guides
  - Migration guide from original Demucs
  - Implementation notes and technical details
  - Prominent attribution to original research

- **CLI Tool**: `demucs-infer` command (avoids conflicts with original `demucs`)

### Changed

- **Package Name**: `demucs` → `demucs-infer` (no naming conflicts)
- **Import Name**: `demucs` → `demucs_infer` (explicit, no conflicts)
- **License**: Updated with dual copyright attribution
  - Original: Copyright (c) Meta Platforms, Inc. and affiliates
  - Modifications: Copyright (c) 2025 Bo-Yu Chen

### Removed

- **Training Infrastructure**:
  - Training scripts (`train.py`, `solver.py`, etc.)
  - Training dependencies (hydra, dora-search, omegaconf, submitit)
  - Dataset utilities (musdb, museval)
  - Evaluation scripts

### Unchanged

All core functionality from original Demucs is preserved with **zero modifications**:

- ✅ All separation models (HTDemucs, HTDemucs-FT, HTDemucs-6s, MDX, MDX-Extra, quantized variants)
- ✅ Model architectures (identical neural networks)
- ✅ Separation algorithms (identical audio processing)
- ✅ Model weights (same pretrained checkpoints from official repositories)
- ✅ Audio quality (100% identical output to original Demucs)

### Credits

**All credit for the models, algorithms, and research belongs to:**
- **Alexandre Défossez** (Original author)
- **Meta AI Research** (Original research team)

**Research papers:**
- Hybrid Demucs (2021): [arXiv:2111.03600](https://arxiv.org/abs/2111.03600)
- Hybrid Transformer Demucs (2022): [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)

**Maintenance and PyTorch 2.x compatibility:**
- Bo-Yu Chen and worzpro Development Team

---

## Version History

- **[4.1.0]** - 2025-10-03: Initial release of demucs-infer with PyTorch 2.x support

---

## Future Plans

- Continue maintaining PyTorch compatibility
- Address community bug reports and issues
- Keep dependencies up-to-date
- Maintain 100% compatibility with original Demucs inference API

---

**Note**: Version numbering follows the original Demucs versioning (4.x series) to indicate compatibility level with original model architectures.
