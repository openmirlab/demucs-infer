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
