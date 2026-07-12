# Changelog

All notable changes to demucs-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Google Colab quickstart notebook (`notebooks/quickstart_colab.ipynb`) and
  the README's Colab badge/section — maintaining a separate notebook
  environment alongside the PyPI package was more upkeep than the audience
  justified.

## [4.2.2] - 2026-07-12

### Fixed
- **Fresh installs were completely unable to save (and, without FFmpeg,
  load) audio on the latest torch/torchaudio stack.** torchaudio>=2.11
  removed its bundled wav/flac/mp3 decoders in favor of a separate
  `torchcodec` package that nothing in our dependency tree declared; a
  brand-new `pip install demucs-infer` resolving the latest torchaudio
  hit `ImportError('TorchCodec is required for save_with_torchcodec...')`
  on every `save_audio()` call (reproduced with torch 2.13.0+cpu /
  torchaudio 2.11.0+cpu). The load path already degraded to a
  `soundfile`-based fallback (added earlier, see `0d21e4d` / `5d844c6`),
  but the save path's fallback and the `soundfile` package itself were
  never actually declared as a dependency — the fallback code existed
  but silently depended on a package that wasn't installed, so it failed
  too, just with a worse error message.
- `soundfile` is now a declared core dependency (`soundfile>=0.12.1`).
  It ships self-contained wheels (bundled `libsndfile`, no system FFmpeg
  needed, unlike `torchcodec`), so unlike `torchcodec` it's safe as a
  hard floor.
- Added a `torchcodec` extra (`pip install demucs-infer[torchcodec]`)
  for users who want torchaudio's own decoders back on torchaudio>=2.11;
  not required for normal use, since the fixes below cover it.
- `examples/basic_separation.py` no longer calls `torchaudio.load`
  directly (the same fragile pattern the library itself moved away from
  in `api.py`); it now loads via `demucs_infer.audio.AudioFile`, the
  package's own FFmpeg-based primary loader.

### Changed (decoder selection, evidence-based — see below for the numbers)
- **Load** (`Separator._load_audio`, used whenever FFmpeg — the primary
  path — is unavailable): wav/flac now go through `soundfile` directly,
  ahead of `torchaudio`, rather than only as a last-resort fallback.
  Verified bit-identical first: decoded `torchaudio==2.7.0+cpu` and
  `soundfile==0.14.0` against the same 16/24/32-bit PCM wav and FLAC
  files (mono + stereo) and asserted `np.array_equal` — exact match on
  every file. mp3 (and any other extension) is deliberately **not**
  included: the same comparison measured mp3 decode to differ by up to
  `7.15e-7` per sample between torchaudio (ffmpeg-backed) and soundfile
  (libmpg123-backed) — small, but not zero, so silently switching would
  have changed existing users' output. mp3/lossy loads stay on
  `torchaudio` only; if it raises (torchaudio>=2.11 without
  `torchcodec`), `_load_audio` now raises a clear, actionable error
  ("install torchcodec, or convert the file to wav/flac") instead of
  silently decoding via `soundfile`.
- **Save** (`demucs_infer.audio.save_audio`): explicitly **kept
  unchanged** — `torchaudio.save` is tried first, `soundfile` only on
  failure — after the same kind of check found the opposite result for
  encoding: writing identical float32 samples as 16-bit PCM wav via
  `torchaudio.save` vs `soundfile.write` differ by ±1 LSB in ~50% of
  samples (a real rounding-convention difference between the two
  encoders' float→int16 quantization, not measurement noise; file sizes
  also differ slightly due to header/chunk differences). Since bit-
  identity for encode does **not** hold, `soundfile` cannot become the
  default encoder without an unproven, silent output change for anyone
  currently on a working torchaudio install — so it stays a fallback,
  engaged only when torchaudio itself is already broken (nothing to
  regress in that case).
- Added `tests/test_audio_fallback.py` (11 tests) covering: soundfile-
  first-not-fallback for wav/flac loads; torchaudio-only (no soundfile)
  for mp3 loads, with an actionable error message on failure; save
  keeping torchaudio-first ordering; soundfile never invoked when
  torchaudio already works (zero behavior change); the broad `except
  Exception` around `torchaudio.load` from `5d844c6` not regressing to a
  narrower catch; and `soundfile` being a declared dependency.

### Investigated (no change)
- Confirmed `torchaudio` is used for I/O only in the shipped package —
  exactly two call sites (`audio.py`'s `ta.save`, `api.py`'s `ta.load`),
  nothing from `torchaudio.functional`/`torchaudio.transforms`/models.
  `tools/capture_baseline.py` (a dev-only script, not part of the
  installed package) additionally calls `torchaudio.functional.resample`
  for baseline capture. This means a future major version could
  plausibly make `torchaudio` itself optional (soundfile covers wav/flac;
  something FFmpeg-based would be needed for mp3) — not done here, flagged
  for a future decision.

## [4.2.1] - 2026-07-11

(Version note: 4.2.0 on PyPI is an inadvertent re-publish of the pre-campaign
main branch — functionally identical to 4.1.3. All changes below ship in 4.2.1.)

### Changed
- **Dropped the `openunmix` dependency**: the only thing this package used
  from it (`openunmix.filtering.wiener`, optional Wiener-filter
  post-processing on some model configs) is now vendored into
  `demucs_infer/wiener.py` (MIT-licensed, with attribution to
  open-unmix-pytorch v1.3.0). Verified bit-identical output, both via a
  direct unit test against the original across all softmask/residual/
  iterations code paths and a real end-to-end run (mdx_extra, whose
  cac=False bag members exercise this at inference; htdemucs's shipped
  config never does).
- **api.py split**: model discovery/metadata (`ModelInfo`, `get_model_info`,
  `list_models`, `list_supported_separation_types`, `KNOWN_MODELS`,
  `SEPARATION_TYPES`, `SOURCE_TRANSLATIONS`) moved to the new
  `demucs_infer/model_info.py`; `api.py` re-exports all of them, so
  existing `from demucs_infer.api import ...` usage is unaffected.
  `api.py`: 783 -> 412 lines.
- **Version single-sourcing**: `__version__` now lives only in
  `demucs_infer/__about__.py`; `pyproject.toml`'s `version` is `dynamic`
  and hatchling reads it from that file. (Previously duplicated in both
  places and had drifted.)
- File-top navigability headers added across `demucs_infer/*.py`.

### Added
- **Numpy** added as an explicit dependency (it was always imported
  directly by `audio.py`/`transformer.py`, but had been an unlisted
  transitive dependency pulled in via `openunmix`/the `community` extra's
  `gdown`).
- **Checksum verification for community models**: `GDriveRepo` downloads
  are now sha256-verified via the existing `repo.check_checksum` helper
  (previously unverified). See `docs/checkpoints_provenance.json`.
- **`tests/test_checkpoints_liveness.py`**: HEADs every official + community
  checkpoint URL; marked `network`, deselected by default
  (`pytest -m network` to run).
- **`tests/test_baseline_regression.py`** and **`tests/test_import.py`**:
  seeded htdemucs output regression test and an import smoke test.
- **`CLAUDE.md`**: repo scope, verification commands, header convention,
  and the two known-unfixed issues (unseeded `shifts` randomness, the
  `segment=` + HTDemucs crash) with rationale for leaving them as-is.
- **CI test gate**: `publish.yml` now runs the test suite in a `test` job
  before `publish` (`needs: [test]`) -- nothing publishes to PyPI without
  tests passing.

### Removed
- Leftover training-only dead code: `states.py`'s `get_quantizer`,
  `get_state`, `save_with_checksum`, `copy_state`, `swap_state`, and
  `utils.py`'s `random_subset` -- none reachable from any inference path
  (confirmed via repo-wide grep before removal).

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
- **CLI naming**: Fixed argument parser name to "demucs-infer" in `separate.py`
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
