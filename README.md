# demucs-infer

**Inference-only distribution of Demucs for PyTorch 2.x**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyPI](https://img.shields.io/pypi/v/demucs-infer)](https://pypi.org/project/demucs-infer/)

High-quality audio source separation models for extracting vocals, drums, bass, and other instruments from music tracks.

---

## Why This Exists

The original [Demucs](https://github.com/facebookresearch/demucs) repository by Meta AI Research is **no longer actively maintained**. The models remain state-of-the-art, but the package never received updates for modern PyTorch: it pins `torchaudio<2.1`, carries training-only dependencies (`dora-search`, `hydra`) that most inference users never need, and its packaging predates PEP 621.

**demucs-infer** re-provides the same models and separation quality as an inference-only, PyPI-installable package:

1. **Maintain compatibility** — works with PyTorch 2.x (no `torchaudio<2.1` restriction) and Python 3.8+.
2. **Continue development** — addresses issues and papers over gaps (e.g. torchaudio 2.11+ dropping its bundled decoders) that the unmaintained upstream never will.
3. **Focus on inference** — training code, evaluation scripts, and dataset utilities are removed for a leaner package.
4. **Serve the community** — lets researchers and developers keep using these models without maintaining a fork themselves.

### Before / After

| Aspect | Original Demucs | demucs-infer |
|--------|----------------|--------------|
| **Maintenance status** | No longer actively maintained | Active |
| **PyTorch support** | 1.8.x – 2.0.x (with `torchaudio<2.1`) | 2.0+, no restriction |
| **Python files** | 36+ files | 17 files (~47% smaller) |
| **Core dependencies** | 15+ packages | 8 packages (~47% fewer) |
| **Training code** | Included | Removed (inference-only) |
| **Inference code / quality** | High | Identical (zero algorithm changes) |
| **CLI / import name** | `demucs` / `demucs` | `demucs-infer` / `demucs_infer` (no conflicts) |
| **Model weights** | Official Meta checkpoints | Same Meta checkpoints plus verified, source-pinned compatible checkpoints |

## Acknowledgments

**demucs-infer** is built on the research, architectures, and official models of the original Demucs project. This package maintains the packaging and PyTorch 2.x compatibility layer, and its checkpoint registry also exposes compatible third-party weights from their original public sources.

- **Upstream organization:** [Meta AI Research (FAIR)](https://ai.meta.com/research/)
- **Individual authors:** Alexandre Défossez (both papers below, see [Citation](#citation)); Simon Rouard and Francisco Massa (Hybrid Transformer Demucs, co-authors — see [arXiv:2211.08553](https://arxiv.org/abs/2211.08553))
- **Source repository:** [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- **Official pretrained weights host:** [dl.fbaipublicfiles.com/demucs](https://dl.fbaipublicfiles.com/demucs/) — Meta's public checkpoint CDN. Compatible third-party sources are credited in [Registry-backed compatible models](#registry-backed-compatible-models); demucs-infer hosts no mirror.

**What this package changed vs. upstream:** PyTorch 2.x compatibility, inference-only packaging, and modern dependency management. **What stayed identical:** every model architecture, every separation algorithm, and every pretrained weight — see [Scope](#scope) below for the full breakdown.

## Citation

If you use demucs-infer in your research, please cite the original Demucs papers — this package is a maintenance fork; all credit for the models, algorithms, and research belongs to the original authors.

**Hybrid Demucs (2021):**

```bibtex
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

**Hybrid Transformer Demucs (2022):**

```bibtex
@article{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  journal={arXiv preprint arXiv:2211.08553},
  year={2022}
}
```

## Features

- **PyTorch 2.x Support**: Compatible with modern PyTorch versions (no `torchaudio<2.1` restriction)
- **Inference-Only**: ~50% smaller than original package (removed training code)
- **Minimal Dependencies**: 8 core packages (vs 15+ in original)
- **API Compatible**: Drop-in replacement for inference workflows
- **Same Quality**: Zero changes to separation algorithms
- **Schema-v2 Model Registry**: Official models plus verified UVR, CDX23, MSST, and DrumSep recipes
- **Model Info API**: Query model capabilities, separation types, and source translations
- **Third-Party Model Support**: Stable public names, exact stems, source URLs, and full SHA-256 verification

## Scope

### In scope — what we built

- PyTorch 2.x compatibility layer (removed version restrictions)
- PyTorch 2.6+ support (compatible with `weights_only` default changes)
- A minimal logging module replacing the `dora-search` dependency
- Lazy imports so optional dependencies are truly optional
- Inference-only packaging (7 core packages instead of 15+)
- Model Info API — query model capabilities, separation types, and source translations
- Third-party model support (module aliasing for community-trained models)

### In scope — what stays unchanged from upstream

- All separation models: HTDemucs, MDX, and all variants
- Model architectures: zero modifications to the neural networks
- Separation algorithms: identical audio processing
- Model weights: unchanged official checkpoints plus source-pinned compatible checkpoints
- Audio quality: 100% identical output (bit-for-bit gated — see [CLAUDE.md](CLAUDE.md))

### Out of scope, forever

- Training code (`train.py`, `solver.py`, etc.) — this is an inference-only package by design, not a temporary gap
- Evaluation scripts (`evaluate.py`)
- Training-only dependencies (`hydra`, `dora-search`, `omegaconf`)
- Dataset utilities (`musdb`, `museval`)
- Distributed training tools (`submitit`)

To retrain or evaluate against the original benchmarks, use the upstream [facebookresearch/demucs](https://github.com/facebookresearch/demucs) repository directly.

## Install

demucs-infer is available on [PyPI](https://pypi.org/project/demucs-infer/) and supports both **UV** (recommended, faster) and **pip** (traditional).

**With UV:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have UV yet
uv add demucs-infer
```

**With pip:**
```bash
python -m venv .venv && source .venv/bin/activate  # recommended
pip install demucs-infer
```

### Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0 or later
- **OS**: Linux, macOS, Windows
- **GPU**: Optional (CUDA-capable GPU recommended for speed)

### Optional Dependencies

**With UV:**
```bash
uv add "demucs-infer[mp3]"         # MP3 output support
uv add "demucs-infer[quantized]"   # Quantized models
uv add "demucs-infer[community]"   # Community model downloads (Google Drive)
uv add "demucs-infer[torchcodec]"  # Restore torchaudio's own decoders on torchaudio>=2.11
uv add "demucs-infer[mp3,quantized,community,torchcodec]"  # all of the above
```

**With pip:**
```bash
pip install demucs-infer[mp3]         # Adds: lameenc>=1.2
pip install demucs-infer[quantized]   # Adds: diffq>=0.2.1
pip install demucs-infer[community]   # Adds: gdown>=5.0.0
pip install demucs-infer[torchcodec]  # Adds: torchcodec
pip install "demucs-infer[mp3,quantized,community,torchcodec]"
```

## Quick Start

### Task-level facade

For a small, package-qualified entry point, use `DemucsSeparator`. It accepts
audio paths or tensors and returns the existing Demucs output tuple: the
normalized mixture tensor and a mapping of source names to separated tensors.
Tensor inputs require their original sample rate; path inputs are decoded and
resampled by the existing `Separator` loader.

```python
import torch

from demucs_infer import DemucsSeparator, separate_file

# One-shot calls create and discard a fresh model helper.
mixture, stems = separate_file("song.wav", model="htdemucs")

# Reuse a loaded model across calls (calls are serialized for safety).
separator = DemucsSeparator(model="htdemucs", device="cpu")
mixture, stems = separator("song.wav")
waveform = torch.zeros(2, 44100)  # or a loaded [channels, time] waveform
mixture, stems = separator(waveform, sample_rate=44100)
```

### Model lifecycle and checkpoint customization

`DemucsSession` is the package-qualified lifecycle surface. It owns model
loading, checkpoint verification, the in-process model, and release while
leaving the existing `Separator` API unchanged:

```python
from demucs_infer import DemucsSession

with DemucsSession(model="htdemucs", device="cpu") as session:
    mixture, stems = session.infer("song.wav")
    print(session.samplerate, session.sources)
    print(session.status, session.cache_info())
```

`load()` may be called explicitly; the session-level `infer()` requires the
session to be `ready` after `load()` (or context-manager entry). The legacy
callable form, `session(...)`, remains lazy for backward compatibility.
`release()` clears the in-memory model but keeps disk checkpoints cached and
permits a later reload; `close()` is terminal and idempotent. Device requests
preserve legacy `None`/`auto` selection and accept explicit `cpu`, `cuda`,
`cuda:N`, or available `mps`, rejecting invalid/unavailable requests before
loading. `status` reports `new`, `loading`, `ready`, `failed`, `released`, or
`closed`. A custom checkpoint
can be supplied with `checkpoint_path`, or downloaded with
`checkpoint_url` plus its required full `checkpoint_sha256`. The package ships
its release-pinned metadata in the package-local
`demucs_infer/config/checkpoints.toml` file (URLs, SHA-256 digests, license,
provenance, and source revision). Runtime inference reads this local snapshot
and does not depend on a remote catalog. Use `checkpoint_catalog()` or
`checkpoint_config_path()` to inspect it.

Named models download to `~/.cache/demucs-infer/` by default. Pass
`cache_dir=Path("/your/checkpoint/cache")` to `DemucsSession` or
`DemucsSeparator` to override that location. `checkpoint_path` remains the
single-file override, while explicit legacy `get_model(name, repo=Path(...))`
repositories remain supported for existing callers.

The facade is additive: advanced users can continue composing
`demucs_infer.api.Separator`, `demucs_infer.pretrained.get_model`, and
`demucs_infer.apply.apply_model` directly. Optional backends remain lazy and
retain their existing installation requirements.

```python
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import AudioFile, save_audio
import torch

# Load model
model = get_model("htdemucs_ft")
model.eval()

# Load audio (AudioFile is demucs-infer's own FFmpeg-based reader -- see
# "torchaudio 2.11+ and audio decoders" below for why this is preferred
# over calling torchaudio.load directly)
sr = model.samplerate
wav = AudioFile("song.wav").read(streams=0, samplerate=sr, channels=model.audio_channels)
wav = wav.unsqueeze(0)  # Add batch dimension

# Separate audio
with torch.no_grad():
    sources = apply_model(model, wav, device="cuda")

# Save separated stems
# sources shape: [1, 4, channels, time]
# sources order: drums, bass, other, vocals
for i, source_name in enumerate(model.sources):
    source = sources[0, i]  # Remove batch dimension
    save_audio(source, f"output/{source_name}.wav", sr)
```

Equivalent from the CLI:

```bash
demucs-infer "song.wav"                      # basic usage (all 4 stems)
demucs-infer --two-stems=drums "song.wav"    # drums only, faster
demucs-infer -n htdemucs_ft -o output/ "song.wav"   # pick a model + output dir
```

(Prefix any of these with `uv run` if you installed with UV into a project environment rather than activating a venv.)

---

## Available Models

### Official Demucs Models

#### 4-Source Models (drums, bass, other, vocals)

| Model | Quality | Speed | Description |
|-------|---------|-------|-------------|
| `htdemucs` | ⭐⭐⭐⭐⭐ | Medium | Hybrid Transformer Demucs (default) |
| `htdemucs_ft` | ⭐⭐⭐⭐⭐ | Medium | Fine-tuned version (recommended) |
| `mdx` | ⭐⭐⭐⭐ | Fast | MDX model |
| `mdx_extra` | ⭐⭐⭐⭐⭐ | Medium | Enhanced MDX |
| `mdx_q` | ⭐⭐⭐ | Very Fast | Quantized MDX |
| `mdx_extra_q` | ⭐⭐⭐⭐ | Fast | Quantized enhanced MDX |

#### 6-Source Models (drums, bass, other, vocals, guitar, piano)

| Model | Quality | Speed | Description |
|-------|---------|-------|-------------|
| `htdemucs_6s` | ⭐⭐⭐⭐⭐ | Medium | 6-source separation |

### Registry-backed compatible models

These names use the same package-owned resolver as official models. Stems in
the table are the exact programmatic dictionary keys returned by the API.
`speech` is the dialogue stem in the CDX23 recipe. DrumSep's Spanish keys are
preserved; their English meanings are bombo = kick, redoblante = snare, and
platillos = cymbals.

| Public model name | Recipe | Programmatic stems | Weight license recorded by source |
|-------------------|--------|--------------------|-----------------------------------|
| `uvr_demucs_model_1` | UVR Model 1 (`ebf34a2db`) | vocals, non_vocals | not stated |
| `uvr_demucs_model_2` | UVR Model 2 (`ebf34a2d`) | vocals, non_vocals | not stated |
| `uvr_demucs_model_bag` | UVR ensemble, Model 2 then Model 1 | vocals, non_vocals | not stated |
| `cdx23_dnr` | Three-component CDX23 DnR bag | music, sfx, speech | not stated |
| `msst_htdemucs_vocals` | MSST HTDemucs vocals state dict | vocals, other | MIT |
| `drumsep` | DrumSep (`49469ca8`) | bombo, redoblante, platillos, toms | MIT |

```python
from demucs_infer.pretrained import get_model

# Registry-backed models download and verify automatically.
model = get_model("cdx23_dnr")

# Explicit local repositories remain available for legacy/custom layouts.
from pathlib import Path
local_model = get_model("my_signature", repo=Path("/path/to/models"))
```

## Use Cases

**Music Production** — extract vocals for remixing, isolate drums for sampling, remove vocals for karaoke, separate instruments for analysis.

**Machine Learning** — prepare training data for downstream music ML models, audio preprocessing, dataset augmentation.

**Research** — music information retrieval (MIR), audio signal processing, music transcription.

## Model Info API

Query model capabilities, separation types, and get source name translations programmatically.

```python
from demucs_infer.api import get_model_info, list_supported_separation_types

# Get detailed info about a model
info = get_model_info("htdemucs_ft")
print(info)
# Output:
# HT-Demucs Fine-tuned (htdemucs_ft)
#   Type: Music Separation (4 stems)
#   Architecture: HTDemucs (ensemble of 4)
#   Sources: drums, bass, other, vocals
#   Sample Rate: 44100 Hz
#   Use Case: High quality music separation

# Access individual properties
print(info.sources)          # ['drums', 'bass', 'other', 'vocals']
print(info.separation_type)  # 'music_4stem'
print(info.is_bag)           # True (ensemble model)
print(info.num_models)       # 4

# Get info for third-party model with source translation
from pathlib import Path
info = get_model_info("49469ca8", repo=Path("/path/to/drumsep"))
print(info.sources)          # ['bombo', 'redoblante', 'platillos', 'toms'] (Spanish)
print(info.sources_english)  # ['kick', 'snare', 'cymbals', 'toms'] (English)
```

List supported separation types:

```python
from demucs_infer.api import list_supported_separation_types

types = list_supported_separation_types()
for key, info in types.items():
    print(f"{key}: {info['name']}")

# Output:
# music_4stem: Music Separation (4 stems)
# music_6stem: Music Separation (6 stems)
# drum_kit: Drum Kit Separation
# cinematic: Cinematic/Film Audio Separation
# speech: Speech Separation
# vocal_instrumental: Vocal/Instrumental Separation
```

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Model name/signature |
| `display_name` | str | Human-readable name |
| `architecture` | str | Model architecture (HTDemucs, HDemucs, etc.) |
| `sources` | List[str] | Original source names |
| `sources_english` | List[str] | English-translated source names |
| `separation_type` | str | Type key (e.g., 'music_4stem') |
| `separation_type_name` | str | Human-readable type name |
| `description` | str | Model description |
| `use_case` | str | Recommended use case |
| `sample_rate` | int | Audio sample rate (Hz) |
| `audio_channels` | int | Number of audio channels |
| `is_bag` | bool | Whether it's an ensemble model |
| `num_models` | int | Number of models in ensemble |

## Advanced Usage

### Two-Stems Separation (Faster)

```bash
# CLI: Extract drums only (faster than 4-source)
demucs-infer --two-stems=drums "song.wav"
```

```python
# Python API: Extract specific stem
model = get_model("htdemucs_ft")
# Model will automatically optimize for two-stem separation
```

### Batch Processing

```python
import torch
from pathlib import Path
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import AudioFile, save_audio

model = get_model("htdemucs_ft").cuda().eval()

audio_files = list(Path("input/").glob("*.wav"))

for audio_file in audio_files:
    wav = AudioFile(audio_file).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels
    )
    sr = model.samplerate
    wav = wav.unsqueeze(0).cuda()

    with torch.no_grad():
        sources = apply_model(model, wav, device="cuda")

    output_dir = Path("output") / audio_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, source_name in enumerate(model.sources):
        save_audio(sources[0, i].cpu(), output_dir / f"{source_name}.wav", sr)

    print(f"Processed: {audio_file.name}")
```

### Custom Device Selection

```python
import torch

# Auto-detect best device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model("htdemucs_ft")
model = model.to(device)
model.eval()

# Or specify explicitly
model = model.to("cuda:0")  # GPU 0
model = model.to("cpu")     # CPU
```

## Dependencies

### Core Dependencies (8 packages)

```toml
torch>=2.0.0
torchaudio>=2.0.0
soundfile>=0.12.1
einops
julius>=0.2.3
numpy
pyyaml
tqdm
```

> `openunmix` was dropped as a dependency: the only thing this package ever
> used from it (`openunmix.filtering.wiener`, for optional Wiener-filter
> post-processing on some model configs) is now vendored directly into
> `demucs_infer/wiener.py` (MIT-licensed, with attribution). `numpy` was
> added explicitly -- it was always used directly by this package, but had
> been an unlisted transitive dependency (pulled in by openunmix) until now.
> `soundfile` was added in 4.2.2 as the wav/flac decoder used whenever
> FFmpeg isn't available (see "torchaudio 2.11+ and audio decoders" below)
> -- unlike `torchcodec`, it ships self-contained wheels with no system
> FFmpeg requirement, so it's safe as a hard dependency.

### torchaudio 2.11+ and audio decoders

`torchaudio>=2.11` removed its bundled wav/flac/mp3 decoders; `torchaudio.load`
and `torchaudio.save` now require the separate
[`torchcodec`](https://github.com/pytorch/torchcodec) package (which itself
needs system FFmpeg shared libraries), and raise `ImportError` without it.

demucs-infer handles this automatically for the vast majority of installs
(anyone with FFmpeg on `PATH`, which is already how tracks are read
primarily) and degrades predictably otherwise:

- **Loading** tries FFmpeg first (unaffected by any of this), same as
  always. If FFmpeg isn't available:
  - **wav/flac** go through `soundfile` directly -- verified bit-identical
    to torchaudio's own decode (`np.array_equal` exact, PCM 16/24/32-bit + FLAC,
    mono/stereo), so there's no accuracy difference either way.
  - **mp3** (and anything else) stays on `torchaudio` only -- its mp3
    decode was measured to differ slightly from soundfile's (~7e-7 max
    per-sample difference, different underlying decoders), so
    demucs-infer does **not** silently switch decoders for lossy formats.
    If torchaudio itself can't decode (missing `torchcodec` on
    torchaudio>=2.11), you'll get a clear error telling you to install
    `torchcodec` or convert the file to wav/flac first.
- **Saving** tries `torchaudio.save` first and uses `soundfile` only if
  that raises. This one is intentionally *not* soundfile-first: writing
  identical samples as 16-bit PCM wav via `torchaudio.save` vs
  `soundfile.write` was measured to differ by ±1 LSB in about half of
  samples (a real rounding-convention difference, not noise), so
  soundfile can't be the default encoder without changing output for
  installs where torchaudio already works -- it's used only when
  torchaudio itself is already broken.

Two ways to opt back into torchaudio's own decoders instead of relying on
these fallbacks, if you prefer:

```bash
# Option A: install torchcodec (needs system FFmpeg shared libraries)
pip install demucs-infer[torchcodec]

# Option B: pin an older torchaudio that still bundles its own decoders
pip install "torchaudio<2.11"
```

## Troubleshooting

### ImportError: No module named 'demucs_infer'

```bash
# Make sure you installed demucs-infer, not demucs
pip uninstall demucs
pip install demucs-infer
```

### CUDA Out of Memory

```python
# Use smaller chunks or CPU
model = model.to("cpu")

# Or use two-stems mode (faster)
# demucs-infer --two-stems=drums "audio.wav"
```

### `ImportError: TorchCodec is required for ...` / `LoadAudioError` mentioning torchcodec

This comes from `torchaudio` itself (`torchaudio>=2.11` requires the
separate `torchcodec` package for its own decoders). For **wav/flac**,
demucs-infer catches it internally and uses `soundfile` instead, so you
shouldn't normally see this surface for those formats. For **mp3** (and
other lossy formats), demucs-infer deliberately does *not* silently switch
to `soundfile` (see "torchaudio 2.11+ and audio decoders" above for why),
so if FFmpeg also isn't available you'll see this error for real -- either
`pip install demucs-infer[torchcodec]`, pin `torchaudio<2.11`, install
FFmpeg, or convert the file to wav/flac.

## What This Project Will NEVER Bundle

demucs-infer downloads pretrained model weights at runtime; it does not, and
will never, ship them inside the git repository or the published package.

- **No weight files in the repo or the PyPI package.** `.th` checkpoints are
  never committed to git and never included in the sdist/wheel — the package
  is source code only.
- **No re-hosted or mirrored weights.** Official models are fetched from
  Meta's CDN; compatible models are fetched from their registry-recorded
  original public sources. demucs-infer does not run its own mirror or CDN.
- **No silently altered or re-derived weights.** What you get from the
  official checkpoint URLs is bit-for-bit what upstream Demucs produced;
  this package does not quantize, prune, or fine-tune models and ship the
  result as a default.
- **Compatible models remain separate artifacts.** Selecting a registry-backed
  DrumSep, UVR, CDX23, or MSST model downloads its exact source artifact on
  first use and verifies the full SHA-256. The bytes are never bundled.

Default cache location (where downloaded weights land on disk):

```
~/.cache/demucs-infer/
```

Override it per session with `cache_dir`:

```python
from pathlib import Path
from demucs_infer import DemucsSession

session = DemucsSession(
    model="uvr_demucs_model_1",
    cache_dir=Path("/srv/demucs-checkpoints"),
)
```

### Manual checkpoint download

For offline or air-gapped use, download each required URL ahead of time and
save it under the exact cache filename shown below. Use the default directory
above or the directory passed as `cache_dir`. Multi-component bags require
every row listed for that recipe.

| Used by | Save as | Exact source URL |
|---------|---------|------------------|
| `uvr_demucs_model_1`, `uvr_demucs_model_bag` | `ebf34a2db.th` | https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/ebf34a2db.th |
| `uvr_demucs_model_2`, `uvr_demucs_model_bag` | `ebf34a2d.th` | https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/ebf34a2d.th |
| `cdx23_dnr` component A | `cdx23_dnr_a-a778de4a.th` | https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-a778de4a.th |
| `cdx23_dnr` component B | `cdx23_dnr_b-dbb4db15.th` | https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-dbb4db15.th |
| `cdx23_dnr` component C | `cdx23_dnr_c-e41a5468.th` | https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-e41a5468.th |
| `msst_htdemucs_vocals` | `model_vocals_htdemucs_sdr_8.78.ckpt` | https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt |

The registry verifies every file before loading. A wrong or incomplete file is
rejected rather than used. Official Meta and DrumSep artifacts follow the same
resolver; inspect `demucs_infer/config/checkpoints.toml` for their exact
filenames, URLs, and hashes. If a live download fails, check the source host,
internet connection, and firewall settings.

## Development

### Development Installation

**With UV:**
```bash
cd /path/to/demucs-infer
uv pip install -e ".[dev]"
# Or add to your project as editable dependency
uv add -e ../path/to/demucs-infer
```

**With pip:**
```bash
cd /path/to/demucs-infer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

The package includes a comprehensive test suite using pytest:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_import.py -v

# Run with coverage
uv run pytest tests/ --cov=demucs_infer

# Run network tests too (checkpoint URL liveness; deselected by default)
uv run pytest tests/ -v -m "network"
```

See [CLAUDE.md](CLAUDE.md) for the bit-for-bit accuracy gate that any change
to `demucs_infer/` must pass.

**Continuous Integration:**
- GitHub Actions runs the test suite as a release gate: nothing publishes to PyPI without it passing first (`.github/workflows/publish.yml`)
- Tests validate both library API and CLI commands
- Python 3.10 with PyTorch 2.x compatibility verified

### Documentation

- **[Migration Guide](docs/MIGRATION.md)** - Migrate from original Demucs
- **[Implementation Notes](docs/dev/IMPLEMENTATION_NOTES.md)** - Technical details
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[Test Examples](tests/test_import.py)** - Import verification

## License

**MIT License** (same as original Demucs)

Copyright (c) Meta Platforms, Inc. (Original Demucs)
Copyright (c) 2025 (demucs-infer modifications)

See [LICENSE](LICENSE) for details.

## Support

- **Migration Help**: See [MIGRATION.md](docs/MIGRATION.md)
- **Version History**: See [CHANGELOG.md](CHANGELOG.md)
- **Bug Reports**: [GitHub Issues](https://github.com/openmirlab/demucs-infer/issues)
- **Original Demucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)

---

Made for the ML community. Based on the excellent work by Alexandre Défossez and Meta AI Research.
