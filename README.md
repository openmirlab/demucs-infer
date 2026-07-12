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
| **Model weights** | Official Meta checkpoints | Same official checkpoints, same host |

## Acknowledgments

**demucs-infer** is built entirely on the research and models of the original Demucs project. All algorithms, model architectures, and pretrained weights originate there — this package only maintains the packaging and PyTorch 2.x compatibility layer around them.

- **Upstream organization:** [Meta AI Research (FAIR)](https://ai.meta.com/research/)
- **Individual authors:** Alexandre Défossez (both papers below, see [Citation](#citation)); Simon Rouard and Francisco Massa (Hybrid Transformer Demucs, co-authors — see [arXiv:2211.08553](https://arxiv.org/abs/2211.08553))
- **Source repository:** [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- **Pretrained weights host:** [dl.fbaipublicfiles.com/demucs](https://dl.fbaipublicfiles.com/demucs/) — Meta's own public checkpoint CDN; demucs-infer downloads directly from it and hosts no mirror

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
- **All Models Supported**: HTDemucs, MDX, and all variants
- **Model Info API**: Query model capabilities, separation types, and source translations
- **Third-Party Model Support**: Compatible with community models (drumsep, cinematic, etc.)

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
- Model weights: same pretrained checkpoints, same host
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

### Compatible Third-Party Models

demucs-infer supports loading community-trained Demucs models. Place `.th` model files in a local directory and use the `repo` parameter.

| Model Signature | Name | Separation Type | Sources |
|-----------------|------|-----------------|---------|
| `49469ca8` | [Drumsep](https://github.com/inagoy/drumsep) | Drum Kit | kick, snare, cymbals, toms |
| `97d170e1` | CDX23 Cinematic | Film/Video | dialog, music, sfx |
| `phantom_center` | Phantom Center Extractor | Stereo Center/Sides | similarity, difference |
| `ebf34a2d` | UVR Demucs Model 1 | Vocal/Instrumental | vocals, non_vocals |

```python
from pathlib import Path
from demucs_infer.pretrained import get_model

# Load third-party model from local directory
model = get_model("49469ca8", repo=Path("/path/to/models"))
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
# stereo_center: Stereo Center/Sides Separation
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
- **No re-hosted or mirrored weights.** Official models are always fetched
  directly from Meta's own CDN, [dl.fbaipublicfiles.com/demucs](https://dl.fbaipublicfiles.com/demucs/)
  — demucs-infer does not run its own mirror or CDN.
- **No silently altered or re-derived weights.** What you get from the
  official checkpoint URLs is bit-for-bit what upstream Demucs produced;
  this package does not quantize, prune, or fine-tune models and ship the
  result as a default.
- **Community/third-party models stay opt-in and user-supplied.** Models
  like Drumsep or UVR variants (see [Available Models](#available-models))
  are loaded only from a local directory or Google Drive link you provide
  via the optional `[community]` extra — never bundled or auto-downloaded
  by default.

Default cache location (where downloaded weights land on disk):

```
# Linux:   ~/.cache/torch/hub/checkpoints/
# macOS:   ~/Library/Caches/torch/hub/checkpoints/
# Windows: %USERPROFILE%\.cache\torch\hub\checkpoints\
```

If a download fails, check your internet connection and firewall settings —
these are live fetches from Meta's CDN, not files shipped with the package.

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
