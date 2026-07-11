# demucs-infer

**Inference-only distribution of Demucs for PyTorch 2.x**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyPI](https://img.shields.io/pypi/v/demucs-infer)](https://pypi.org/project/demucs-infer/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cWELsXIFuqeEHStAKm8oP-1kME7Jfd4-)

High-quality audio source separation models for extracting vocals, drums, bass, and other instruments from music tracks.

---

## 📌 Overview

**demucs-infer** is a streamlined, inference-only version of [Demucs](https://github.com/facebookresearch/demucs) by Meta AI Research, optimized for PyTorch 2.x with minimal dependencies.

### 🎯 Key Features

- **PyTorch 2.x Support**: Compatible with modern PyTorch versions (no `torchaudio<2.1` restriction)
- **Inference-Only**: ~50% smaller than original package (removed training code)
- **Minimal Dependencies**: 7 core packages (vs 15+ in original)
- **API Compatible**: Drop-in replacement for inference workflows
- **Same Quality**: Zero changes to separation algorithms
- **All Models Supported**: HTDemucs, MDX, and all variants
- **Model Info API**: Query model capabilities, separation types, and source translations
- **Third-Party Model Support**: Compatible with community models (drumsep, cinematic, etc.)

---

## Try it in Colab

No installation needed! Try the demo directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cWELsXIFuqeEHStAKm8oP-1kME7Jfd4-)

---

## 🙏 Acknowledgments

### Original Research by Alexandre Défossez and Meta AI Research

**demucs-infer** is built upon the groundbreaking work of [Demucs](https://github.com/facebookresearch/demucs) by **Alexandre Défossez** and **Meta AI Research**. The original Demucs represents a major advancement in music source separation, achieving state-of-the-art results through innovative hybrid architectures and transformer-based approaches.

### Research Papers

The models in this package are based on two pioneering research papers:

#### Hybrid Demucs (2021)
**[Hybrid Spectrogram and Waveform Source Separation](https://arxiv.org/abs/2111.03600)**

This seminal work introduced the hybrid time-frequency domain approach that significantly improved separation quality by combining the strengths of both spectrogram and waveform-based processing.

```bibtex
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

#### Hybrid Transformer Demucs (2022)
**[Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)**

This follow-up research integrated transformer architectures into the hybrid approach, further pushing the boundaries of separation quality and establishing new benchmarks in the field.

```bibtex
@article{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  journal={arXiv preprint arXiv:2211.08553},
  year={2022}
}
```

### Citation

**If you use demucs-infer in your research, please cite the original Demucs papers above.** This package is merely a maintenance fork to ensure continued compatibility with modern PyTorch versions - all credit for the models, algorithms, and research belongs to the original authors.

### About This Fork

> **Note**: The original Demucs repository is no longer actively maintained by Meta AI Research. This package was created to continue the excellent work by providing ongoing maintenance and PyTorch 2.x compatibility for the inference capabilities, while preserving 100% of the original model quality and algorithms.

**What we maintain:**
- PyTorch 2.x compatibility
- Modern dependency management
- Inference-only packaging

**What remains unchanged:**
- All model architectures (100% original)
- All separation algorithms (100% original)
- All model weights (100% original)
- Audio quality (100% identical to original)

---

## 🚀 Quick Start

### Installation

demucs-infer is available on [PyPI](https://pypi.org/project/demucs-infer/) and supports both **UV** (recommended, faster) and **pip** (traditional) installation methods.

#### Option 1: UV (Recommended) ⚡

[UV](https://github.com/astral-sh/uv) is a blazing-fast Python package installer and resolver.

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to existing project
uv add demucs-infer

# Or create new project with demucs-infer
uv init my-audio-project
cd my-audio-project
uv add demucs-infer

# Run Python with demucs-infer available
uv run python your_script.py
```

**Benefits of UV:**
- ⚡ 10-100x faster than pip
- 🔒 Automatic virtual environment management
- 📦 Consistent dependency resolution
- 🎯 Works seamlessly with PyPI packages

#### Option 2: pip (Traditional)

```bash
# Install in current environment
pip install demucs-infer

# Or create virtual environment first (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install demucs-infer
```

### Python API

```python
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio
import torch
import torchaudio

# Load model
model = get_model("htdemucs_ft")
model.eval()

# Load audio
wav, sr = torchaudio.load("song.wav")
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

### CLI

**With UV:**
```bash
# Basic usage
uv run demucs-infer "song.wav"

# Extract specific stems (drums only)
uv run demucs-infer --two-stems=drums "song.wav"

# Use specific model
uv run demucs-infer -n htdemucs_ft "song.wav"

# Specify output directory
uv run demucs-infer -o output/ "song.wav"
```

**With pip:**
```bash
# Basic usage
demucs-infer "song.wav"

# Extract specific stems (drums only)
demucs-infer --two-stems=drums "song.wav"

# Use specific model
demucs-infer -n htdemucs_ft "song.wav"

# Specify output directory
demucs-infer -o output/ "song.wav"
```

---

## 📦 Why demucs-infer?

The original [Demucs](https://github.com/facebookresearch/demucs) repository is **no longer actively maintained** by Meta AI Research. While the models remain state-of-the-art, the package has not received updates for modern PyTorch versions.

**demucs-infer** was created to:

1. **Maintain compatibility** - Keep working with PyTorch 2.x and Python 3.10+
2. **Continue development** - Address issues and improve user experience
3. **Focus on inference** - Remove training code for a leaner package
4. **Serve the community** - Ensure researchers and developers can keep using these excellent models

### Comparison with Original Demucs

| Feature | Original Demucs | demucs-infer |
|---------|----------------|--------------|
| **Maintenance Status** | ⚠️ No longer actively maintained | ✅ Active |
| **PyTorch Support** | 1.8.x - 2.0.x (with `torchaudio<2.1`) | 2.0+ (no restrictions) ✅ |
| **Package Size** | ~Full codebase | ~50% smaller ✅ |
| **Dependencies** | 15+ packages | 7 core packages ✅ |
| **Training Code** | ✅ Included | ❌ Removed (inference-only) |
| **Inference Code** | ✅ Included | ✅ Included |
| **CLI Command** | `demucs` | `demucs-infer` (no conflicts) |
| **Import Name** | `demucs` | `demucs_infer` (no conflicts) |
| **Model Weights** | ✅ Same repositories | ✅ Same repositories |
| **Audio Quality** | ✅ High quality | ✅ Same quality (zero algorithm changes) |

---

## 🎵 Available Models

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

### Usage

```python
# Load specific model
model = get_model("htdemucs_ft")  # Best quality
model = get_model("mdx")          # Faster
model = get_model("htdemucs_6s")  # 6 sources
```

---

## 💡 Use Cases

### Music Production
- Extract vocals for remixing
- Isolate drums for sampling
- Remove vocals for karaoke tracks
- Separate instruments for analysis

### Machine Learning
- Prepare training data for music ML models
- Audio preprocessing for downstream tasks
- Dataset augmentation

### Research
- Music information retrieval (MIR)
- Audio signal processing research
- Music transcription

---

## 📋 Model Info API

Query model capabilities, separation types, and get source name translations programmatically.

### Get Model Information

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

### List Supported Separation Types

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

### ModelInfo Properties

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

---

## 🔧 Advanced Usage

### Two-Stems Separation (Faster)

```python
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
import torchaudio

model = get_model("htdemucs_ft").cuda().eval()

audio_files = list(Path("input/").glob("*.wav"))

for audio_file in audio_files:
    wav, sr = torchaudio.load(str(audio_file))
    wav = wav.unsqueeze(0).cuda()

    with torch.no_grad():
        sources = apply_model(model, wav, device="cuda")

    output_dir = Path("output") / audio_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, source_name in enumerate(model.sources):
        save_audio(sources[0, i].cpu(), output_dir / f"{source_name}.wav", sr)

    print(f"✅ Processed: {audio_file.name}")
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

---

## 📚 Documentation

- **[Migration Guide](docs/MIGRATION.md)** - Migrate from original Demucs
- **[Implementation Notes](docs/dev/IMPLEMENTATION_NOTES.md)** - Technical details
- **[Test Examples](tests/test_import.py)** - Import verification

---

## 🛠 Dependencies

### Core Dependencies (7 packages)

```toml
torch>=2.0.0
torchaudio>=2.0.0
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

### Optional Dependencies

**With UV:**
```bash
# For MP3 output support
uv add "demucs-infer[mp3]"

# For quantized models
uv add "demucs-infer[quantized]"

# For downloading community models (Google Drive)
uv add "demucs-infer[community]"

# Or install all optional features
uv add "demucs-infer[mp3,quantized,community]"
```

**With pip:**
```bash
# For MP3 output support
pip install demucs-infer[mp3]  # Adds: lameenc>=1.2

# For quantized models
pip install demucs-infer[quantized]  # Adds: diffq>=0.2.1

# For downloading community models (Google Drive)
pip install demucs-infer[community]  # Adds: gdown>=5.0.0

# Or install all optional features
pip install "demucs-infer[mp3,quantized,community]"
```

### Development Installation

**With UV:**
```bash
# Install in editable mode from local directory
cd /path/to/demucs-infer
uv pip install -e ".[dev]"

# Or add to your project as editable dependency
uv add -e ../path/to/demucs-infer
```

**With pip:**
```bash
# Install in editable mode from local directory
cd /path/to/demucs-infer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
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

**Continuous Integration:**
- GitHub Actions runs the test suite as a release gate: nothing publishes to PyPI without it passing first (`.github/workflows/publish.yml`)
- Tests validate both library API and CLI commands
- Python 3.10 with PyTorch 2.x compatibility verified

---

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0 or later
- **OS**: Linux, macOS, Windows
- **GPU**: Optional (CUDA-capable GPU recommended for speed)

---

## 🔍 What demucs-infer Changes

### ✅ What We Built

- **PyTorch 2.x compatibility layer** - Removed version restrictions
- **PyTorch 2.6+ support** - Compatible with `weights_only` default changes
- **Minimal logging module** - Replaced dora-search dependency
- **Lazy imports** - Made optional dependencies truly optional
- **Inference-only packaging** - Removed training code
- **Clean dependency tree** - 7 core packages instead of 15+
- **Model Info API** - Query model capabilities, separation types, and source translations
- **Third-party model support** - Module aliasing for community-trained models

### ✅ What Stays Unchanged

- ✅ **All separation models** - HTDemucs, MDX, all variants
- ✅ **Model architectures** - Zero modifications to neural networks
- ✅ **Separation algorithms** - Identical audio processing
- ✅ **Model weights** - Same pretrained checkpoints
- ✅ **Audio quality** - 100% identical output

### ❌ What's Not Included

- ❌ Training code (`train.py`, `solver.py`, etc.)
- ❌ Evaluation scripts (`evaluate.py`)
- ❌ Training dependencies (hydra, dora-search, omegaconf)
- ❌ Dataset utilities (musdb, museval)
- ❌ Distributed training tools (submitit)

---

## 📊 Package Comparison

| Metric | Original Demucs | demucs-infer | Improvement |
|--------|----------------|--------------|-------------|
| **Python Files** | 36+ files | 17 files | ~47% smaller |
| **Core Dependencies** | 15+ packages | 7 packages | ~53% fewer |
| **PyTorch Restriction** | `torchaudio<2.1` ❌ | No restriction ✅ | Flexible |
| **Training Code** | Included | Removed | Focused |
| **Inference Quality** | High | **Same** ✅ | Identical |

---

## 🐛 Troubleshooting

### ImportError: No module named 'demucs_infer'

**With UV:**
```bash
# Make sure you added demucs-infer to your project
uv add demucs-infer

# Or run with UV
uv run python your_script.py
```

**With pip:**
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

### Model Download Issues

```python
# Models are downloaded from official Demucs repositories
# Check internet connection and firewall settings

# Default model cache location:
# Linux: ~/.cache/torch/hub/checkpoints/
# macOS: ~/Library/Caches/torch/hub/checkpoints/
# Windows: %USERPROFILE%\.cache\torch\hub\checkpoints\
```

---

## 📄 License

**MIT License** (same as original Demucs)

Copyright (c) Meta Platforms, Inc. (Original Demucs)
Copyright (c) 2025 (demucs-infer modifications)

See [LICENSE](LICENSE) for details.

--

---

## 📞 Support

- **Migration Help**: See [MIGRATION.md](docs/MIGRATION.md)
- **Original Demucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)

---

**Made with ❤️ for the ML community**

Based on the excellent work by Alexandre Défossez and Meta AI Research.
