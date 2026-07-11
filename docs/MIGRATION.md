# Migration Guide: Original Demucs → demucs-infer

## About Original Demucs

The original [Demucs](https://github.com/facebookresearch/demucs) by **Alexandre Défossez** and **Meta AI Research** represents groundbreaking work in music source separation. The research introduced innovative hybrid architectures and transformer-based approaches that achieved state-of-the-art results and have become foundational to the field.

**Key Research Contributions:**
- **Hybrid Demucs** (2021): Pioneered hybrid time-frequency domain processing
- **Hybrid Transformer Demucs** (2022): Integrated transformers for superior separation quality

All credit for the models, algorithms, and research belongs to the original authors. Please cite their papers when using demucs-infer in research.

## About demucs-infer

**demucs-infer** is a maintenance fork created because the original Demucs repository is **no longer actively maintained** by Meta AI Research. While the original package remains an excellent research contribution, it has not received updates for modern PyTorch versions (2.x).

This fork exists solely to:
- Maintain PyTorch 2.x compatibility
- Provide inference-only packaging for easier deployment
- Continue serving the research and developer community

**Important:** 100% of the model quality and algorithms come from the original Demucs research. demucs-infer only provides packaging and compatibility updates.

## Overview

This guide helps you migrate from the original Demucs package to demucs-infer.

---

## Why Migrate to demucs-infer?

| Original Demucs | demucs-infer |
|----------------|--------------|
| ⚠️ No longer maintained | ✅ Actively maintained |
| PyTorch 1.x only | PyTorch 2.x support ✅ |
| Training + Inference | Inference-only (smaller) |
| 15+ dependencies | 7 core dependencies |
| `torchaudio<2.1` restriction | No version restrictions ✅ |

---

## Installation

### Uninstall original Demucs

```bash
pip uninstall demucs
```

### Install demucs-infer

```bash
pip install demucs-infer
```

---

## Code Migration

### Python API

**Original Demucs:**
```python
import demucs.api
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio

separator = demucs.api.Separator(model="htdemucs")
origin, separated = separator.separate_audio_file("audio.wav")
```

**demucs-infer:**
```python
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio

# Option 1: Using lower-level API (same as original)
model = get_model("htdemucs")
sources = apply_model(model, wav_tensor)
save_audio(sources, "output/", sr=44100)

# Option 2: Using Separator API (if available)
# Same as original, just import from demucs_infer.api
```

### CLI

**Original Demucs:**
```bash
demucs "audio.wav"
demucs --two-stems=drums "audio.wav"
```

**demucs-infer:**
```bash
demucs-infer "audio.wav"
demucs-infer --two-stems=drums "audio.wav"
```

**Note:** CLI command name changed from `demucs` to `demucs-infer` to avoid conflicts.

---

## Import Changes

| Original | demucs-infer |
|----------|--------------|
| `from demucs.pretrained import get_model` | `from demucs_infer.pretrained import get_model` |
| `from demucs.apply import apply_model` | `from demucs_infer.apply import apply_model` |
| `from demucs.audio import save_audio` | `from demucs_infer.audio import save_audio` |
| `from demucs.api import Separator` | `from demucs_infer.api import Separator` (if available) |

---

## Compatibility

### ✅ Fully Compatible

- All model names (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx`, `mdx_extra`, etc.)
- All audio formats (WAV, MP3, FLAC)
- All model weights (downloads from same repositories)
- All separation algorithms (zero changes)

### ⚠️ Not Included

- Training code (`train.py`, `solver.py`, etc.) - inference-only
- Evaluation scripts (`evaluate.py`) - inference-only
- Command-line training utilities - inference-only

---

## Dependencies

### Removed (training-only)

- ❌ `dora-search` (training infrastructure)
- ❌ `hydra-core` (training configuration)
- ❌ `omegaconf` (training configuration)
- ❌ `submitit` (cluster training)
- ❌ `musdb`, `museval` (training datasets)

### Kept (inference)

- ✅ `torch>=2.0.0`
- ✅ `torchaudio>=2.0.0`
- ✅ `einops`
- ✅ `julius>=0.2.3`
- ✅ `numpy`
- ✅ `pyyaml`
- ✅ `tqdm`

### Vendored (no longer a dependency)

- `openunmix` -- only ever used for `openunmix.filtering.wiener` (optional
  Wiener-filter post-processing on some model configs); vendored directly
  into `demucs_infer/wiener.py` (MIT-licensed, with attribution) so the
  whole open-unmix-pytorch package no longer needs installing for one
  function.

### Optional

- `lameenc>=1.2` - Install with `pip install demucs-infer[mp3]`
- `diffq>=0.2.1` - Install with `pip install demucs-infer[quantized]`

---

## Example: Full Migration

**Before (original Demucs):**

```python
# requirements.txt
demucs

# your_script.py
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio
import torch
import torchaudio

model = get_model("htdemucs_ft")
model.eval()

wav, sr = torchaudio.load("song.wav")
wav = wav.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    sources = apply_model(model, wav, device="cuda")

# sources shape: [1, 4, channels, time]
# sources order: drums, bass, other, vocals
for i, source_name in enumerate(model.sources):
    source = sources[0, i]  # Remove batch dimension
    save_audio(source, f"output/{source_name}.wav", sr)
```

**After (demucs-infer):**

```python
# requirements.txt
demucs-infer

# your_script.py
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio
import torch
import torchaudio

model = get_model("htdemucs_ft")
model.eval()

wav, sr = torchaudio.load("song.wav")
wav = wav.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    sources = apply_model(model, wav, device="cuda")

# sources shape: [1, 4, channels, time]
# sources order: drums, bass, other, vocals
for i, source_name in enumerate(model.sources):
    source = sources[0, i]  # Remove batch dimension
    save_audio(source, f"output/{source_name}.wav", sr)
```

**Changes:**
- Line 2: `demucs` → `demucs-infer`
- Lines 5-7: `demucs` → `demucs_infer`
- Everything else: **identical** ✅

---

## FAQ

**Q: Will my existing models work?**
A: Yes! demucs-infer downloads the exact same model weights from the same repositories.

**Q: Will audio quality be the same?**
A: Yes! Zero changes to separation algorithms.

**Q: Can I train models with demucs-infer?**
A: No, demucs-infer is inference-only. Use original Demucs for training.

**Q: Can I use both demucs and demucs-infer?**
A: No, they will conflict. Choose one based on your needs:
- Original Demucs: Training + Inference, PyTorch 1.x
- demucs-infer: Inference-only, PyTorch 2.x

**Q: What about the Separator API?**
A: Check if `demucs_infer.api` module exists. If not, use the lower-level API shown above.

---

## Support

- **Original Demucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- **Documentation**: [docs/](.)

---

## License

MIT License (same as original Demucs)

See [LICENSE](../LICENSE) for full details including attribution to original authors.
