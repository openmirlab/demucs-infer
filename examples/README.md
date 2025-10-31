# Examples

Example scripts demonstrating how to use `demucs-infer`.

## Available Examples

### 🎵 [basic_separation.py](basic_separation.py)
**Simple audio source separation**

The most straightforward way to separate audio into stems (drums, bass, other, vocals).

```bash
# Run with a test audio file named test.wav, audio.wav, or song.wav
uv run python examples/basic_separation.py
```

**What it does:**
- Loads a pretrained model (htdemucs_ft by default)
- Loads audio file
- Separates into 4 stems: drums, bass, other, vocals
- Saves stems to `output/` directory

**Use as a library:**
```python
from examples.basic_separation import separate_audio

separate_audio(
    input_file="my_song.wav",
    output_dir="stems",
    model_name="htdemucs_ft",
    device="cuda"  # or "cpu" or "auto"
)
```

## Running Examples

All examples should be run using UV:

```bash
# Basic usage
uv run python examples/basic_separation.py

# Or with specific Python
uv run python examples/basic_separation.py
```

## Creating Your Own Scripts

Use these examples as templates for your own audio separation scripts.

### Minimal Example

```python
import torch
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
import torchaudio

# Load model
model = get_model("htdemucs_ft").eval()

# Load audio
wav, sr = torchaudio.load("song.wav")
wav = wav.unsqueeze(0)  # Add batch dimension

# Separate
with torch.no_grad():
    sources = apply_model(model, wav, device="cpu")

# sources shape: [1, 4, channels, time]
# sources order: drums, bass, other, vocals
```

### Available Models

- `htdemucs_ft` - Best quality (recommended)
- `htdemucs` - High quality, slightly faster
- `htdemucs_6s` - 6 stems (adds guitar, piano)
- `mdx_extra` - Good quality, faster
- `mdx` - Fast
- `mdx_q`, `mdx_extra_q` - Quantized (very fast)

## Tips

### GPU vs CPU

GPU is much faster but requires CUDA:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Memory Management

For large files or limited memory:
- Use `--two-stems` for faster processing (CLI)
- Process in chunks
- Use CPU if GPU runs out of memory

### Two-Stem Separation (Faster)

To extract only one stem (e.g., vocals):
```bash
uv run demucs-infer --two-stems=vocals song.wav
```

## Need More?

See the main [README.md](../README.md) for:
- CLI usage
- API documentation
- Advanced features
- Troubleshooting

