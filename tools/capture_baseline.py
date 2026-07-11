#!/usr/bin/env python3
"""ADOPT P0 -- seeded htdemucs separation baseline capture.

Runs htdemucs separation (via the package's primary inference API:
``demucs_infer.pretrained.get_model`` + ``demucs_infer.apply.apply_model``,
the same call pattern used by downstream callers such as
all-in-one-infer's ``DemucsProvider``) against one seeded synthetic stereo
clip and, if available, a real-audio excerpt. For each stem it records a
sha256 digest of the full float32 sample bytes plus the first/last 1000
samples, so later phases (vendoring openunmix's wiener, restructuring
files, ...) can re-run this script and diff the JSON to prove outputs
stayed bit-identical.

``apply_model(shifts=1)`` (the default, used here since production callers
never override it) draws its random time-shift offset from the *unseeded*
stdlib ``random`` module (see ``demucs_infer/apply.py``, ``random.randint``
in the shifts loop) -- so ``random.seed(1234)`` is pinned immediately
before every ``apply_model()`` call. Without that pin, re-running this
script would legitimately produce different (still valid) separations.

Usage:
    python tools/capture_baseline.py [--out PATH] [--device cuda|cpu]
                                      [--skip-real] [--model NAME]

Reads: demucs_infer (pretrained.get_model, apply.apply_model)
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demucs_infer.pretrained import get_model  # noqa: E402
from demucs_infer.apply import apply_model  # noqa: E402

SAMPLE_RATE = 44100
DURATION_S = 10.0
SEED_SHIFT = 1234  # pinned immediately before apply_model(), per known quirk
SEED_SYNTH = 42  # independent generator seed for the synthetic clip itself

REAL_AUDIO_DIR = Path(
    "/tmp/claude-1000/-home-worzpro-Desktop-dev-openmirlab-all-in-one-fix/"
    "352379ff-8ff0-4de6-99d7-032503d55b7e/scratchpad/perf/natten_accept/real"
)
REAL_AUDIO_FILE = "rick_astley_never_gonna_give_you_up.mp3"


def make_synthetic_clip(seed=SEED_SYNTH, duration=DURATION_S, sr=SAMPLE_RATE):
    """Deterministic ~10s stereo clip built from a dedicated numpy Generator
    (never touches the global `random` or torch RNGs, so it can't disturb
    the shifts-reproducibility seed pinned later)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr

    freqs = [110.0, 220.0, 330.0, 440.0, 550.0]
    left = np.zeros(n)
    right = np.zeros(n)
    for i, f in enumerate(freqs):
        amp = 0.15 / (i + 1)
        phase_l = rng.uniform(0, 2 * np.pi)
        phase_r = rng.uniform(0, 2 * np.pi)
        left += amp * np.sin(2 * np.pi * f * t + phase_l)
        right += amp * np.sin(2 * np.pi * f * t + phase_r)

    # light percussive envelope + noise floor so all 4 stems get *something*
    # to separate rather than a static tone
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    noise = rng.normal(0, 0.02, size=(2, n))
    left = left * env + noise[0]
    right = right * env + noise[1]

    wav = np.stack([left, right], axis=0).astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0.9:
        wav = wav * (0.9 / peak)
    return torch.from_numpy(wav)


def load_real_clip(path, duration=DURATION_S, sr=SAMPLE_RATE):
    wav, orig_sr = torchaudio.load(str(path))
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    n = int(duration * sr)
    if wav.shape[1] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[1]))
    else:
        wav = wav[:, :n]
    return wav.contiguous().to(torch.float32)


def digest_stem(tensor: torch.Tensor) -> dict:
    arr = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
    flat = arr.tobytes()
    sha = hashlib.sha256(flat).hexdigest()
    flat_view = arr.reshape(-1)
    return {
        "shape": list(arr.shape),
        "sha256": sha,
        "first_1000": [float(x) for x in flat_view[:1000]],
        "last_1000": [float(x) for x in flat_view[-1000:]],
    }


def separate(model, wav: torch.Tensor, device: str, overlap: float = 0.25) -> torch.Tensor:
    """Mirrors all-in-one-infer's DemucsProvider._run_demucs_separation:
    unsqueeze batch dim, move to device, apply_model with production
    defaults (overlap=0.25, shifts left at its default of 1, no explicit
    `segment=` -- see the known htdemucs segment= crash, out of scope
    here), squeeze batch dim back off, move to cpu."""
    wav_batch = wav.unsqueeze(0).to(device)
    random.seed(SEED_SHIFT)
    with torch.no_grad():
        sources = apply_model(model, wav_batch, device=device, overlap=overlap, progress=False)
    return sources.cpu().squeeze(0)


def build_fixture(model_name: str, device: str, skip_real: bool) -> dict:
    model = get_model(model_name)
    model.eval()

    clips = {"synthetic": make_synthetic_clip()}

    real_path = REAL_AUDIO_DIR / REAL_AUDIO_FILE
    if not skip_real and real_path.exists():
        clips["real_excerpt"] = load_real_clip(real_path)
    elif not skip_real:
        print(f"[capture_baseline] real audio not found at {real_path}, "
              f"skipping real_excerpt clip", file=sys.stderr)

    fixture = {
        "meta": {
            "model": model_name,
            "sources_order": list(model.sources),
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": DURATION_S,
            "shift_seed": SEED_SHIFT,
            "synth_seed": SEED_SYNTH,
            "apply_model_kwargs": {"overlap": 0.25, "shifts": "default(1)"},
            "torch_version": torch.__version__,
            "device": device,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "clips": {},
    }

    for clip_name, wav in clips.items():
        sources = separate(model, wav, device)
        stems = {}
        for i, stem_name in enumerate(model.sources):
            stems[stem_name] = digest_stem(sources[i])
        fixture["clips"][clip_name] = {"stems": stems}

    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                         default=REPO_ROOT / "tests" / "fixtures" / "baseline_htdemucs.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    fixture = build_fixture(args.model, args.device, args.skip_real)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"[capture_baseline] wrote {args.out}")
    for clip_name, clip in fixture["clips"].items():
        for stem_name, digest in clip["stems"].items():
            print(f"  {clip_name}/{stem_name}: sha256={digest['sha256'][:16]}...")


if __name__ == "__main__":
    main()
