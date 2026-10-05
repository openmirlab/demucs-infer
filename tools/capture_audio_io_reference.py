"""Capture the audio I/O reference using the unmodified v4.3.0 package.

Run this script with PYTHONPATH pointed at commit 4c4a623e (v4.3.0) and
Torch 2.8.0+cu128. It generates its own licensed audio, records decoded PCM
and real HTDemucs outputs, and never changes the reference in place.

Reads: demucs_infer.api (reader override), demucs_infer.audio (save_audio),
demucs_infer.clean_api (DemucsSession)
"""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import demucs_infer
import demucs_infer.api as api
from demucs_infer import DemucsSession
from demucs_infer.audio import save_audio


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def capture(destination):
    if demucs_infer.__version__ != "4.3.0":
        raise RuntimeError("Capture requires the unmodified v4.3.0 package")
    destination.mkdir(parents=True, exist_ok=True)
    sr = 44100
    size = sr
    rng = np.random.default_rng(20261005)
    time = np.arange(size, dtype=np.float64) / sr
    source = np.stack([
        .22 * np.sin(2 * np.pi * 220 * time)
        + .08 * np.sin(2 * np.pi * 880 * time)
        + rng.normal(0, .01, size),
        .19 * np.sin(2 * np.pi * 330 * time)
        + .05 * np.sin(2 * np.pi * 1760 * time)
        + rng.normal(0, .01, size),
    ]).astype(np.float32)
    source[:, int(.75 * sr):] = 0
    source[:, 0:12] = np.array([
        [-1, -.5, -1 / 32768, -.5 / 32768, 0, .5 / 32768,
         1 / 32768, .5, .9, 1, 0, 0],
    ] * 2, dtype=np.float32)
    np.save(destination / "audio_io_generated.npy", source)
    source_wav = destination / "source.wav"
    sf.write(source_wav, source.T, sr, subtype="PCM_16")
    mp3 = destination / "audio_io_generated.mp3"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(source_wav),
        "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3),
    ], check=True)

    pcm = {}
    for ext, bits, as_float in [
        ("wav", 16, False), ("wav", 24, False),
        ("wav", 32, False), ("wav", 32, True),
        ("flac", 16, False), ("flac", 24, False),
    ]:
        key = "{}_{}_{}".format(ext, bits, "float" if as_float else "pcm")
        path = destination / (key + "." + ext)
        save_audio(torch.from_numpy(source.copy()), path, sr,
                   clip="none", bits_per_sample=bits, as_float=as_float)
        dtype = "float32" if as_float else ("int16" if bits == 16 else "int32")
        decoded, _ = sf.read(path, dtype=dtype, always_2d=True)
        pcm[key] = {"dtype": dtype, "shape": list(decoded.shape),
                    "sha256": digest(decoded)}
        path.unlink()

    arrays = {}
    with DemucsSession(model="htdemucs", device="cpu") as session:
        def no_ffmpeg(self, *args, **kwargs):
            raise FileNotFoundError("forced unavailable FFmpeg executable")

        original_read = api.AudioFile.read
        api.AudioFile.read = no_ffmpeg
        try:
            random.seed(1234)
            with torch.no_grad():
                mixture, stems = session.infer(mp3)
        finally:
            api.AudioFile.read = original_read
    arrays["mp3_mixture"] = mixture.detach().cpu().numpy()
    for name, value in stems.items():
        arrays["mp3_" + name] = value.detach().cpu().numpy()
    np.savez_compressed(destination / "audio_io_mp3_v430.npz", **arrays)
    metadata = {
        "source": "Generated stereo tones/noise with a 0.25-second silent tail; no third-party audio.",
        "recorded_from": "demucs-infer v4.3.0 / TorchAudio 2.8.0+cu128 / soundfile 0.14.0",
        "sample_rate": sr,
        "input_sha256": digest(source),
        "pcm": pcm,
        "mp3_reference": {
            "fixture": "audio_io_mp3_v430.npz",
            "decoder": "TorchAudio 2.8.0 FFmpeg backend with system ffmpeg executable forced unavailable",
            "fields": {
                key: {"shape": list(value.shape), "sha256": digest(value)}
                for key, value in arrays.items()
            },
        },
    }
    (destination / "audio_io_reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    source_wav.unlink()
    print("Captured reference in", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    capture(parser.parse_args().out)
