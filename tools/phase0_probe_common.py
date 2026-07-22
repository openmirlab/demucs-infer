"""Pinned artifact resolution and deterministic evidence helpers for Phase 0.

The probe scripts share this maintenance-only module so URL, checksum, cache,
and float-digest policy have one owner. It never participates in package
runtime loading and never stores checkpoint or audio bytes in the repository.

Reads: public HTTPS checkpoint sources, caller-selected cache directory.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Artifact:
    key: str
    filename: str
    url: str
    sha256: str


ARTIFACTS = (
    Artifact(
        "uvr_model_2",
        "ebf34a2d.th",
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/ebf34a2d.th",
        "0556f2145e97e96d3c2a33c3879711e7559d7d261fd9773b20e2c580a3beffd0",
    ),
    Artifact(
        "uvr_model_1",
        "ebf34a2db.th",
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/ebf34a2db.th",
        "dbdd521017d706829716055df8a86e7ee5d05a590508488c82ffb551c6edb918",
    ),
    Artifact(
        "cdx23_a",
        "97d170e1-a778de4a.th",
        "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-a778de4a.th",
        "a778de4a72b90482a578bfb6ea6bf41462785d9136c11802c67a864a40f29434",
    ),
    Artifact(
        "cdx23_b",
        "97d170e1-dbb4db15.th",
        "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-dbb4db15.th",
        "dbb4db154df7e45a5cb72d1659c48937e757f6d6b0eef8ca4199e6e38f8d8f37",
    ),
    Artifact(
        "cdx23_c",
        "97d170e1-e41a5468.th",
        "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-e41a5468.th",
        "e41a54684d3cd6794ee2bb59183ffeb11a9a3d42db873a6a08a9829ac3ef4cfe",
    ),
    Artifact(
        "msst_htdemucs_vocals",
        "model_vocals_htdemucs_sdr_8.78.ckpt",
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt",
        "0ea6e9495685045e6b4e66174131be5d19808bb0d6d1a1ba717d238f9380e8d0",
    ),
    Artifact(
        "msst_htdemucs_config",
        "config_vocals_htdemucs.yaml",
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_htdemucs.yaml",
        "3e74bc81f35f198297b82e1a16a7736375af4290db54afa048126591b220b29c",
    ),
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(artifact: Artifact, cache_dir: Path, *, offline: bool) -> Path:
    """Return verified cache bytes, downloading atomically unless offline."""
    path = cache_dir / artifact.filename
    if path.is_file():
        actual = sha256_file(path)
        if actual == artifact.sha256:
            return path
        raise ValueError(f"{artifact.filename}: expected {artifact.sha256}, got {actual}")
    if offline:
        raise FileNotFoundError(f"offline cache is missing {path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(artifact.url) as source:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_dir, prefix=f".{artifact.filename}.", suffix=".part", delete=False
            ) as target:
                temporary = Path(target.name)
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        actual = sha256_file(temporary)
        if actual != artifact.sha256:
            raise ValueError(f"{artifact.filename}: expected {artifact.sha256}, got {actual}")
        os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def deterministic_audio(samples: int = 16384) -> torch.Tensor:
    """Build fixed stereo float32 input without depending on global RNG state."""
    generator = torch.Generator(device="cpu").manual_seed(20260722)
    noise = torch.randn((1, 2, samples), generator=generator, dtype=torch.float32) * 0.005
    time = torch.arange(samples, dtype=torch.float32) / 44100
    tone = 0.02 * torch.sin(2 * torch.pi * 220 * time) + 0.01 * torch.sin(2 * torch.pi * 997 * time)
    noise[:, 0] += tone
    noise[:, 1] += 0.7 * tone
    return noise


def tensor_evidence(tensor: torch.Tensor) -> dict[str, object]:
    """Record an environment-guarded float digest plus diagnostic moments."""
    value = tensor.detach().cpu().contiguous().to(torch.float32)
    raw = value.numpy().tobytes()
    flat = value.flatten()
    return {
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "max_abs": float(value.abs().max()),
        "first_16": [float(item) for item in flat[:16]],
    }


def environment_metadata(device: str) -> dict[str, object]:
    concrete = torch.device(device)
    return {
        "torch_version": torch.__version__,
        "device": str(concrete),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(concrete) if concrete.type == "cuda" else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "input_seed": 20260722,
        "input_samples": 16384,
    }
