"""Compare trusted native HTDemucs .th checkpoints with the safetensors loader.

Temporary serialization is verification-only: no converted weights are published
or retained. Both arms use the public session on real audio, padded audio,
silence and a deterministic synthetic input; every output is compared in memory.
Reads: states.load_model, safetensors.load_safetensors_model, DemucsSession.
"""

import argparse
import gc
import hashlib
import json
import random
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

from demucs_infer import DemucsSession
from demucs_infer.htdemucs import HTDemucs
from demucs_infer.safetensors import load_safetensors_model
from demucs_infer.states import load_model


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def encode(value):
    if isinstance(value, Fraction):
        return {"_type": "fraction", "numerator": value.numerator,
                "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def run_arm(path, checksum, inputs, rate, device):
    outputs = {}
    with DemucsSession(checkpoint_path=path, checkpoint_sha256=checksum,
                       device=device, shifts=1, split=True, overlap=.25,
                       jobs=0, progress=False) as session:
        for name, wave in inputs.items():
            random.seed(1234)
            np.random.seed(42)
            torch.manual_seed(42)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(42)
            with torch.no_grad():
                mixture, stems = session.infer(wave.clone(), sample_rate=rate)
            outputs[name] = {key: value.detach().cpu().clone()
                             for key, value in {"mixture": mixture, **stems}.items()}
    return outputs


def verify(source, expected_digest, inputs, rate, device):
    if digest(source) != expected_digest:
        raise ValueError(f"Trusted source checksum mismatch: {source}")
    reference = load_model(source, strict=True)
    assert type(reference) is HTDemucs, type(reference)
    args, kwargs = reference._init_args_kwargs
    with tempfile.TemporaryDirectory(prefix="demucs-safe-parity-") as temporary:
        safe = Path(temporary) / "reference.safetensors"
        save_file({key: value.detach().cpu().contiguous().clone()
                   for key, value in reference.state_dict().items()}, str(safe),
                  metadata={"klass": "demucs_infer.htdemucs.HTDemucs",
                            "args": json.dumps(encode(args), allow_nan=False),
                            "kwargs": json.dumps(encode(kwargs), allow_nan=False)})
        candidate = load_safetensors_model(safe)
        assert encode(candidate._init_args_kwargs) == encode(reference._init_args_kwargs)
        assert candidate.sources == reference.sources
        assert candidate.samplerate == reference.samplerate
        left, right = reference.state_dict(), candidate.state_dict()
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].dtype == right[key].dtype, key
            assert torch.equal(left[key], right[key]), key
        del reference, candidate, left, right
        gc.collect()
        old = run_arm(source, expected_digest, inputs, rate, device)
        new = run_arm(safe, digest(safe), inputs, rate, device)

    rows = []
    for name in inputs:
        assert old[name].keys() == new[name].keys()
        for stem, expected in old[name].items():
            actual = new[name][stem]
            assert actual.shape == expected.shape
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            error = (actual - expected).double()
            scale = expected.double().square().mean().sqrt().item()
            rmse = error.square().mean().sqrt().item()
            row = dict(input=name, stem=stem, shape=list(actual.shape),
                       max_abs=error.abs().max().item(), rmse=rmse,
                       reference_rms=scale, relative_rmse=rmse / max(scale, 1e-12),
                       exact=torch.equal(actual, expected))
            rows.append(row)
            assert row["exact"], row
    assert any(row["reference_rms"] > 1e-6 for row in rows
               if row["input"] == "music" and row["stem"] != "mixture"), "Trivial output"
    return dict(checkpoint=str(source), sha256=expected_digest, outputs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="PATH=SHA256", help="Trusted .th file and independently known digest")
    parser.add_argument("--audio", type=Path, required=True, help="Real music, at least 10 seconds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if rate != 44100 or audio.shape[1] != 2 or len(audio) < rate * 10:
        raise ValueError("Provide at least 10 seconds of stereo 44.1kHz real music")
    music = torch.from_numpy(audio[:rate * 10].T.copy())
    inputs = {"music": music,
              "music_silent_tail": torch.cat([music, torch.zeros(2, rate)], dim=1),
              "silence": torch.zeros_like(music),
              "synthetic": torch.randn(music.shape, generator=torch.Generator().manual_seed(42)) * .05}
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, device=args.device,
                  audio=str(args.audio), audio_sha256=digest(args.audio), checkpoints=[])
    if args.device.startswith("cuda"):
        report["hardware"] = torch.cuda.get_device_name(torch.device(args.device))
    for value in args.checkpoint:
        path, checksum = value.rsplit("=", 1)
        result = verify(Path(path), checksum, inputs, rate, args.device)
        report["checkpoints"].append(result)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {path}: metadata/state and all {len(result['outputs'])} outputs exact", flush=True)


if __name__ == "__main__":
    main()
