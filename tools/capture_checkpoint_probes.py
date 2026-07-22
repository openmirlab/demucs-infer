"""Capture or compare Phase 0 evidence for UVR, CDX23, and MSST weights.

Every artifact is resolved through pinned public URLs and full SHA-256 values.
The probe uses the current direct loader seam for exports, reconstructs MSST's
HTDemucs from its release config, and records small deterministic outputs.

Reads: phase0_probe_common, demucs_infer.states, demucs_infer model classes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from demucs_infer import __version__  # noqa: E402
from demucs_infer.apply import BagOfModels, apply_model  # noqa: E402
from demucs_infer.htdemucs import HTDemucs  # noqa: E402
from demucs_infer.states import load_model  # noqa: E402
from phase0_probe_common import (  # noqa: E402
    ARTIFACTS,
    deterministic_audio,
    environment_metadata,
    resolve_artifact,
    tensor_evidence,
)


def model_metadata(model) -> dict[str, object]:
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "sources": list(model.sources),
        "samplerate": model.samplerate,
        "audio_channels": model.audio_channels,
        "segment": str(model.segment) if hasattr(model, "segment") else None,
    }


def run_export_family(keys: list[str], paths: dict[str, Path], audio: torch.Tensor, device: str) -> dict:
    models = [load_model(paths[key]) for key in keys]
    components = {}
    for key, model in zip(keys, models):
        output = apply_model(model, audio, shifts=0, split=False, device=device)
        components[key] = {"model": model_metadata(model), "output": tensor_evidence(output)}
    bag = BagOfModels(models)
    bag_output = apply_model(bag, audio, shifts=0, split=False, device=device)
    bag_meta = model_metadata(bag)
    bag_meta["component_order"] = keys
    bag_meta["weights"] = bag.weights
    return {
        "loader": "demucs_infer.states.load_model",
        "components": components,
        "bag": {"model": bag_meta, "output": tensor_evidence(bag_output)},
    }


def load_msst(paths: dict[str, Path]):
    config = yaml.safe_load(paths["msst_htdemucs_config"].read_text(encoding="utf-8"))
    training = config["training"]
    allowed = inspect.signature(HTDemucs).parameters
    kwargs = {key: value for key, value in config["htdemucs"].items() if key in allowed}
    kwargs["dconv_init"] = float(kwargs["dconv_init"])
    kwargs.update(
        sources=list(training["instruments"]),
        samplerate=int(training["samplerate"]),
        audio_channels=int(training["channels"]),
        segment=int(training["segment"]),
    )
    model = HTDemucs(**kwargs)
    state = torch.load(paths["msst_htdemucs_vocals"], map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"MSST strict load mismatch: {incompatible}")
    return model, sorted(set(config["htdemucs"]) - set(kwargs))


def build_evidence(cache_dir: Path, *, offline: bool, device: str) -> dict[str, object]:
    paths = {artifact.key: resolve_artifact(artifact, cache_dir, offline=offline) for artifact in ARTIFACTS}
    audio = deterministic_audio()
    uvr = run_export_family(["uvr_model_2", "uvr_model_1"], paths, audio, device)
    cdx23 = run_export_family(["cdx23_a", "cdx23_b", "cdx23_c"], paths, audio, device)
    msst_model, ignored_config_keys = load_msst(paths)
    msst_output = apply_model(msst_model, audio, shifts=0, split=False, device=device)
    return {
        "schema_version": 1,
        "environment": environment_metadata(device) | {"demucs_infer_version": __version__},
        "probe_parameters": {"shifts": 0, "split": False, "sample_rate": 44100},
        "artifacts": {
            artifact.key: {"filename": artifact.filename, "url": artifact.url, "sha256": artifact.sha256}
            for artifact in ARTIFACTS
        },
        "uvr": uvr,
        "cdx23": cdx23,
        "msst": {
            "loader": "HTDemucs constructor plus raw state_dict",
            "model": model_metadata(msst_model),
            "strict_load": True,
            "ignored_non_constructor_config_keys": ignored_config_keys,
            "output": tensor_evidence(msst_output),
        },
    }


def compare_evidence(actual: dict, expected: dict) -> None:
    for key in ("schema_version", "probe_parameters", "artifacts"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Phase 0 discrete evidence changed: {key}")
    for family in ("uvr", "cdx23"):
        for key, value in expected[family]["components"].items():
            if actual[family]["components"][key]["model"] != value["model"]:
                raise AssertionError(f"{family}/{key} model metadata changed")
        if actual[family]["bag"]["model"] != expected[family]["bag"]["model"]:
            raise AssertionError(f"{family} bag recipe changed")
    for key in ("loader", "model", "strict_load", "ignored_non_constructor_config_keys"):
        if actual["msst"][key] != expected["msst"][key]:
            raise AssertionError(f"MSST {key} changed")

    environment_keys = (
        "torch_version", "device", "cuda_version", "cuda_device", "matmul_allow_tf32", "cudnn_allow_tf32"
    )
    if any(actual["environment"].get(key) != expected["environment"].get(key) for key in environment_keys):
        print("SKIP numeric digests: current torch/device environment differs from the recorded fixture")
        return
    for family in ("uvr", "cdx23"):
        for key, value in expected[family]["components"].items():
            if actual[family]["components"][key]["output"] != value["output"]:
                raise AssertionError(f"{family}/{key} output changed")
        if actual[family]["bag"]["output"] != expected[family]["bag"]["output"]:
            raise AssertionError(f"{family} bag output changed")
    if actual["msst"]["output"] != expected["msst"]["output"]:
        raise AssertionError("MSST output changed")
    print("numeric digests match the recorded environment exactly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--offline", action="store_true", help="forbid downloads and require verified cache bytes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if not args.out and not args.compare:
        parser.error("one of --out or --compare is required")

    evidence = build_evidence(args.cache_dir, offline=args.offline, device=args.device)
    if args.compare:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        compare_evidence(evidence, expected)
        print(f"Phase 0 checkpoint evidence matches {args.compare}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"wrote Phase 0 checkpoint evidence to {args.out}")


if __name__ == "__main__":
    main()
