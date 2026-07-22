"""Phase 3 registry surfaces and opt-in real-checkpoint parity gates.

Offline tests prove every new public name reaches the unified resolver. The
opt-in test uses the Phase 0 cache to verify all six weight files, strict-load
MSST, reproduce recorded outputs, and exercise each one-shot public name.

Reads: checkpoint runtime/config, Phase 0 probe fixture/tools, public APIs.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from demucs_infer.api import Separator, list_models
from demucs_infer.apply import BagOfModels, apply_model
from demucs_infer.checkpoint_catalog import checkpoint_config
from demucs_infer.checkpoint_runtime import CheckpointRuntime
from demucs_infer.clean_api import DemucsSession, separate_tensor
from demucs_infer.pretrained import get_model
from demucs_infer.separate import get_parser
from demucs_infer.states import load_model


REPO_ROOT = Path(__file__).parent.parent
TOOLS_ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import capture_checkpoint_probes  # noqa: E402
from phase0_probe_common import deterministic_audio, sha256_file  # noqa: E402


PUBLIC_NAMES = (
    "uvr_demucs_model_1",
    "uvr_demucs_model_2",
    "uvr_demucs_model_bag",
    "cdx23_dnr",
    "msst_htdemucs_vocals",
)
ARTIFACT_FILES = {
    "ebf34a2db": "ebf34a2db.th",
    "ebf34a2d": "ebf34a2d.th",
    "cdx23_dnr_a": "97d170e1-a778de4a.th",
    "cdx23_dnr_b": "97d170e1-dbb4db15.th",
    "cdx23_dnr_c": "97d170e1-e41a5468.th",
    "msst_htdemucs_vocals_state": "model_vocals_htdemucs_sdr_8.78.ckpt",
}


def _link_phase3_cache(source_root, runtime_root):
    registry = checkpoint_config()
    for artifact_id, source_name in ARTIFACT_FILES.items():
        source = source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        artifact = registry.artifacts[artifact_id]
        assert sha256_file(source) == artifact.sha256
        (runtime_root / artifact.path).symlink_to(source)


def _direct_eval_oracle(name, source_root):
    """Build an eval-mode oracle without using registry architecture/runtime loading."""
    artifact_path = {
        artifact_id: source_root / filename
        for artifact_id, filename in ARTIFACT_FILES.items()
    }
    if name == "uvr_demucs_model_1":
        return load_model(artifact_path["ebf34a2db"]).eval()
    if name == "uvr_demucs_model_2":
        return load_model(artifact_path["ebf34a2d"]).eval()
    if name == "uvr_demucs_model_bag":
        return BagOfModels([
            load_model(artifact_path["ebf34a2d"]).eval(),
            load_model(artifact_path["ebf34a2db"]).eval(),
        ]).eval()
    if name == "cdx23_dnr":
        return BagOfModels([
            load_model(artifact_path["cdx23_dnr_a"]).eval(),
            load_model(artifact_path["cdx23_dnr_b"]).eval(),
            load_model(artifact_path["cdx23_dnr_c"]).eval(),
        ]).eval()
    if name == "msst_htdemucs_vocals":
        model, ignored = capture_checkpoint_probes.load_msst({
            "msst_htdemucs_config": source_root / "config_vocals_htdemucs.yaml",
            "msst_htdemucs_vocals": artifact_path["msst_htdemucs_vocals_state"],
        })
        assert ignored == ["num_subbands"]
        return model.eval()
    raise AssertionError(f"missing direct oracle for {name}")


def test_new_public_names_are_listed_and_cli_selectable():
    models = list_models()
    for name in PUBLIC_NAMES:
        assert name in models["bag"]
        assert get_parser().parse_args(["-n", name, "song.wav"]).name == name


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_get_model_uses_registry_runtime_for_every_new_public_name(monkeypatch, name):
    sentinel = object()
    calls = []

    def load(runtime, resolution=None):
        calls.append(runtime.model_name)
        return sentinel

    monkeypatch.setattr(CheckpointRuntime, "load_registered_model", load)
    assert get_model(name) is sentinel
    assert calls == [name]


def test_closed_architecture_factory_rejects_unknown_recipe():
    class Unknown:
        name = "arbitrary.module.Class"

    with pytest.raises(ValueError, match="unsupported checkpoint architecture"):
        CheckpointRuntime._build_architecture(Unknown())


def test_phase3_real_checkpoints_match_phase0_and_public_helpers(monkeypatch, tmp_path):
    cache_value = os.environ.get("DEMUCS_PHASE0_PROBE_CACHE")
    if not cache_value:
        pytest.skip("set DEMUCS_PHASE0_PROBE_CACHE to run Phase 3 real-checkpoint parity")
    source_root = Path(cache_value)
    _link_phase3_cache(source_root, tmp_path)
    monkeypatch.setattr(
        CheckpointRuntime,
        "_cache_root",
        lambda runtime, create=True: tmp_path,
    )

    # All five package exports plus the raw MSST state dict load through their
    # declared format. The five demucs_package artifacts also remain directly
    # loadable through the frozen Meta-package loader.
    registry = checkpoint_config()
    for artifact_id in (
        "ebf34a2db", "ebf34a2d", "cdx23_dnr_a", "cdx23_dnr_b", "cdx23_dnr_c"
    ):
        model = load_model(tmp_path / registry.artifacts[artifact_id].path)
        assert list(model.sources) in (
            ["vocals", "non_vocals"],
            ["music", "sfx", "speech"],
        )
        del model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = deterministic_audio()
    expected = json.loads(
        (REPO_ROOT / "tests/fixtures/checkpoint_probe_baseline.json").read_text(encoding="utf-8")
    )
    parity = {
        "uvr_demucs_model_1": expected["uvr"]["components"]["uvr_model_1"]["output"],
        "uvr_demucs_model_2": expected["uvr"]["components"]["uvr_model_2"]["output"],
        "uvr_demucs_model_bag": expected["uvr"]["bag"]["output"],
        "cdx23_dnr": expected["cdx23"]["bag"]["output"],
        "msst_htdemucs_vocals": expected["msst"]["output"],
    }
    expected_sources = {
        "uvr_demucs_model_1": ["vocals", "non_vocals"],
        "uvr_demucs_model_2": ["vocals", "non_vocals"],
        "uvr_demucs_model_bag": ["vocals", "non_vocals"],
        "cdx23_dnr": ["music", "sfx", "speech"],
        "msst_htdemucs_vocals": ["vocals", "other"],
    }

    for name in PUBLIC_NAMES:
        model = get_model(name)
        assert list(model.sources) == expected_sources[name]
        output = apply_model(model, audio, shifts=0, split=False, device=device)
        assert list(output.shape) == parity[name]["shape"]
        assert torch.isfinite(output).all()

        oracle = _direct_eval_oracle(name, source_root)
        oracle_output = apply_model(oracle, audio, shifts=0, split=False, device=device)
        assert torch.equal(output, oracle_output)
        del model, output, oracle, oracle_output
        gc.collect()

        separator = Separator(model=name, device=device, shifts=0, split=False)
        assert list(separator.model.sources) == expected_sources[name]
        del separator
        gc.collect()

        session = DemucsSession(
            model=name,
            cache_dir=tmp_path,
            device=device,
            shifts=0,
            split=False,
        ).load()
        _, stems = session.infer(audio.squeeze(0), sample_rate=44100)
        assert list(stems) == expected_sources[name]
        session.close()

        _, one_shot_stems = separate_tensor(
            audio.squeeze(0),
            sample_rate=44100,
            model=name,
            cache_dir=tmp_path,
            device=device,
            shifts=0,
            split=False,
        )
        assert list(one_shot_stems) == expected_sources[name]
        gc.collect()
