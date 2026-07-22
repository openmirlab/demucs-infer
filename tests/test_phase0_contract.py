"""Freeze Phase 0 APIs and checkpoint evidence before runtime refactoring.

Discrete public signatures, CLI behavior, registry metadata, and loader seams
are always checked offline. Real numeric probes are opt-in through a verified
local cache and guard float digests to their recorded torch/device environment.

Reads: tests/fixtures Phase 0 JSON, tools capture modules, public APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from demucs_infer.clean_api import DemucsSession
from demucs_infer.hdemucs import HDemucs
from demucs_infer.pretrained import SOURCES, get_model


REPO_ROOT = Path(__file__).parent.parent
TOOLS_ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import capture_checkpoint_probes  # noqa: E402
import capture_public_contract  # noqa: E402
from phase0_probe_common import ARTIFACTS  # noqa: E402

CONTRACT_FIXTURE = REPO_ROOT / "tests/fixtures/public_contract.json"
PROBE_FIXTURE = REPO_ROOT / "tests/fixtures/checkpoint_probe_baseline.json"


def test_public_contract_matches_phase0_snapshot() -> None:
    expected = json.loads(CONTRACT_FIXTURE.read_text(encoding="utf-8"))
    capture_public_contract.assert_matches_baseline(
        capture_public_contract.build_snapshot(),
        expected,
    )


def test_probe_fixture_owns_pinned_urls_hashes_and_discrete_loading_facts() -> None:
    fixture = json.loads(PROBE_FIXTURE.read_text(encoding="utf-8"))
    expected_artifacts = {
        artifact.key: {"filename": artifact.filename, "url": artifact.url, "sha256": artifact.sha256}
        for artifact in ARTIFACTS
    }
    assert fixture["artifacts"] == expected_artifacts
    assert all(metadata["url"].startswith("https://") for metadata in expected_artifacts.values())
    assert all(len(metadata["sha256"]) == 64 for metadata in expected_artifacts.values())

    assert fixture["uvr"]["bag"]["model"]["sources"] == ["vocals", "non_vocals"]
    assert fixture["uvr"]["loader"] == "demucs_infer.states.load_model"
    assert fixture["uvr"]["bag"]["model"]["component_order"] == ["uvr_model_2", "uvr_model_1"]
    assert fixture["cdx23"]["bag"]["model"]["sources"] == ["music", "sfx", "speech"]
    assert fixture["cdx23"]["loader"] == "demucs_infer.states.load_model"
    assert fixture["cdx23"]["bag"]["model"]["component_order"] == ["cdx23_a", "cdx23_b", "cdx23_c"]
    assert fixture["msst"]["model"]["sources"] == ["vocals", "other"]
    assert fixture["msst"]["loader"] == "HTDemucs constructor plus raw state_dict"
    assert fixture["msst"]["strict_load"] is True
    for family in ("uvr", "cdx23"):
        assert fixture[family]["bag"]["model"]["samplerate"] == 44100
        assert fixture[family]["bag"]["model"]["audio_channels"] == 2
    assert fixture["msst"]["model"]["samplerate"] == 44100
    assert fixture["msst"]["model"]["audio_channels"] == 2


def _write_tiny_export(path: Path) -> str:
    model = HDemucs(channels=4, sources=SOURCES)
    args, kwargs = model._init_args_kwargs
    torch.save({"klass": type(model), "args": args, "kwargs": kwargs, "state": model.state_dict()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repo_path_and_direct_checkpoint_override_remain_loadable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "localmodel.th"
    digest = _write_tiny_export(checkpoint)

    legacy = get_model("localmodel", repo=tmp_path)
    assert isinstance(legacy, HDemucs)
    assert legacy.training is False

    session = DemucsSession(checkpoint_path=checkpoint, checkpoint_sha256=digest, device="cpu").load()
    assert isinstance(session._separator.model, HDemucs)
    assert session.status == "ready"
    session.close()


def test_real_checkpoint_probe_matches_when_cache_is_opted_in() -> None:
    cache_value = os.environ.get("DEMUCS_PHASE0_PROBE_CACHE")
    if not cache_value:
        pytest.skip("set DEMUCS_PHASE0_PROBE_CACHE to run verified real-checkpoint probes")
    expected = json.loads(PROBE_FIXTURE.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual = capture_checkpoint_probes.build_evidence(Path(cache_value), offline=True, device=device)
    capture_checkpoint_probes.compare_evidence(actual, expected)
