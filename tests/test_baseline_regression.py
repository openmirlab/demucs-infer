"""ADOPT P5 -- htdemucs baseline regression test.

Reruns tools/capture_baseline.py's seeded htdemucs separation (random.seed(1234)
pinned immediately before apply_model, per the shifts-reproducibility quirk
documented in apply.py's header) and asserts every stem's sha256 digest
matches tests/fixtures/baseline_htdemucs.json (the P0 fixture) exactly.

This is the constitution's "accuracy cannot drop" gate made permanent: any
future change that alters htdemucs's numerical output -- even a refactor
believed to be behavior-preserving -- fails this test.

Requires the htdemucs checkpoint to already be cached locally (torch hub
cache); does not download anything itself, so it is NOT marked `network`
and runs by default. On a machine without the checkpoint cached, this test
will attempt a download on first run (same as any normal `get_model()` call).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import capture_baseline  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline_htdemucs.json"


def test_baseline_matches_fixture():
    with open(FIXTURE_PATH) as f:
        expected = json.load(f)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual = capture_baseline.build_fixture(
        expected["meta"]["model"], device, skip_real=False,
    )

    expected_meta = dict(expected["meta"])
    actual_meta = dict(actual["meta"])
    expected_meta.pop("captured_at", None)
    actual_meta.pop("captured_at", None)
    # device/torch_version are allowed to differ across machines; only the
    # actual separation output (clips) must match bit-for-bit.
    for key in ("device", "torch_version"):
        expected_meta.pop(key, None)
        actual_meta.pop(key, None)
    assert expected_meta == actual_meta, "baseline metadata (seed/config) drifted"

    assert set(actual["clips"].keys()) >= {"synthetic"}, "expected at least the synthetic clip"
    for clip_name, expected_clip in expected["clips"].items():
        if clip_name not in actual["clips"]:
            # real_excerpt is skipped gracefully if the scratchpad audio isn't
            # present on this machine (see capture_baseline.py); don't fail
            # the whole suite over an environment-specific fixture.
            continue
        actual_clip = actual["clips"][clip_name]
        for stem_name, expected_stem in expected_clip["stems"].items():
            actual_stem = actual_clip["stems"][stem_name]
            assert actual_stem["sha256"] == expected_stem["sha256"], (
                f"{clip_name}/{stem_name}: htdemucs output diverged from the "
                f"P0 baseline -- accuracy regression"
            )
            assert actual_stem["shape"] == expected_stem["shape"]
