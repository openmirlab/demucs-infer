"""Regression test for the openunmix -> demucs_infer.wiener vendoring (ADOPT P1).

htdemucs's shipped config (cac=True) never calls wiener() at inference (see
hdemucs.py/htdemucs.py's _mask()), so this can't reuse the htdemucs baseline
fixture. Instead it replays the exact seeded inputs from
tests/fixtures/wiener_reference.json (captured from openunmix.filtering.wiener
before vendoring -- see tools/capture_wiener_fixture.py) through
demucs_infer.wiener.wiener across the softmask x residual x iterations
combinations htdemucs doesn't reach, asserting bit-identical output.

This does not require openunmix to be installed.
"""
import hashlib
import json
from pathlib import Path

import torch

from demucs_infer.wiener import wiener

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wiener_reference.json"


def _load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _digest(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().to(torch.float64).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_wiener_matches_reference_fixture():
    fixture = _load_fixture()
    shape = fixture["meta"]["shape"]
    nb_frames, nb_bins, nb_channels, nb_sources = shape

    torch.manual_seed(fixture["meta"]["seed"])
    gen = torch.Generator().manual_seed(fixture["meta"]["seed"])
    targets = torch.rand(nb_frames, nb_bins, nb_channels, nb_sources, generator=gen, dtype=torch.float64)
    mix_stft = torch.randn(nb_frames, nb_bins, nb_channels, 2, generator=gen, dtype=torch.float64)

    assert hashlib.sha256(targets.numpy().tobytes()).hexdigest() == fixture["inputs"]["targets_sha256"]
    assert hashlib.sha256(mix_stft.numpy().tobytes()).hexdigest() == fixture["inputs"]["mix_stft_sha256"]

    assert len(fixture["cases"]) == 12, "expected all softmask x residual x iterations combinations"

    for case in fixture["cases"]:
        out = wiener(
            targets.clone(), mix_stft.clone(),
            iterations=case["iterations"], softmask=case["softmask"], residual=case["residual"],
        )
        assert list(out.shape) == case["output"]["shape"], case
        assert _digest(out) == case["output"]["sha256"], (
            f"wiener() output digest mismatch for {case} -- vendored implementation "
            f"diverged from the recorded openunmix reference"
        )


def test_wiener_has_no_openunmix_import():
    """demucs_infer.wiener must not import openunmix (only reference it in
    the vendoring attribution comment/docstring)."""
    import ast

    import demucs_infer.wiener as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if "openunmix" in a.name]
        elif isinstance(node, ast.ImportFrom):
            if node.module and "openunmix" in node.module:
                offenders.append(node.module)
    assert offenders == [], f"found live openunmix import(s): {offenders}"
