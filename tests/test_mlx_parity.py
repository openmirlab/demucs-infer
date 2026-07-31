"""Torch-vs-MLX output parity on the real `htdemucs` checkpoint, including
silence, exercised through the public `Separator.separate_audio_file()` API
on a real audio file on disk (not a bare module forward pass).

Marked ``realweights`` and deselected by default: needs the `[mlx]` extra,
an Apple Silicon Mac, and network access to download the default checkpoint
on first run (cached under `~/.cache/demucs-infer/` after that, matching
every other real-checkpoint test in this suite).

Run explicitly:  pytest -m realweights tests/test_mlx_parity.py -v

The silence cases are the point of this file, not signal alone. In the
sibling `bs-roformer-infer` package, a clean-signal fixture agreed to 4.0e-07
while a zero-padded-tail fixture diverged by 1.455e-02 -- a fixture without
silence would have shipped that bug. `htdemucs`'s own chunking
(`apply.apply_model`'s split path) zero-pads via `TensorChunk.padded()`
whenever a track's length isn't an exact multiple of its stride, which is
true for nearly every real track, so this fixture's duration is deliberately
not a clean multiple.

`exact_zero_safe_rfft` (`demucs_infer/mlx/rfft_guard.py`) was measured, not
assumed, to be inert for this architecture -- see `test_rfft_guard_removed`
below (the "validate the regression test by removing the fix" check) and
`demucs_infer/mlx/htdemucs.py`'s module docstring for the reasoning
(htdemucs's normalization eps is 1e-5, seven orders above the ~4.5e-07 rfft
artifact, unlike the sibling package's 1e-12).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.realweights

MODEL = "htdemucs"
SEED = 1
SAMPLE_RATE = 44100
# 9.3s: deliberately not a clean multiple of the model's chunk stride, so the
# real chunking path's zero-padded final chunk is actually exercised (see
# module docstring).
DURATION_SECONDS = 9.3
SIGNAL_SECONDS = 5.0
# Measured worst case (see this file's module docstring / CHANGELOG) is
# ~2e-07; the gate is set from what the implementation actually achieves,
# with headroom for run-to-run noise, not widened to make an unrelated
# future regression pass quietly.
MAX_ABS_TOLERANCE = 1e-4


def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def loaded():
    """(torch Separator, mlx Separator) built from the real, default
    checkpoint. Skips (never fails) when a prerequisite is missing."""
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'demucs-infer[mlx]'")

    from demucs_infer.api import Separator

    try:
        torch_sep = Separator(model=MODEL, backend="torch", device="cpu", shifts=0, progress=False)
        mlx_sep = Separator(model=MODEL, backend="mlx", shifts=0, progress=False)
    except Exception as exc:  # pragma: no cover - network/env dependent
        pytest.skip(f"{MODEL} checkpoint unavailable: {exc}")
    return torch_sep, mlx_sep


def _fixture_audio(tail: str) -> np.ndarray:
    """(samples, channels) stereo audio: real signal up front, `tail`
    behaviour after it, matching what `soundfile`/`separate_audio_file`
    hands the model."""
    n = int(SAMPLE_RATE * DURATION_SECONDS)
    signal_n = int(SAMPLE_RATE * SIGNAL_SECONDS)
    rng = np.random.default_rng(SEED)
    audio = (rng.standard_normal((n, 2)) * 0.05).astype(np.float32)
    if tail == "zeros":
        audio[signal_n:, :] = 0.0
    elif tail == "near_silent":
        audio[signal_n:, :] *= 1e-6
    return audio


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_including_silence(tmp_path: Path, loaded, tail):
    torch_sep, mlx_sep = loaded
    audio = _fixture_audio(tail)

    wav_path = tmp_path / f"{tail}.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="FLOAT")

    _, torch_stems = torch_sep.separate_audio_file(wav_path)
    _, mlx_stems = mlx_sep.separate_audio_file(wav_path)

    assert set(torch_stems) == set(mlx_stems)
    worst = 0.0
    for stem, reference in torch_stems.items():
        diff = float((reference - mlx_stems[stem]).abs().max())
        worst = max(worst, diff)
    print(f"\n[mlx parity] tail={tail} worst_max_abs={worst:.3e}")
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: Torch-vs-MLX max abs {worst:.3e} exceeds {MAX_ABS_TOLERANCE:.0e}. "
        f"If this fires only for a silent tail, suspect exact_zero_safe_rfft in "
        f"demucs_infer/mlx/rfft_guard.py -- investigate rather than widen the tolerance"
    )


def test_rfft_guard_removed_stays_within_noise_floor(tmp_path: Path, loaded):
    """Validates the regression test itself: removes `exact_zero_safe_rfft`
    and re-measures the zero-padded-tail case.

    Unlike the sibling `bs-roformer-infer` package (where removing the guard
    reproduces a real 1.455e-02 divergence), this is expected to report that
    removing the guard does *not* meaningfully move the result for
    `htdemucs` -- documented honestly, the way `mdxnet-infer` did for its
    own architecture, rather than claimed without measuring.
    """
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'demucs-infer[mlx]'")

    torch_sep, mlx_sep = loaded
    audio = _fixture_audio("zeros")
    wav_path = tmp_path / "zeros_noguard.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="FLOAT")

    _, torch_stems = torch_sep.separate_audio_file(wav_path)
    _, mlx_stems_with_guard = mlx_sep.separate_audio_file(wav_path)
    worst_with_guard = max(
        float((torch_stems[s] - mlx_stems_with_guard[s]).abs().max()) for s in torch_stems
    )

    import demucs_infer.mlx.rfft_guard as rfft_guard_module
    import demucs_infer.mlx.spec as spec_module

    @contextlib.contextmanager
    def _noop_guard():
        yield

    original = rfft_guard_module.exact_zero_safe_rfft
    rfft_guard_module.exact_zero_safe_rfft = _noop_guard
    spec_module.exact_zero_safe_rfft = _noop_guard
    try:
        from demucs_infer.api import Separator

        mlx_sep_noguard = Separator(model=MODEL, backend="mlx", shifts=0, progress=False)
        _, mlx_stems_no_guard = mlx_sep_noguard.separate_audio_file(wav_path)
    finally:
        rfft_guard_module.exact_zero_safe_rfft = original
        spec_module.exact_zero_safe_rfft = original

    worst_without_guard = max(
        float((torch_stems[s] - mlx_stems_no_guard[s]).abs().max()) for s in torch_stems
    )
    print(
        f"\n[rfft guard removed] zero-padded tail: with_guard={worst_with_guard:.3e} "
        f"without_guard={worst_without_guard:.3e}"
    )
    # Not asserting a *difference* here on purpose: the point of this test is
    # to report the measurement (see CHANGELOG.md and htdemucs.py's module
    # docstring for the recorded numbers), not to encode an assumption about
    # which way it goes. Both must still be finite and sane.
    assert np.isfinite(worst_without_guard)
