"""Unit tests for the torchaudio/soundfile decoder selection logic.

torchaudio>=2.11 dropped its bundled wav/flac/mp3 decoders in favor of the
separate `torchcodec` package; `torchaudio.load`/`torchaudio.save` raise
`ImportError` without it. `demucs_infer.audio.save_audio` and
`demucs_infer.api.Separator._load_audio` carry fallbacks to `soundfile` for
exactly this case (see CHANGELOG 4.2.2) -- these tests pin down the
*selection* logic itself via monkeypatching, so they don't require
installing two different torchaudio versions. The bit-identity claims
referenced below were verified separately (once) against a real second
torchaudio install; see the CHANGELOG entry for the exact numbers.

Load (`Separator._load_audio`), after ffmpeg:
1. Lossless (wav/flac): soundfile is used directly, ahead of torchaudio --
   not merely as a last-resort fallback -- because decode was verified
   bit-identical (`np.array_equal`, torchaudio==2.7.0+cpu vs
   soundfile==0.14.0, 16/24/32-bit PCM + flac, mono/stereo).
2. Lossy (mp3, anything else): torchaudio only. If it raises, a clear
   actionable error is raised instead of silently decoding via soundfile --
   verified mp3 decode differs by up to ~7e-7 per sample between
   torchaudio (ffmpeg) and soundfile (libmpg123), which would silently
   change existing users' output.
3. The fallback catches broad exceptions, not just `ImportError`/
   `RuntimeError` -- a regression guard for commit 5d844c6 ("catch all
   exceptions from torchaudio.load() for torchcodec fallback").

Save (`demucs_infer.audio.save_audio`): the same wav-encode comparison was
also run (torchaudio.save vs soundfile.write, identical float32 input,
16-bit PCM) and found to differ by +/-1 LSB in ~50% of samples -- a real
rounding-convention difference, not noise. Because that bit-identity check
*failed*, save deliberately keeps its original design: torchaudio is
always tried first, soundfile is used only when torchaudio itself raises.
Switching save's default encoder to soundfile would be an unproven,
silent output change and is out of scope here.

Reads: demucs_infer.audio (save_audio, _save_audio_soundfile), demucs_infer.api
(Separator, AudioFile, LoadAudioError, _LOSSLESS_SOUNDFILE_EXTS)
"""
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import demucs_infer.audio as audio_mod
import demucs_infer.api as api_mod
from demucs_infer.api import Separator


class _TorchCodecImportError(ImportError):
    """Mirrors the actual exception torchaudio>=2.11 raises without
    torchcodec: `ImportError('TorchCodec is required for
    save_with_torchcodec. Please install torchcodec to use this
    function.')` (message reproduced verbatim from an empirical repro
    against torch==2.13.0+cpu / torchaudio==2.11.0+cpu)."""


class _SomeOtherLoaderError(Exception):
    """A deliberately-not-ImportError-or-RuntimeError exception, to prove
    the fallback around torchaudio.load catches broadly rather than only
    the two types the pre-5d844c6 code checked for."""


def _make_stub_wav(seconds=0.25, samplerate=8000):
    torch.manual_seed(0)
    return torch.rand(2, int(seconds * samplerate)) * 2 - 1  # [-1, 1)


def _make_separator_stub(samplerate=8000, channels=2):
    sep = object.__new__(Separator)
    sep._samplerate = samplerate
    sep._audio_channels = channels
    return sep


def _write_reference_wav(path, seconds=0.25, samplerate=8000, channels=2):
    import soundfile as sf
    audio_np = (np.random.RandomState(0).rand(int(seconds * samplerate), channels) * 2 - 1).astype(
        np.float32
    )
    sf.write(str(path), audio_np, samplerate)
    return audio_np


# ---------------------------------------------------------------------------
# save_audio: torchaudio raises -> soundfile fallback engages and succeeds
# (save keeps torchaudio-first ordering -- see module docstring for why)
# ---------------------------------------------------------------------------

def test_save_audio_wav_falls_back_to_soundfile_when_torchaudio_raises(tmp_path, monkeypatch):
    wav = _make_stub_wav()
    calls = {"soundfile": 0}

    def fake_ta_save(*args, **kwargs):
        raise _TorchCodecImportError(
            "TorchCodec is required for save_with_torchcodec. Please "
            "install torchcodec to use this function."
        )

    real_soundfile_save = audio_mod._save_audio_soundfile

    def spy_soundfile_save(*args, **kwargs):
        calls["soundfile"] += 1
        return real_soundfile_save(*args, **kwargs)

    monkeypatch.setattr(audio_mod.ta, "save", fake_ta_save)
    monkeypatch.setattr(audio_mod, "_save_audio_soundfile", spy_soundfile_save)

    out = tmp_path / "out.wav"
    audio_mod.save_audio(wav, out, samplerate=8000)

    assert calls["soundfile"] == 1
    assert out.exists() and out.stat().st_size > 0

    import soundfile as sf
    data, sr = sf.read(str(out))
    assert sr == 8000
    assert data.shape[0] == wav.shape[1]


def test_save_audio_flac_falls_back_to_soundfile_when_torchaudio_raises(tmp_path, monkeypatch):
    wav = _make_stub_wav()

    def fake_ta_save(*args, **kwargs):
        raise _TorchCodecImportError("TorchCodec is required for save_with_torchcodec.")

    monkeypatch.setattr(audio_mod.ta, "save", fake_ta_save)

    out = tmp_path / "out.flac"
    audio_mod.save_audio(wav, out, samplerate=8000)

    import soundfile as sf
    data, sr = sf.read(str(out))
    assert sr == 8000
    assert data.shape[0] == wav.shape[1]


def test_save_audio_does_not_use_soundfile_when_torchaudio_succeeds(tmp_path, monkeypatch):
    """torchaudio works -> soundfile must never run: zero behavior change
    for any install where torchaudio already worked."""
    wav = _make_stub_wav()
    ta_calls = {"save": 0}
    sf_calls = {"save": 0}

    def fake_ta_save(path, wav_arg, sample_rate, **kwargs):
        ta_calls["save"] += 1
        Path(path).write_bytes(b"fake-torchaudio-encoded-audio")

    def fake_soundfile_save(*args, **kwargs):
        sf_calls["save"] += 1
        raise AssertionError("soundfile fallback must not run when torchaudio succeeds")

    monkeypatch.setattr(audio_mod.ta, "save", fake_ta_save)
    monkeypatch.setattr(audio_mod, "_save_audio_soundfile", fake_soundfile_save)

    out = tmp_path / "out.wav"
    audio_mod.save_audio(wav, out, samplerate=8000)

    assert ta_calls["save"] == 1
    assert sf_calls["save"] == 0
    assert out.read_bytes() == b"fake-torchaudio-encoded-audio"


# ---------------------------------------------------------------------------
# Separator._load_audio, lossless (wav/flac): soundfile is used as soon as
# ffmpeg is unavailable -- ahead of torchaudio, not just a fallback for it.
# ---------------------------------------------------------------------------

def test_load_audio_wav_uses_soundfile_before_torchaudio_when_ffmpeg_fails(tmp_path, monkeypatch):
    wav_path = tmp_path / "in.wav"
    _write_reference_wav(wav_path)
    ta_calls = {"load": 0}

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_ta_load(path):
        ta_calls["load"] += 1
        raise AssertionError("torchaudio.load must not run: soundfile should have succeeded first")

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)

    sep = _make_separator_stub()
    wav = sep._load_audio(wav_path)

    assert ta_calls["load"] == 0
    assert wav.shape[0] == sep._audio_channels
    assert wav.shape[1] > 0


def test_load_audio_flac_uses_soundfile_before_torchaudio_when_ffmpeg_fails(tmp_path, monkeypatch):
    import soundfile as sf

    flac_path = tmp_path / "in.flac"
    audio_np = (np.random.RandomState(1).rand(2000, 2) * 2 - 1).astype(np.float32)
    sf.write(str(flac_path), audio_np, 8000)
    ta_calls = {"load": 0}

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_ta_load(path):
        ta_calls["load"] += 1
        raise AssertionError("torchaudio.load must not run: soundfile should have succeeded first")

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)

    sep = _make_separator_stub()
    wav = sep._load_audio(flac_path)

    assert ta_calls["load"] == 0
    assert wav.shape[0] == sep._audio_channels
    assert wav.shape[1] > 0


def test_load_audio_wav_falls_further_back_to_torchaudio_if_soundfile_itself_fails(
    tmp_path, monkeypatch
):
    """If soundfile can't read a (nominally) lossless file either, torchaudio
    still gets a chance -- soundfile-first doesn't mean torchaudio-never."""
    wav_path = tmp_path / "in.wav"
    _write_reference_wav(wav_path)

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_soundfile_read(*args, **kwargs):
        raise RuntimeError("simulated: soundfile could not parse this file")

    def fake_ta_load(path):
        return torch.zeros(2, 100), 8000

    # soundfile is imported lazily inside _try_soundfile_load, so patch the
    # real module directly.
    import soundfile as real_sf

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(real_sf, "read", fake_soundfile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)

    sep = _make_separator_stub()
    wav = sep._load_audio(wav_path)

    assert wav.shape[0] == sep._audio_channels
    assert wav.shape[1] > 0


# ---------------------------------------------------------------------------
# Separator._load_audio, lossy (mp3): torchaudio only -- never silently
# falls back to soundfile even though libsndfile can technically decode mp3,
# because the decode is not bit-identical (verified separately, ~7e-7 max
# abs diff) and would silently change existing users' output.
# ---------------------------------------------------------------------------

def test_load_audio_mp3_does_not_fall_back_to_soundfile_when_torchaudio_fails(tmp_path, monkeypatch):
    mp3_path = tmp_path / "in.mp3"
    mp3_path.write_bytes(b"not-real-mp3-bytes")  # never actually decoded; both backends are mocked
    soundfile_calls = {"attempted": 0}

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_ta_load(path):
        raise _TorchCodecImportError(
            "TorchCodec is required for load_with_torchcodec. Please "
            "install torchcodec to use this function."
        )

    def spy_try_soundfile_load(self, track, errors):
        soundfile_calls["attempted"] += 1
        return None

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)
    monkeypatch.setattr(Separator, "_try_soundfile_load", spy_try_soundfile_load)

    sep = _make_separator_stub()
    with pytest.raises(api_mod.LoadAudioError) as excinfo:
        sep._load_audio(mp3_path)

    assert soundfile_calls["attempted"] == 0, "mp3 must never attempt the soundfile path"
    message = str(excinfo.value)
    assert "torchcodec" in message.lower()
    assert "wav/flac" in message or "convert" in message.lower()


# ---------------------------------------------------------------------------
# Broad-exception-catch regression guard (commit 5d844c6): an exception
# type that is neither ImportError nor RuntimeError must not propagate
# uncaught past torchaudio.load -- it should fold into a clean LoadAudioError.
# ---------------------------------------------------------------------------

def test_load_audio_catches_broad_exceptions_not_just_importerror_or_runtimeerror(
    tmp_path, monkeypatch
):
    """Regression guard for 5d844c6: the pre-fix code only caught
    `(RuntimeError, ImportError)` around torchaudio.load, which missed
    other exception types some torchaudio/torchcodec versions raise. Use a
    lossy (mp3) extension so torchaudio.load is actually reached (wav/flac
    would resolve via soundfile first and never exercise this path), and
    confirm the odd exception type is caught and folded into a clean
    LoadAudioError instead of propagating raw."""
    mp3_path = tmp_path / "in.mp3"
    mp3_path.write_bytes(b"not-real-mp3-bytes")

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_ta_load(path):
        raise _SomeOtherLoaderError("some unrelated backend failure")

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)

    sep = _make_separator_stub()
    with pytest.raises(api_mod.LoadAudioError) as excinfo:
        sep._load_audio(mp3_path)  # must not raise _SomeOtherLoaderError directly

    assert "some unrelated backend failure" in str(excinfo.value)


def test_load_audio_raises_actionable_error_when_every_backend_fails(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.wav"

    def fake_audiofile_read(self, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg is not installed")

    def fake_ta_load(path):
        raise _TorchCodecImportError("TorchCodec is required for load_with_torchcodec.")

    monkeypatch.setattr(api_mod.AudioFile, "read", fake_audiofile_read)
    monkeypatch.setattr(api_mod.ta, "load", fake_ta_load)

    sep = _make_separator_stub()
    with pytest.raises(api_mod.LoadAudioError) as excinfo:
        sep._load_audio(missing_path)

    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "torchaudio" in message
    assert "soundfile" in message


# ---------------------------------------------------------------------------
# soundfile must be a *declared* dependency, not a silent transitive
# assumption (the actual bug: the fallback code above already existed, but
# nothing installed soundfile on a fresh `pip install demucs-infer`).
# ---------------------------------------------------------------------------

def test_soundfile_is_a_declared_core_dependency():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    match = re.search(r"dependencies = \[(.*?)\]", text, re.DOTALL)
    assert match, "could not find [project] dependencies array in pyproject.toml"
    deps_block = match.group(1)
    assert "soundfile" in deps_block, (
        "soundfile must be a declared core dependency: it is the wav/flac "
        "decoder used as soon as ffmpeg is unavailable, and an undeclared "
        "fallback dependency is the exact bug this test guards against."
    )


def test_soundfile_importable():
    """Smoke check that the environment running the suite actually has the
    now-declared dependency installed (would fail loudly instead of the
    fallback tests above failing confusingly)."""
    import soundfile  # noqa: F401
