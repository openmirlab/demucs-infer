"""Golden audio I/O checks after removing TorchAudio from demucs-infer.

The generated stereo fixture has active tones/noise plus a real silent tail.
PCM digests and MP3 inference arrays were recorded from v4.3.0 before the
dependency change. WAV/FLAC outputs must remain sample-identical; MP3's
alternative decoder has an explicitly bounded scale-relative difference.

Reads: demucs_infer.api (AudioFile, Separator, LoadAudioError),
demucs_infer.audio (save_audio)
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import demucs_infer.api as api_mod
from demucs_infer import DemucsSession
from demucs_infer.api import Separator
from demucs_infer.audio import save_audio


FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "audio_io_reference.json").read_text())


def _separator_stub():
    separator = object.__new__(Separator)
    separator._samplerate = REFERENCE["sample_rate"]
    separator._audio_channels = 2
    return separator


def _no_ffmpeg(self, *args, **kwargs):
    raise FileNotFoundError("forced unavailable FFmpeg executable")


def test_generated_input_matches_recorded_fixture():
    audio = np.load(FIXTURES / "audio_io_generated.npy", allow_pickle=False)
    assert audio.shape == (2, REFERENCE["sample_rate"])
    assert np.any(audio[:, :1000] != 0)
    assert np.count_nonzero(audio[:, int(.75 * REFERENCE["sample_rate"]):]) == 0
    assert hashlib.sha256(audio.tobytes()).hexdigest() == REFERENCE["input_sha256"]


@pytest.mark.parametrize("key,bits,as_float,ext", [
    ("wav_16_pcm", 16, False, ".wav"),
    ("wav_24_pcm", 24, False, ".wav"),
    ("wav_32_pcm", 32, False, ".wav"),
    ("wav_32_float", 32, True, ".wav"),
    ("flac_16_pcm", 16, False, ".flac"),
    ("flac_24_pcm", 24, False, ".flac"),
])
def test_save_audio_matches_v430_decoded_pcm(tmp_path, key, bits, as_float, ext):
    audio = np.load(FIXTURES / "audio_io_generated.npy", allow_pickle=False)
    path = tmp_path / (key + ext)
    save_audio(torch.from_numpy(audio.copy()), path, REFERENCE["sample_rate"],
               clip="none", bits_per_sample=bits, as_float=as_float)

    expected = REFERENCE["pcm"][key]
    decoded, sample_rate = sf.read(path, dtype=expected["dtype"], always_2d=True)
    assert sample_rate == REFERENCE["sample_rate"]
    assert list(decoded.shape) == expected["shape"]
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("ext", [".wav", ".flac"])
def test_lossless_input_uses_soundfile_without_ffmpeg(tmp_path, monkeypatch, ext):
    source = np.load(FIXTURES / "audio_io_generated.npy", allow_pickle=False)
    path = tmp_path / ("input" + ext)
    sf.write(path, source.T, REFERENCE["sample_rate"], subtype="PCM_16")
    monkeypatch.setattr(api_mod.AudioFile, "read", _no_ffmpeg)

    actual = _separator_stub()._load_audio(path)
    decoded, _ = sf.read(path, dtype="float32", always_2d=True)
    assert torch.equal(actual, torch.from_numpy(decoded.T.copy()))


def test_mp3_input_uses_soundfile_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(api_mod.AudioFile, "read", _no_ffmpeg)
    actual = _separator_stub()._load_audio(FIXTURES / "audio_io_generated.mp3")
    expected = np.load(FIXTURES / "audio_io_mp3_v430.npz", allow_pickle=False)["mp3_mixture"]
    delta = actual.numpy().astype(np.float64) - expected.astype(np.float64)
    reference_rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
    assert reference_rms > .1  # nontrivial audio, so relative error is meaningful
    assert np.max(np.abs(delta)) / reference_rms <= 1e-5
    assert np.sqrt(np.mean(delta ** 2)) / reference_rms <= 1e-6


def test_ffmpeg_reader_stays_primary(monkeypatch):
    marker = torch.ones(2, 25)
    monkeypatch.setattr(api_mod.AudioFile, "read", lambda self, **kwargs: marker)
    monkeypatch.setattr(Separator, "_try_soundfile_load",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")))
    assert _separator_stub()._load_audio(Path("unused.mp3")) is marker


def test_both_readers_fail_with_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.AudioFile, "read", _no_ffmpeg)
    path = tmp_path / "invalid.mp3"
    path.write_bytes(b"not an audio file")
    with pytest.raises(api_mod.LoadAudioError) as exc:
        _separator_stub()._load_audio(path)
    assert "FFmpeg" in str(exc.value)
    assert "soundfile" in str(exc.value)


@pytest.mark.realweights
def test_real_checkpoint_mp3_fallback_matches_v430_outputs(monkeypatch):
    if torch.__version__ != "2.8.0+cu128":
        pytest.skip("Golden inference arrays were recorded on Torch 2.8.0+cu128")
    session = DemucsSession(model="htdemucs", device="cpu")
    if not session.cache_info()["cached"]:
        pytest.skip("Cached htdemucs checkpoint is required for this parity test")

    monkeypatch.setattr(api_mod.AudioFile, "read", _no_ffmpeg)
    random.seed(1234)
    with session:
        with torch.no_grad():
            mixture, stems = session.infer(FIXTURES / "audio_io_generated.mp3")

    reference = np.load(FIXTURES / "audio_io_mp3_v430.npz", allow_pickle=False)
    reference_rms = np.sqrt(np.mean(reference["mp3_mixture"].astype(np.float64) ** 2))
    assert reference_rms > .1
    observed = {"mp3_mixture": mixture, **{"mp3_" + key: value for key, value in stems.items()}}
    assert set(observed) == set(reference.files)
    for key, value in observed.items():
        expected = reference[key]
        actual = value.detach().cpu().numpy()
        assert actual.shape == expected.shape
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        assert np.max(np.abs(delta)) / reference_rms <= 1e-5, key
        assert np.sqrt(np.mean(delta ** 2)) / reference_rms <= 1e-6, key
