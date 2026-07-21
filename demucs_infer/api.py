# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""High-level Separator API -- the primary entry point for using demucs-infer
programmatically.

Wraps model loading (pretrained.get_model / repo.*Repo) and the low-level
apply.apply_model chunking/shifting loop behind a single Separator class,
plus audio I/O convenience (load, save, format conversion). Downstream
callers wanting the raw model + apply_model call pattern directly
(bypassing this class) can still do so -- see apply.py and pretrained.py,
the same functions this module composes.

Model discovery/metadata (ModelInfo, get_model_info, list_models,
list_supported_separation_types, KNOWN_MODELS, SEPARATION_TYPES,
SOURCE_TRANSLATIONS) moved to model_info.py in the ADOPT campaign's P4 --
that was api.py's cleanest seam (describing models vs. running separation
are different concerns) and is re-exported below unchanged so every
existing `from demucs_infer.api import ...` keeps working.

Classes
-------
`demucs.api.Separator`: The base separator class
`demucs.api.ModelInfo`: Model information container (see model_info.py)

Functions
---------
`demucs.api.save_audio`: Save an audio
`demucs.api.list_models`: Get models list (see model_info.py)
`demucs.api.get_model_info`: Get detailed model information (see model_info.py)

Examples
--------
See the end of this module (if __name__ == "__main__")

Reads: apply (apply_model, BagOfModels), audio (AudioFile, convert_audio,
save_audio), model_info (ModelInfo, LoadModelError, list_models,
get_model_info, list_supported_separation_types, KNOWN_MODELS,
SEPARATION_TYPES, SOURCE_TRANSLATIONS -- re-exported for backward compat),
pretrained (get_model)
"""

import subprocess

import torch as th
import torchaudio as ta

from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Union

from .apply import apply_model, _replace_dict, BagOfModels
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model

# Re-exported for backward compatibility: these used to be defined directly
# in this module (see model_info.py's header for why they moved).
from .model_info import (  # noqa: F401
    ModelInfo,
    LoadModelError,
    KNOWN_MODELS,
    SEPARATION_TYPES,
    SOURCE_TRANSLATIONS,
    list_models,
    get_model_info,
    list_supported_separation_types,
)


class LoadAudioError(Exception):
    pass


# Extensions where soundfile's decode is empirically bit-identical to
# torchaudio's (verified: torchaudio==2.7.0+cpu vs soundfile==0.14.0,
# 16/24/32-bit PCM wav + flac, mono and stereo, `np.array_equal` exact --
# see tests/test_audio_fallback.py and the 4.2.2 CHANGELOG entry). For
# these, soundfile is tried as soon as ffmpeg is unavailable, ahead of
# torchaudio -- not merely as a last-resort fallback -- since there is no
# accuracy cost either way.
#
# mp3 (and anything else) is deliberately excluded: the same verification
# measured torchaudio's and soundfile's mp3 decodes to differ by up to
# ~7e-7 per sample (different underlying decoders, ffmpeg vs libmpg123),
# so silently decoding mp3 via soundfile would change existing users'
# output. Lossy formats stay on torchaudio only; if it can't decode
# (torchaudio>=2.11 without the separate torchcodec package), _load_audio
# raises a clear, actionable error instead of silently switching decoders.
_LOSSLESS_SOUNDFILE_EXTS = {".wav", ".flac"}


class _NotProvided:
    pass


NotProvided = _NotProvided()


def _resolve_device(device):
    """Resolve legacy automatic selection and validate explicit devices."""
    if device is None or device == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if isinstance(device, th.device):
        device = str(device)
    if device == "cpu":
        return "cpu"
    if not isinstance(device, str):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    if device == "mps":
        mps = getattr(th.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        return "mps"
    if device == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return "cuda"
    if not device.startswith("cuda:"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    index_text = device[5:]
    if not index_text.isdigit():
        raise ValueError("CUDA device index must be a non-negative integer")
    if not th.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    if int(index_text) >= th.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index_text} is not available")
    return device


class Separator:
    def __init__(
        self,
        model: str = "htdemucs",
        repo: Optional[Path] = None,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """
        `class Separator`
        =================

        Parameters
        ----------
        model: Pretrained model name or signature. Default is htdemucs.
        repo: Folder containing all pre-trained models for use.
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option. \
            The literal string `"auto"` is also accepted, and resolves to the same \
            cuda-if-available-else-cpu choice as leaving `device` unset.
        jobs: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.
        progress: If true, show a progress bar.

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.
        To abort the separation, raise `KeyboardInterrupt`.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.
        """
        self._name = model
        self._repo = repo
        self._load_model()
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                              segment=segment, jobs=jobs, progress=progress, callback=callback,
                              callback_arg=callback_arg)

    def update_parameter(
        self,
        device: Union[str, _NotProvided] = NotProvided,
        shifts: Union[int, _NotProvided] = NotProvided,
        overlap: Union[float, _NotProvided] = NotProvided,
        split: Union[bool, _NotProvided] = NotProvided,
        segment: Optional[Union[int, _NotProvided]] = NotProvided,
        jobs: Union[int, _NotProvided] = NotProvided,
        progress: Union[bool, _NotProvided] = NotProvided,
        callback: Optional[
            Union[Callable[[dict], None], _NotProvided]
        ] = NotProvided,
        callback_arg: Optional[Union[dict, _NotProvided]] = NotProvided,
    ):
        """
        Update the parameters of separation.

        Parameters
        ----------
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option. \
            The literal string `"auto"` is also accepted, and resolves to the same \
            cuda-if-available-else-cpu choice as leaving `device` unset.
        jobs: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.
        progress: If true, show a progress bar.

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.
        To abort the separation, raise `KeyboardInterrupt`.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.
        """
        if not isinstance(device, _NotProvided):
            self._device = _resolve_device(device)
        if not isinstance(shifts, _NotProvided):
            self._shifts = shifts
        if not isinstance(overlap, _NotProvided):
            self._overlap = overlap
        if not isinstance(split, _NotProvided):
            self._split = split
        if not isinstance(segment, _NotProvided):
            self._segment = segment
        if not isinstance(jobs, _NotProvided):
            self._jobs = jobs
        if not isinstance(progress, _NotProvided):
            self._progress = progress
        if not isinstance(callback, _NotProvided):
            self._callback = callback
        if not isinstance(callback_arg, _NotProvided):
            self._callback_arg = callback_arg

    def _load_model(self):
        self._model = get_model(name=self._name, repo=self._repo)
        if self._model is None:
            raise LoadModelError("Failed to load model")
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

    def _load_audio(self, track: Path):
        """Load `track`, preferring ffmpeg (via AudioFile), then falling
        back by format: soundfile first for lossless wav/flac (bit-identical
        to torchaudio, see `_LOSSLESS_SOUNDFILE_EXTS`'s comment above),
        otherwise torchaudio only -- lossy formats never silently fall back
        to soundfile, since its decode isn't guaranteed to match
        torchaudio's for those."""
        track = Path(track)
        errors = {}
        wav = None

        try:
            wav = AudioFile(track).read(streams=0, samplerate=self._samplerate,
                                        channels=self._audio_channels)
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."

        is_lossless = track.suffix.lower() in _LOSSLESS_SOUNDFILE_EXTS

        if wav is None and is_lossless:
            wav = self._try_soundfile_load(track, errors)

        if wav is None:
            try:
                raw, sr = ta.load(str(track))
            except Exception as err:
                msg = str(err.args[0]) if err.args else str(err)
                if not is_lossless:
                    msg += (
                        " -- demucs-infer does not fall back to soundfile for "
                        "lossy formats like this one (different decoders "
                        "produce different samples); install torchcodec "
                        "(`pip install demucs-infer[torchcodec]`) or convert "
                        "the file to wav/flac."
                    )
                errors["torchaudio"] = msg
            else:
                wav = convert_audio(raw, sr, self._samplerate, self._audio_channels)

        if wav is None:
            raise LoadAudioError(
                "\n".join(
                    "When trying to load using {}, got the following error: {}".format(
                        backend, error
                    )
                    for backend, error in errors.items()
                )
            )
        return wav

    def _try_soundfile_load(self, track: Path, errors: dict):
        """soundfile-based load for lossless formats (wav/flac) -- see
        `_LOSSLESS_SOUNDFILE_EXTS`. Returns None (and records into `errors`)
        on failure instead of raising, so callers can continue falling
        back."""
        try:
            import soundfile as sf
            audio_np, sr = sf.read(str(track))
            # Convert from [time, channels] to [channels, time]
            if len(audio_np.shape) == 1:
                # Mono
                wav = th.tensor(audio_np, dtype=th.float32).unsqueeze(0)
            else:
                wav = th.tensor(audio_np.T, dtype=th.float32)
            return convert_audio(wav, sr, self._samplerate, self._audio_channels)
        except ImportError:
            errors["soundfile"] = "soundfile is not installed. Install with: pip install soundfile"
        except Exception as err:
            errors["soundfile"] = str(err)
        return None

    def separate_tensor(
        self, wav: th.Tensor, sr: Optional[int] = None
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Separate a loaded tensor.

        Parameters
        ----------
        wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, \
            while the second is the waveform of each channel. Type should be float32. \
            e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.
        sr: Sample rate of the original audio, the wave will be resampled if it doesn't match the \
            model.

        Returns
        -------
        A tuple, whose first element is the original wave and second element is a dict, whose keys
        are the name of stems and values are separated waves. The original wave will have already
        been resampled.

        Notes
        -----
        Use this function with cautiousness. This function does not provide data verifying.
        """
        if sr is not None and sr != self.samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        out = apply_model(
                self._model,
                wav[None],
                segment=self._segment,
                shifts=self._shifts,
                split=self._split,
                overlap=self._overlap,
                device=self._device,
                num_workers=self._jobs,
                callback=self._callback,
                callback_arg=_replace_dict(
                    self._callback_arg, ("audio_length", wav.shape[1])
                ),
                progress=self._progress,
            )
        if out is None:
            raise KeyboardInterrupt
        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()
        return (wav, dict(zip(self._model.sources, out[0])))

    def separate_audio_file(self, file: Path):
        """
        Separate an audio file. The method will automatically read the file.

        Parameters
        ----------
        wav: Path of the file to be separated.

        Returns
        -------
        A tuple, whose first element is the original wave and second element is a dict, whose keys
        are the name of stems and values are separated waves. The original wave will have already
        been resampled.
        """
        return self.separate_tensor(self._load_audio(file), self.samplerate)

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def audio_channels(self):
        return self._audio_channels

    @property
    def model(self):
        return self._model


if __name__ == "__main__":
    # Test API functions
    # two-stem not supported

    from .separate import get_parser

    args = get_parser().parse_args()
    separator = Separator(
        model=args.name,
        repo=args.repo,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        split=args.split,
        segment=args.segment,
        jobs=args.jobs,
        callback=print
    )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    for file in args.tracks:
        separated = separator.separate_audio_file(file)[1]
        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        for stem, source in separated.items():
            stem = out / args.filename.format(
                track=Path(file).name.rsplit(".", 1)[0],
                trackext=Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)
