# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API methods for demucs

Classes
-------
`demucs.api.Separator`: The base separator class
`demucs.api.ModelInfo`: Model information container

Functions
---------
`demucs.api.save_audio`: Save an audio
`demucs.api.list_models`: Get models list
`demucs.api.get_model_info`: Get detailed model information

Examples
--------
See the end of this module (if __name__ == "__main__")
"""

import subprocess

import torch as th
import torchaudio as ta

from .log import fatal
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Union, List
from dataclasses import dataclass, field

from .apply import apply_model, _replace_dict, BagOfModels
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model, _parse_remote_files, REMOTE_ROOT
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo


# =============================================================================
# Source name translations and model metadata
# =============================================================================

# Translation mapping for non-English source names to English
SOURCE_TRANSLATIONS: Dict[str, str] = {
    # Spanish (drumsep)
    'bombo': 'kick',
    'redoblante': 'snare',
    'platillos': 'cymbals',
    'toms': 'toms',
    # Add more translations as needed
}

# Separation type categorization based on sources
SEPARATION_TYPES: Dict[str, Dict] = {
    'music_4stem': {
        'name': 'Music Separation (4 stems)',
        'description': 'Separates music into drums, bass, other instruments, and vocals',
        'sources': {'drums', 'bass', 'other', 'vocals'},
    },
    'music_6stem': {
        'name': 'Music Separation (6 stems)',
        'description': 'Separates music into drums, bass, other, vocals, guitar, and piano',
        'sources': {'drums', 'bass', 'other', 'vocals', 'guitar', 'piano'},
    },
    'drum_kit': {
        'name': 'Drum Kit Separation',
        'description': 'Separates drum recordings into individual kit pieces (kick, snare, cymbals, toms)',
        'sources': {'kick', 'snare', 'cymbals', 'toms', 'bombo', 'redoblante', 'platillos'},
    },
    'cinematic': {
        'name': 'Cinematic/Film Audio Separation',
        'description': 'Separates film/video audio into music, sound effects (sfx), and speech/dialog',
        'sources': {'music', 'sfx', 'speech', 'dialog', 'effect'},
    },
    'speech': {
        'name': 'Speech Separation',
        'description': 'Separates speech/vocals from other audio',
        'sources': {'speech', 'vocals', 'voice'},
    },
    'stereo_center': {
        'name': 'Stereo Center/Sides Separation',
        'description': 'Separates audio into center (mono/similar) and sides (stereo difference) components',
        'sources': {'similarity', 'difference', 'center', 'sides'},
    },
    'vocal_instrumental': {
        'name': 'Vocal/Instrumental Separation',
        'description': 'Separates audio into vocals and non-vocals (instrumental)',
        'sources': {'vocals', 'non_vocals', 'instrumental', 'accompaniment'},
    },
    'unknown': {
        'name': 'Custom Separation',
        'description': 'Custom model with non-standard source configuration',
        'sources': set(),
    },
}

# Known model metadata for common models
KNOWN_MODELS: Dict[str, Dict] = {
    'htdemucs': {
        'display_name': 'HT-Demucs',
        'description': 'Hybrid Transformer Demucs - high quality music separation',
        'use_case': 'General music separation',
    },
    'htdemucs_ft': {
        'display_name': 'HT-Demucs Fine-tuned',
        'description': 'Fine-tuned Hybrid Transformer Demucs ensemble (4 models) - best quality',
        'use_case': 'High quality music separation',
    },
    'htdemucs_6s': {
        'display_name': 'HT-Demucs 6-stem',
        'description': 'Hybrid Transformer Demucs with 6 output stems including guitar and piano',
        'use_case': 'Extended music separation with guitar/piano',
    },
    'hdemucs_mmi': {
        'display_name': 'H-Demucs MMI',
        'description': 'Hybrid Demucs trained with MMI criterion',
        'use_case': 'Music separation (alternative training)',
    },
    'mdx': {
        'display_name': 'MDX',
        'description': 'Demucs model from MDX challenge',
        'use_case': 'Music separation',
    },
    'mdx_extra': {
        'display_name': 'MDX Extra',
        'description': 'Enhanced Hybrid Demucs from MDX challenge',
        'use_case': 'Music separation',
    },
    '49469ca8': {
        'display_name': 'Drumsep',
        'description': 'Drum stem separation model - separates drum recordings into kit pieces',
        'use_case': 'Drum recording separation (kick, snare, cymbals, toms)',
    },
    '97d170e1': {
        'display_name': 'CDX23 Cinematic',
        'description': 'Cinematic sound demixing model from CDX23 challenge',
        'use_case': 'Film/video audio separation (dialog, music, sfx)',
    },
    '04573f0d': {
        'display_name': 'HTDemucs (MDX23)',
        'description': 'HTDemucs variant used in MVSep-MDX23 ensemble',
        'use_case': 'Music separation',
    },
    'phantom_center': {
        'display_name': 'Phantom Center Extractor',
        'description': 'HTDemucs model for extracting center (similarity) and sides (difference) from stereo audio',
        'use_case': 'Stereo center/sides separation (phantom center extraction)',
    },
    'ebf34a2d': {
        'display_name': 'UVR Demucs Model 1',
        'description': 'HDemucs model optimized for vocal/non-vocal (instrumental) separation',
        'use_case': 'Vocal/instrumental separation (2-stem)',
    },
}


@dataclass
class ModelInfo:
    """
    Container for model information.

    Attributes
    ----------
    name : str
        Model name or signature used to load the model.
    display_name : str
        Human-readable display name.
    architecture : str
        Model architecture (e.g., 'HTDemucs', 'HDemucs', 'Demucs').
    sources : List[str]
        List of source/stem names the model outputs.
    sources_english : List[str]
        List of source names translated to English.
    separation_type : str
        Type of separation (e.g., 'music_4stem', 'drum_kit', 'cinematic').
    separation_type_name : str
        Human-readable name for the separation type.
    description : str
        Description of the model and its purpose.
    use_case : str
        Recommended use case for this model.
    sample_rate : int
        Audio sample rate the model expects.
    audio_channels : int
        Number of audio channels (typically 2 for stereo).
    is_bag : bool
        Whether this is a bag of models (ensemble).
    num_models : int
        Number of models in the ensemble (1 if not a bag).
    """
    name: str
    display_name: str
    architecture: str
    sources: List[str]
    sources_english: List[str]
    separation_type: str
    separation_type_name: str
    description: str
    use_case: str
    sample_rate: int
    audio_channels: int
    is_bag: bool = False
    num_models: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'architecture': self.architecture,
            'sources': self.sources,
            'sources_english': self.sources_english,
            'separation_type': self.separation_type,
            'separation_type_name': self.separation_type_name,
            'description': self.description,
            'use_case': self.use_case,
            'sample_rate': self.sample_rate,
            'audio_channels': self.audio_channels,
            'is_bag': self.is_bag,
            'num_models': self.num_models,
        }

    def __str__(self) -> str:
        sources_str = ', '.join(self.sources_english)
        return (
            f"{self.display_name} ({self.name})\n"
            f"  Type: {self.separation_type_name}\n"
            f"  Architecture: {self.architecture}"
            f"{f' (ensemble of {self.num_models})' if self.is_bag else ''}\n"
            f"  Sources: {sources_str}\n"
            f"  Sample Rate: {self.sample_rate} Hz\n"
            f"  Use Case: {self.use_case}"
        )


class LoadAudioError(Exception):
    pass


class LoadModelError(Exception):
    pass


class _NotProvided:
    pass


NotProvided = _NotProvided()


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
            will be stored on `wav.device`. If not specified, will use the command line option.
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
            will be stored on `wav.device`. If not specified, will use the command line option.
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
            self._device = device
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
        errors = {}
        wav = None

        try:
            wav = AudioFile(track).read(streams=0, samplerate=self._samplerate,
                                        channels=self._audio_channels)
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."

        if wav is None:
            try:
                wav, sr = ta.load(str(track))
            except RuntimeError as err:
                errors["torchaudio"] = err.args[0]
            else:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)

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


def list_models(repo: Optional[Path] = None) -> Dict[str, Dict[str, Union[str, Path]]]:
    """
    List the available models. Please remember that not all the returned models can be
    successfully loaded.

    Parameters
    ----------
    repo: The repo whose models are to be listed.

    Returns
    -------
    A dict with two keys ("single" for single models and "bag" for bag of models). The values are
    lists whose components are strs.
    """
    model_repo: ModelOnlyRepo
    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    return {"single": model_repo.list_model(), "bag": bag_repo.list_model()}


def _translate_source(source: str) -> str:
    """Translate a source name to English if translation exists."""
    return SOURCE_TRANSLATIONS.get(source.lower(), source)


def _detect_separation_type(sources: List[str]) -> Tuple[str, str]:
    """
    Detect the separation type based on the source names.

    Returns
    -------
    Tuple of (separation_type_key, separation_type_name)
    """
    # Translate sources to English for comparison
    sources_english = {_translate_source(s).lower() for s in sources}

    # Check each separation type
    for sep_type, info in SEPARATION_TYPES.items():
        if sep_type == 'unknown':
            continue
        type_sources = {s.lower() for s in info['sources']}
        # Check if sources match or are a subset
        if sources_english == type_sources or sources_english.issubset(type_sources):
            return sep_type, info['name']
        # Special case: check if it's a standard music separation
        if sources_english == {'drums', 'bass', 'other', 'vocals'}:
            return 'music_4stem', SEPARATION_TYPES['music_4stem']['name']

    return 'unknown', SEPARATION_TYPES['unknown']['name']


def get_model_info(
    name: str,
    repo: Optional[Path] = None,
    load_model_weights: bool = True
) -> ModelInfo:
    """
    Get detailed information about a model.

    Parameters
    ----------
    name : str
        Model name or signature (e.g., 'htdemucs', 'htdemucs_ft', '49469ca8').
    repo : Path, optional
        Folder containing pre-trained models for local models.
    load_model_weights : bool, default True
        If True, loads the full model to get accurate information.
        If False, returns only known metadata (faster but may be incomplete).

    Returns
    -------
    ModelInfo
        Detailed information about the model.

    Raises
    ------
    LoadModelError
        If the model cannot be loaded.

    Examples
    --------
    >>> info = get_model_info('htdemucs')
    >>> print(info)
    HT-Demucs (htdemucs)
      Type: Music Separation (4 stems)
      Architecture: HTDemucs (ensemble of 1)
      Sources: drums, bass, other, vocals
      Sample Rate: 44100 Hz
      Use Case: General music separation

    >>> info = get_model_info('49469ca8', repo=Path('/path/to/drumsep'))
    >>> print(info.sources_english)
    ['kick', 'snare', 'cymbals', 'toms']
    """
    # Get known metadata
    known = KNOWN_MODELS.get(name, {})
    display_name = known.get('display_name', name)
    description = known.get('description', '')
    use_case = known.get('use_case', '')

    if load_model_weights:
        # Load the model to get accurate information
        model = get_model(name=name, repo=repo)
        if model is None:
            raise LoadModelError(f"Failed to load model: {name}")

        # Check if it's a bag of models
        is_bag = isinstance(model, BagOfModels)
        if is_bag:
            num_models = len(model.models)
            base_model = model.models[0]
            architecture = base_model.__class__.__name__
            sources = list(base_model.sources)
            sample_rate = base_model.samplerate
            audio_channels = base_model.audio_channels
        else:
            num_models = 1
            architecture = model.__class__.__name__
            sources = list(model.sources)
            sample_rate = model.samplerate
            audio_channels = model.audio_channels
    else:
        # Return only known information
        sources = []
        architecture = 'Unknown'
        sample_rate = 44100
        audio_channels = 2
        is_bag = False
        num_models = 1

    # Translate sources to English
    sources_english = [_translate_source(s) for s in sources]

    # Detect separation type
    sep_type, sep_type_name = _detect_separation_type(sources)

    # Use separation type description if model description is empty
    if not description:
        description = SEPARATION_TYPES.get(sep_type, {}).get('description', '')

    # Generate use case from separation type if not specified
    if not use_case:
        use_case = sep_type_name

    return ModelInfo(
        name=name,
        display_name=display_name,
        architecture=architecture,
        sources=sources,
        sources_english=sources_english,
        separation_type=sep_type,
        separation_type_name=sep_type_name,
        description=description,
        use_case=use_case,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        is_bag=is_bag,
        num_models=num_models,
    )


def list_supported_separation_types() -> Dict[str, Dict]:
    """
    List all supported separation types.

    Returns
    -------
    Dict
        Dictionary mapping separation type keys to their information.

    Examples
    --------
    >>> types = list_supported_separation_types()
    >>> for key, info in types.items():
    ...     print(f"{key}: {info['name']}")
    music_4stem: Music Separation (4 stems)
    music_6stem: Music Separation (6 stems)
    drum_kit: Drum Kit Separation
    cinematic: Cinematic/Film Audio Separation
    ...
    """
    return {k: {'name': v['name'], 'description': v['description']}
            for k, v in SEPARATION_TYPES.items() if k != 'unknown'}


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
