# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Model discovery/metadata layer: what models exist, what they're called,
and how their stems categorize -- split out of api.py (ADOPT campaign P4;
api.py was the 3rd-largest file at 783 lines and this was its cleanest
seam: describing/cataloging models is a distinct concern from actually
running separation, which stays in api.py's `Separator`).

`api.py` re-exports everything here (`ModelInfo`, `LoadModelError`,
`list_models`, `get_model_info`, `list_supported_separation_types`,
`KNOWN_MODELS`, `SEPARATION_TYPES`, `SOURCE_TRANSLATIONS`) so
`from demucs_infer.api import get_model_info` (used by README.md's
documented API and by separate.py/tests for `list_models`) keeps working
unchanged. This module never imports from api.py, so the re-export is a
one-directional (non-circular) dependency.

Reads: apply (BagOfModels), checkpoint_catalog, community (GDriveRepo),
log (fatal), pretrained (get_model, REMOTE_ROOT), repo (LocalRepo,
ModelOnlyRepo, BagOnlyRepo)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .apply import BagOfModels
from .checkpoint_catalog import checkpoint_config, checkpoint_config_path
from .community import GDriveRepo
from .log import fatal
from .pretrained import get_model, REMOTE_ROOT
from .repo import LocalRepo, ModelOnlyRepo, BagOnlyRepo


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
    '04573f0d': {
        'display_name': 'HTDemucs (MDX23)',
        'description': 'HTDemucs variant used in MVSep-MDX23 ensemble',
        'use_case': 'Music separation',
    },
}

# Stable recipe names and display labels come from the packaged registry. Keep
# richer descriptions above only where they add user guidance beyond the
# registry-owned identity.
for _recipe in checkpoint_config().recipes.values():
    _metadata = KNOWN_MODELS.setdefault(_recipe.name, {})
    if _recipe.display_name is not None:
        _metadata['display_name'] = _recipe.display_name


class LoadModelError(Exception):
    pass


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


def list_models(repo: Optional[Path] = None) -> Dict[str, Dict[str, Union[str, Path]]]:
    """
    List the available models. Please remember that not all the returned models can be
    successfully loaded.

    Parameters
    ----------
    repo: The repo whose models are to be listed.

    Returns
    -------
    A dict with keys "single" (single models), "bag" (bag of models), and
    "community" (community models downloadable from Google Drive).
    """
    model_repo: ModelOnlyRepo
    if repo is None:
        registry = checkpoint_config()
        community_signatures = set(GDriveRepo().list_model())
        single = {
            artifact.signature: artifact.urls[0]
            for artifact in registry.artifacts.values()
            if artifact.signature not in community_signatures
        }
        bag = {}
        for recipe in registry.recipes.values():
            if recipe.components[0] in community_signatures:
                continue
            legacy_yaml = REMOTE_ROOT / f"{recipe.name}.yaml"
            bag[recipe.name] = (
                legacy_yaml if legacy_yaml.is_file() else checkpoint_config_path()
            )
        result = {"single": single, "bag": bag}
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
        result = {"single": model_repo.list_model(), "bag": bag_repo.list_model()}

    # Include community models when listing from default repo
    if repo is None:
        community_repo = GDriveRepo()
        result["community"] = community_repo.list_model()

    return result


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
