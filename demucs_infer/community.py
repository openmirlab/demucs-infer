"""Community model repository — auto-downloads models from Google Drive.

Provides GDriveRepo, a ModelOnlyRepo subclass that downloads .th checkpoint
files via gdown to a local cache (~/.cache/demucs-infer/) on first use.
"""

import logging
from pathlib import Path
import typing as tp

from .apply import Model
from .repo import ModelOnlyRepo, ModelLoadingError
from .states import load_model

logger = logging.getLogger(__name__)

# Registry of community models: signature -> download metadata
COMMUNITY_MODELS: tp.Dict[str, tp.Dict[str, str]] = {
    '49469ca8': {
        'gdrive_id': '1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC',
        'name': 'Drumsep',
        'description': 'Drum kit separation (kick, snare, cymbals, toms)',
        'origin': 'https://github.com/inagoy/drumsep',
    },
}

DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'demucs-infer'


class GDriveRepo(ModelOnlyRepo):
    """Downloads community models from Google Drive on first use.

    Requires ``gdown`` (install with ``pip install demucs-infer[community]``).
    Downloaded checkpoints are cached in ``cache_dir`` so subsequent loads
    are instant.
    """

    def __init__(self, cache_dir: tp.Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def has_model(self, sig: str) -> bool:
        return sig in COMMUNITY_MODELS

    def get_model(self, sig: str) -> Model:
        if sig not in COMMUNITY_MODELS:
            raise ModelLoadingError(
                f'Could not find community model with signature {sig}.')

        entry = COMMUNITY_MODELS[sig]
        cached_path = self.cache_dir / f'{sig}.th'

        if not cached_path.exists():
            try:
                import gdown
            except ImportError:
                raise ImportError(
                    f'gdown is required to download community model "{entry["name"]}" ({sig}). '
                    'Install it with: pip install demucs-infer[community]'
                )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                'Downloading community model %s (%s) from Google Drive...',
                entry['name'], sig)
            gdown.download(
                id=entry['gdrive_id'],
                output=str(cached_path),
                quiet=False,
                fuzzy=False,
            )
            if not cached_path.exists():
                raise ModelLoadingError(
                    f'Failed to download community model {sig} from Google Drive.')

        return load_model(cached_path)

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return {sig: f'gdrive:{entry["gdrive_id"]}'
                for sig, entry in COMMUNITY_MODELS.items()}
