# GDrive Community Models Implementation Plan

> **For agentic workers:** Execute this plan task by task with scoped workers and verification gates. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `GDriveRepo` that auto-downloads community models (starting with Drumsep `49469ca8`) from Google Drive, making them first-class models that work without `--repo`.

**Architecture:** New `GDriveRepo` subclass of `ModelOnlyRepo` downloads `.th` files via `gdown` to `~/.cache/demucs-infer/`, then loads them like `LocalRepo`. A `COMMUNITY_MODELS` registry maps signatures to Google Drive file IDs. `get_model()` chains: `RemoteRepo` -> `GDriveRepo` -> fail.

**Tech Stack:** gdown (optional dependency), torch, pathlib

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `demucs_infer/community.py` | `GDriveRepo` class + `COMMUNITY_MODELS` registry |
| Modify | `demucs_infer/pretrained.py:59-85` | Chain `GDriveRepo` as fallback in `get_model()` |
| Modify | `demucs_infer/api.py:539-563` | Chain `GDriveRepo` in `list_models()` |
| Modify | `pyproject.toml:47-55` | Add `gdown` optional dependency |
| Create | `tests/test_community.py` | Unit tests for `GDriveRepo` |

---

### Task 1: Add gdown optional dependency

**Files:**
- Modify: `pyproject.toml:47-55`

- [ ] **Step 1: Add community optional dependency group**

In `pyproject.toml`, add after the `quantized` group:

```toml
# Optional: Community model downloads (Google Drive)
community = [
    "gdown>=5.0.0",
]
```

- [ ] **Step 2: Verify pyproject.toml syntax**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add gdown optional dependency for community models"
```

---

### Task 2: Create GDriveRepo and community model registry

**Files:**
- Create: `demucs_infer/community.py`
- Create: `tests/test_community.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_community.py`:

```python
"""Tests for community model registry and GDriveRepo."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from demucs_infer.community import COMMUNITY_MODELS, GDriveRepo


def test_drumsep_in_registry():
    """Drumsep model 49469ca8 must be registered."""
    assert '49469ca8' in COMMUNITY_MODELS
    entry = COMMUNITY_MODELS['49469ca8']
    assert 'gdrive_id' in entry
    assert entry['gdrive_id'] == '1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC'


def test_has_model_true():
    """GDriveRepo.has_model returns True for registered models."""
    repo = GDriveRepo()
    assert repo.has_model('49469ca8') is True


def test_has_model_false():
    """GDriveRepo.has_model returns False for unknown models."""
    repo = GDriveRepo()
    assert repo.has_model('nonexistent') is False


def test_list_model():
    """GDriveRepo.list_model returns all community models."""
    repo = GDriveRepo()
    models = repo.list_model()
    assert '49469ca8' in models


def test_cache_dir_default():
    """GDriveRepo uses ~/.cache/demucs-infer/ by default."""
    repo = GDriveRepo()
    assert repo.cache_dir == Path.home() / '.cache' / 'demucs-infer'


def test_get_model_no_gdown():
    """GDriveRepo.get_model raises ImportError when gdown not installed."""
    repo = GDriveRepo()
    with patch.dict('sys.modules', {'gdown': None}):
        with pytest.raises(ImportError, match='gdown'):
            repo.get_model('49469ca8')


def test_get_model_uses_cache(tmp_path):
    """GDriveRepo loads from cache if file already exists."""
    repo = GDriveRepo(cache_dir=tmp_path)
    # Create a fake cached file
    fake_model = tmp_path / '49469ca8.th'
    fake_model.touch()

    with patch('demucs_infer.community.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        repo.get_model('49469ca8')
        mock_load.assert_called_once_with(fake_model)


def test_get_model_downloads_if_not_cached(tmp_path):
    """GDriveRepo downloads via gdown when file not in cache."""
    repo = GDriveRepo(cache_dir=tmp_path)

    mock_gdown = MagicMock()
    # Simulate gdown creating the file
    def fake_download(id, output, quiet, fuzzy):
        Path(output).touch()
    mock_gdown.download.side_effect = fake_download

    with patch.dict('sys.modules', {'gdown': mock_gdown}), \
         patch('demucs_infer.community.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        repo.get_model('49469ca8')
        mock_gdown.download.assert_called_once()
        mock_load.assert_called_once()


def test_get_model_unknown_sig():
    """GDriveRepo.get_model raises ModelLoadingError for unknown signature."""
    from demucs_infer.repo import ModelLoadingError
    repo = GDriveRepo()
    with pytest.raises(ModelLoadingError):
        repo.get_model('nonexistent')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && python -m pytest tests/test_community.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demucs_infer.community'`

- [ ] **Step 3: Create `demucs_infer/community.py`**

```python
"""Community model repository — auto-downloads models from Google Drive.

Provides GDriveRepo, a ModelOnlyRepo subclass that downloads .th checkpoint
files via gdown to a local cache (~/.cache/demucs-infer/) on first use.
"""

import logging
from pathlib import Path
import typing as tp

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

    def get_model(self, sig: str):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && python -m pytest tests/test_community.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add demucs_infer/community.py tests/test_community.py
git commit -m "feat: add GDriveRepo for community model auto-download"
```

---

### Task 3: Wire GDriveRepo into get_model() and list_models()

**Files:**
- Modify: `demucs_infer/pretrained.py:59-85`
- Modify: `demucs_infer/api.py:539-563`
- Create: `tests/test_pretrained_community.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pretrained_community.py`:

```python
"""Test that get_model and list_models include community models."""
from unittest.mock import patch, MagicMock
from pathlib import Path

from demucs_infer.pretrained import get_model
from demucs_infer.api import list_models


def test_get_model_falls_through_to_community(tmp_path):
    """get_model tries GDriveRepo when RemoteRepo doesn't have the sig."""
    with patch('demucs_infer.pretrained.GDriveRepo') as MockGDrive:
        mock_repo = MagicMock()
        mock_repo.has_model.return_value = True
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_repo.get_model.return_value = mock_model
        MockGDrive.return_value = mock_repo

        model = get_model('49469ca8')
        mock_repo.get_model.assert_called_once_with('49469ca8')


def test_list_models_includes_community():
    """list_models result includes community models."""
    result = list_models()
    assert 'community' in result
    assert '49469ca8' in result['community']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && python -m pytest tests/test_pretrained_community.py -v`
Expected: FAIL — `get_model` doesn't try `GDriveRepo`, `list_models` has no `community` key

- [ ] **Step 3: Update `pretrained.py` — chain GDriveRepo in get_model()**

In `demucs_infer/pretrained.py`, add import at line 16:

```python
from .community import GDriveRepo
```

Then replace the `get_model` function (lines 59-85) with:

```python
def get_model(name: str,
              repo: tp.Optional[Path] = None):
    """`name` must be a bag of models name or a pretrained signature
    from the remote AWS model repo, a community model, or the specified
    local repo if `repo` is not None.
    """
    if name == 'demucs_unittest':
        return demucs_unittest()
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
    any_repo = AnyModelRepo(model_repo, bag_repo)

    # Try official repos first, fall back to community models
    try:
        model = any_repo.get_model(name)
    except (ModelLoadingError, ImportError) as exc:
        if isinstance(exc, ImportError) and 'diffq' in exc.args[0]:
            _check_diffq()
            raise
        # Try community repo (GDrive-hosted models)
        if repo is None:
            community_repo = GDriveRepo()
            if community_repo.has_model(name):
                model = community_repo.get_model(name)
            else:
                raise
        else:
            raise

    model.eval()
    return model
```

- [ ] **Step 4: Update `api.py` — add community models to list_models()**

In `demucs_infer/api.py`, add import at line 38:

```python
from .community import GDriveRepo
```

Then replace the `list_models` function (lines 539-563) with:

```python
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
        models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
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
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add demucs_infer/pretrained.py demucs_infer/api.py tests/test_pretrained_community.py
git commit -m "feat: wire GDriveRepo into get_model() and list_models()"
```

---

### Task 4: Smoke test — verify Drumsep loads end-to-end

**Files:**
- No file changes — manual verification

- [ ] **Step 1: Install with community extras**

Run: `cd /home/worzpro/Desktop/dev/openmirlab/demucs-infer && pip install -e ".[community]"`

- [ ] **Step 2: Verify model listing includes Drumsep**

Run: `python -c "from demucs_infer.api import list_models; m = list_models(); print('community:', m.get('community', {}))"`
Expected: `community: {'49469ca8': 'gdrive:1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC'}`

- [ ] **Step 3: Test model download and load**

Run: `python -c "from demucs_infer.pretrained import get_model; m = get_model('49469ca8'); print('sources:', m.sources); print('class:', m.__class__.__name__)"`
Expected: Downloads model, prints sources `['bombo', 'redoblante', 'platillos', 'toms']`

- [ ] **Step 4: Test CLI**

Run: `demucs-infer --list-models`
Expected: `49469ca8` appears in the output

- [ ] **Step 5: Commit version bump**

```bash
# Bump version to 4.2.0 for new feature
```

Update `pyproject.toml` version from `4.1.3` to `4.2.0`, then:

```bash
git add pyproject.toml
git commit -m "chore: bump version to 4.2.0 for community model support"
```
