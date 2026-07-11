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


def test_get_model_no_gdown(tmp_path):
    """GDriveRepo.get_model raises ImportError when gdown not installed."""
    repo = GDriveRepo(cache_dir=tmp_path)
    with patch.dict('sys.modules', {'gdown': None}):
        with pytest.raises(ImportError, match='gdown'):
            repo.get_model('49469ca8')


def test_get_model_uses_cache(tmp_path):
    """GDriveRepo loads from cache if file already exists."""
    repo = GDriveRepo(cache_dir=tmp_path)
    # Create a fake cached file
    fake_model = tmp_path / '49469ca8.th'
    fake_model.touch()

    with patch('demucs_infer.community.check_checksum') as mock_check, \
         patch('demucs_infer.community.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        repo.get_model('49469ca8')
        mock_check.assert_called_once_with(
            fake_model, COMMUNITY_MODELS['49469ca8']['sha256'])
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
         patch('demucs_infer.community.check_checksum') as mock_check, \
         patch('demucs_infer.community.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        repo.get_model('49469ca8')
        mock_gdown.download.assert_called_once()
        mock_check.assert_called_once()
        mock_load.assert_called_once()


def test_get_model_unknown_sig():
    """GDriveRepo.get_model raises ModelLoadingError for unknown signature."""
    from demucs_infer.repo import ModelLoadingError
    repo = GDriveRepo()
    with pytest.raises(ModelLoadingError):
        repo.get_model('nonexistent')


def test_get_model_rejects_bad_checksum(tmp_path):
    """GDriveRepo.get_model raises ModelLoadingError if the cached file's
    sha256 doesn't match the registry -- the integrity gap this phase closed
    (previously gdown downloads were never checksum-verified at all)."""
    from demucs_infer.repo import ModelLoadingError
    repo = GDriveRepo(cache_dir=tmp_path)
    fake_model = tmp_path / '49469ca8.th'
    fake_model.write_bytes(b'not the real checkpoint bytes')

    with pytest.raises(ModelLoadingError, match='Invalid checksum'):
        repo.get_model('49469ca8')


def test_get_model_verifies_real_cached_checkpoint(tmp_path):
    """End-to-end (no mocking of check_checksum/load_model): the real,
    already-cached drumsep checkpoint verifies and loads if present on this
    machine; skipped otherwise so CI without the cache still passes."""
    real_cached = Path.home() / '.cache' / 'demucs-infer' / '49469ca8.th'
    if not real_cached.exists():
        pytest.skip('drumsep checkpoint not cached locally')
    repo = GDriveRepo(cache_dir=real_cached.parent)
    model = repo.get_model('49469ca8')
    assert model is not None
