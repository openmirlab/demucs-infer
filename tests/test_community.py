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
