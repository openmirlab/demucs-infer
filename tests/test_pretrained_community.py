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
