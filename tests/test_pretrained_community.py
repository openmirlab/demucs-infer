"""Test that get_model and list_models include community models."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from demucs_infer.checkpoint_catalog import checkpoint_config, checkpoint_config_path
from demucs_infer.checkpoint_runtime import CheckpointRuntime
from demucs_infer.pretrained import get_model
from demucs_infer.api import KNOWN_MODELS, list_models, list_supported_separation_types


PUBLIC_MODEL_STEMS = {
    "uvr_demucs_model_1": ("vocals", "non_vocals"),
    "uvr_demucs_model_2": ("vocals", "non_vocals"),
    "uvr_demucs_model_bag": ("vocals", "non_vocals"),
    "cdx23_dnr": ("music", "sfx", "speech"),
    "msst_htdemucs_vocals": ("vocals", "other"),
    "drumsep": ("bombo", "redoblante", "platillos", "toms"),
}


def test_get_model_falls_through_to_community_for_unknown_registry_name(tmp_path):
    """Unknown registry names retain the historical GDrive fallback."""
    with patch('demucs_infer.pretrained.GDriveRepo') as MockGDrive:
        mock_repo = MagicMock()
        mock_repo.has_model.return_value = True
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_repo.get_model.return_value = mock_model
        MockGDrive.return_value = mock_repo

        get_model('legacy-community')
        mock_repo.get_model.assert_called_once_with('legacy-community')


def test_get_model_uses_runtime_for_registered_community_signature():
    """The migrated DrumSep signature no longer enters the GDrive fallback."""
    with patch('demucs_infer.checkpoint_runtime.CheckpointRuntime.load_registered_model') as load:
        model = MagicMock()
        load.return_value = model
        assert get_model('49469ca8') is model
        load.assert_called_once_with()


def test_list_models_includes_community():
    """list_models result includes community models."""
    result = list_models()
    assert 'community' in result
    assert '49469ca8' in result['community']


def test_registry_backed_listing_uses_real_config_paths_and_exact_stems(tmp_path):
    registry = checkpoint_config()
    listed = list_models()
    for name, stems in PUBLIC_MODEL_STEMS.items():
        recipe = registry.recipes[name]
        assert recipe.stems == stems
        assert CheckpointRuntime(name, cache_dir=tmp_path).resolve().recipe is recipe
        if name != "drumsep":
            assert Path(listed["bag"][name]) == checkpoint_config_path()

    assert "phantom_center" not in KNOWN_MODELS
    assert "stereo_center" not in list_supported_separation_types()


def test_readme_model_and_manual_download_tables_match_registry():
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    registry = checkpoint_config()
    for name, stems in PUBLIC_MODEL_STEMS.items():
        model_line = next(line for line in readme.splitlines() if f"`{name}`" in line)
        assert ", ".join(stems) in model_line

    for artifact_id in (
        "ebf34a2db", "ebf34a2d",
        "cdx23_dnr_a", "cdx23_dnr_b", "cdx23_dnr_c",
        "msst_htdemucs_vocals_state",
    ):
        artifact = registry.artifacts[artifact_id]
        assert artifact.path in readme
        assert artifact.urls[0] in readme
