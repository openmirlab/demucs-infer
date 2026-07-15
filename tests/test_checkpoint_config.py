"""Package-owned checkpoint TOML contract tests."""
import hashlib

import pytest

from demucs_infer.checkpoint_catalog import checkpoint_catalog, checkpoint_config_path, validate_checkpoint_config


def test_package_config_is_shipped_and_complete():
    path = checkpoint_config_path()
    assert path.name == "checkpoints.toml"
    config = validate_checkpoint_config()
    assert config["package"]["name"] == "demucs-infer"
    assert len(checkpoint_catalog()) >= 6
    assert all(item["url"].startswith("https://") for item in checkpoint_catalog().values())


def test_invalid_metadata_is_rejected(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text('[schema]\nversion = 1\n[models.bad]\nsignature = "bad"\nurl = "http://insecure"\nsha256 = "nope"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_checkpoint_config(path)


def test_custom_url_download_is_verified(tmp_path, monkeypatch):
    from demucs_infer.clean_api import DemucsSeparator
    payload = b"checkpoint-fixture"
    digest = hashlib.sha256(payload).hexdigest()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return payload

    monkeypatch.setattr("demucs_infer.clean_api.urlopen", lambda *a, **k: Response())
    session = DemucsSeparator(checkpoint_url="https://example.test/model.th", checkpoint_sha256=digest, cache_dir=tmp_path)
    path, signature = session._materialize_checkpoint()
    assert path.read_bytes() == payload
    assert signature == "model"
