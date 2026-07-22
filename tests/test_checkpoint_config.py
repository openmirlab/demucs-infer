"""Schema-v2 registry, oracle-equivalence, and fail-closed contract tests."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from demucs_infer.checkpoint_catalog import (
    checkpoint_catalog,
    checkpoint_config,
    checkpoint_config_path,
    get_model_recipe,
    validate_checkpoint_config,
)
from demucs_infer.pretrained import REMOTE_ROOT, _parse_remote_files
from demucs_infer.checkpoint_runtime import CheckpointRuntime


REPO_ROOT = Path(__file__).parent.parent


def test_package_config_is_shipped_and_complete():
    path = checkpoint_config_path()
    assert path.name == "checkpoints.toml"
    config = validate_checkpoint_config()
    assert config["package"]["name"] == "demucs-infer"
    assert config["schema"]["version"] == 2
    assert len(config["artifacts"]) == 34
    assert len(config["models"]) == 17
    assert len(checkpoint_catalog()) == 34
    assert all(item["url"].startswith("https://") for item in checkpoint_catalog().values())


def test_official_artifacts_and_bags_match_packaged_legacy_oracles_exactly():
    registry = checkpoint_config()
    official = _parse_remote_files(REMOTE_ROOT / "files.txt")
    assert len(official) == 27
    assert {
        signature: artifact.urls[0]
        for signature, artifact in registry.signatures.items()
        if signature in official
    } == official

    yaml_paths = sorted(REMOTE_ROOT.glob("*.yaml"))
    assert len(yaml_paths) == 11
    for path in yaml_paths:
        expected = yaml.safe_load(path.read_text(encoding="utf-8"))
        recipe = get_model_recipe(path.stem)
        assert list(recipe.components) == expected["models"]
        actual_weights = (
            [list(row) for row in recipe.weights]
            if recipe.weights is not None else None
        )
        assert actual_weights == expected.get("weights")
        assert recipe.segment == expected.get("segment")


def test_runtime_resolves_every_recipe_and_physical_signature_from_v2(tmp_path):
    registry = checkpoint_config()
    for recipe in registry.recipes.values():
        resolution = CheckpointRuntime(recipe.name, cache_dir=tmp_path).resolve()
        assert resolution.mode == "named_recipe"
        assert tuple(artifact.id for artifact in resolution.artifacts) == recipe.components
        assert resolution.paths == tuple(
            tmp_path / registry.artifacts[item].path for item in recipe.components
        )
    for signature, artifact in registry.signatures.items():
        resolution = CheckpointRuntime(signature, cache_dir=tmp_path).resolve()
        assert resolution.mode == "named_artifact"
        assert resolution.artifacts == (artifact,)
        assert resolution.paths == (tmp_path / artifact.path,)
    assert CheckpointRuntime("Drumsep", cache_dir=tmp_path).resolve().recipe.name == "drumsep"


def test_all_hashes_are_full_and_match_embedded_filename_prefixes():
    registry = checkpoint_config()
    for artifact in registry.artifacts.values():
        assert len(artifact.sha256) == 64
        assert artifact.sha256 == artifact.sha256.lower()
        if "-" in Path(artifact.path).stem:
            assert artifact.sha256.startswith(Path(artifact.path).stem.split("-", 1)[1])


def test_provenance_record_matches_every_schema_v2_artifact():
    provenance = json.loads(
        (REPO_ROOT / "docs/checkpoints_provenance.json").read_text(encoding="utf-8")
    )
    registry = checkpoint_config()
    assert len(provenance["official"]) == 27
    assert all(entry["sha256"] for entry in provenance["official"].values())
    for signature, entry in provenance["official"].items():
        artifact = registry.signatures[signature]
        assert entry["url"] == artifact.urls[0]
        assert entry["sha256"] == artifact.sha256
        assert entry["prefix_matches_embedded"] is True
    drumsep = provenance["community"]["49469ca8"]
    artifact = registry.signatures["49469ca8"]
    assert drumsep["sha256"] == artifact.sha256
    assert drumsep["verified_https_url"] == artifact.urls[0]
    expected_third_party = {
        "ebf34a2db", "ebf34a2d",
        "cdx23_dnr_a", "cdx23_dnr_b", "cdx23_dnr_c",
        "msst_htdemucs_vocals_state",
    }
    assert set(provenance["third_party"]) == expected_third_party
    for artifact_id, entry in provenance["third_party"].items():
        artifact = registry.artifacts[artifact_id]
        assert entry == {
            "license": artifact.license,
            "path": artifact.path,
            "provenance": artifact.provenance,
            "sha256": artifact.sha256,
            "source_revision": artifact.source_revision,
            "updated_at": artifact.updated_at,
            "urls": list(artifact.urls),
        }


def test_phase0_catalog_entries_remain_exact_while_v2_is_additive():
    baseline = json.loads(
        (REPO_ROOT / "tests/fixtures/public_contract.json").read_text(encoding="utf-8")
    )["checkpoint_catalog"]
    actual = checkpoint_catalog()
    assert len(actual) > len(baseline)
    assert {key: actual[key] for key in baseline} == baseline


@pytest.mark.parametrize(
    "old,new,match",
    [
        ('name = "HTDemucs"', 'name = "ArbitraryClass"', "unsupported architecture"),
        ('channels = 48', 'channels = "48"', "channels must be int"),
        ('channels = 48', 'unknown_field = 1, channels = 48', "unknown=.*unknown_field"),
    ],
)
def test_state_dict_architecture_recipe_fails_closed(tmp_path, old, new, match):
    text = checkpoint_config_path().read_text(encoding="utf-8").replace(old, new, 1)
    path = tmp_path / "invalid-architecture.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        validate_checkpoint_config(path)


def test_invalid_metadata_is_rejected(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text('[schema]\nversion = 1\n[models.bad]\nsignature = "bad"\nurl = "http://insecure"\nsha256 = "nope"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_checkpoint_config(path)


_VALID_CONFIG = '''
[schema]
version = 2
[package]
name = "demucs-infer"
[[artifacts]]
id = "artifact-a"
signature = "aaaaaaaa"
path = "aaaaaaaa-deadbeef.th"
urls = ["https://example.test/a.th"]
sha256 = "deadbeef00000000000000000000000000000000000000000000000000000000"
license = "MIT"
provenance = "fixture"
source_revision = "v1"
updated_at = "2026-07-22"
[[models]]
name = "model-a"
aliases = ["alias-a"]
components = ["artifact-a"]
format = "demucs_package"
'''


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda text: text.replace("version = 2", "version = 1"), "schema version"),
        (lambda text: text.replace('path = "aaaaaaaa-deadbeef.th"', 'path = "../a.th"'), "path"),
        (lambda text: text.replace('path = "aaaaaaaa-deadbeef.th"', 'path = "/a.th"'), "path"),
        (lambda text: text.replace('path = "aaaaaaaa-deadbeef.th"', 'path = "C:\\\\a.th"'), "path"),
        (lambda text: text.replace('urls = ["https://example.test/a.th"]', "urls = []"), "urls"),
        (lambda text: text.replace("https://example.test", "http://example.test"), "URLs"),
        (lambda text: text.replace("deadbeef" + "0" * 56, "bad"), "SHA-256"),
        (lambda text: text.replace('components = ["artifact-a"]', 'components = ["missing"]'), "missing artifacts"),
        (lambda text: text.replace('format = "demucs_package"', 'format = "pickle"'), "unsupported"),
        (lambda text: text + 'segment = 0\n', "segment"),
        (lambda text: text + 'weights = [[1.0], [1.0]]\n', "component count"),
        (lambda text: text.replace('components = ["artifact-a"]', 'components = ["artifact-a", "artifact-a"]') + 'weights = [[1.0, 0.0], [1.0]]\n', "equal dimensions"),
        (lambda text: text + 'weights = [[1.0, 0.0]]\nstems = ["one"]\n', "dimensions do not match stems"),
        (lambda text: text + 'mystery = true\n', "unsupported fields"),
        (lambda text: text + '\n[[models]]\nname = "model-b"\naliases = ["alias-a"]\ncomponents = ["artifact-a"]\nformat = "demucs_package"\n', "duplicate model"),
        (lambda text: text.replace('aliases = ["alias-a"]', 'aliases = ["alias-a", "alias-a"]'), "duplicate aliases"),
        (lambda text: text.replace('aliases = ["alias-a"]', 'aliases = ["model-a"]'), "collides with alias"),
        (lambda text: text.replace('id = "artifact-a"', 'id = "artifact-a"', 1) + '\n[[artifacts]]\nid = "artifact-a"\nsignature = "bbbbbbbb"\npath = "b.th"\nurls = ["https://example.test/b.th"]\nsha256 = "' + '0' * 64 + '"\nlicense = "MIT"\nprovenance = "fixture"\nsource_revision = "v1"\nupdated_at = "2026-07-22"\n', "duplicate artifact ID"),
        (lambda text: text + '\n[[artifacts]]\nid = "artifact-b"\nsignature = "aaaaaaaa"\npath = "b.th"\nurls = ["https://example.test/b.th"]\nsha256 = "' + '0' * 64 + '"\nlicense = "MIT"\nprovenance = "fixture"\nsource_revision = "v1"\nupdated_at = "2026-07-22"\n', "duplicate artifact signature"),
    ],
)
def test_schema_v2_validation_fails_closed(tmp_path, mutate, match):
    path = tmp_path / "invalid-v2.toml"
    path.write_text(mutate(_VALID_CONFIG), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        validate_checkpoint_config(path)


def test_custom_url_download_is_verified(tmp_path, monkeypatch):
    from demucs_infer.clean_api import DemucsSeparator
    payload = b"checkpoint-fixture"
    digest = hashlib.sha256(payload).hexdigest()

    class Response:
        def __init__(self, content):
            self.payload = content

        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size):
            chunk, self.payload = self.payload[:size], self.payload[size:]
            return chunk

    monkeypatch.setattr("demucs_infer.checkpoint_runtime.urlopen", lambda *a, **k: Response(payload))
    session = DemucsSeparator(checkpoint_url="https://example.test/model.th", checkpoint_sha256=digest, cache_dir=tmp_path)
    path, signature = session._materialize_checkpoint()
    assert path.read_bytes() == payload
    assert signature == "model"
