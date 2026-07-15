"""Package-owned, release-pinned Demucs checkpoint metadata.

The TOML file is the source of truth.  This module keeps the historical
``CHECKPOINT_CATALOG`` mapping and ``get_checkpoint_metadata`` function as
read-only compatibility views for callers that imported them directly.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


def _legacy_toml_load(path):
    """Parse this deliberately simple TOML schema on Python 3.8-3.10."""
    import ast
    result = {}
    section = None
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]" ):
            section = line[2:-2]
            target = result
            parts = section.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            current = {}
            target.setdefault(parts[-1], []).append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            target = result
            for part in section.split("."):
                target = target.setdefault(part, {})
            current = target
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        (current if current is not None else target)[key] = ast.literal_eval(value)
    return result


def checkpoint_config_path():
    return Path(__file__).with_name("config") / "checkpoints.toml"


def _load_toml(path):
    if tomllib is None:
        return _legacy_toml_load(path)
    try:
        with Path(path).open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc


def _parse_config(path=None):
    path = Path(path) if path else checkpoint_config_path()
    config = _load_toml(path)
    if config.get("schema", {}).get("version") != 1:
        raise ValueError("unsupported checkpoint config schema version")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("checkpoint config must define models")
    entries = {}
    for name, metadata in models.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid metadata for {name}")
        signature = metadata.get("signature")
        files = metadata.get("files")
        if isinstance(files, list) and files:
            artifact = files[0]
            artifact_path = artifact.get("path")
            if not isinstance(artifact_path, str) or not artifact_path or Path(artifact_path).is_absolute():
                raise ValueError(f"invalid checkpoint path for {name}")
            url = artifact.get("url")
            digest = artifact.get("sha256")
            signature = signature or Path(artifact.get("path", "")).stem.split("-", 1)[0]
        else:
            artifact = metadata
            url = metadata.get("url")
            digest = metadata.get("sha256")
        if not isinstance(signature, str) or not signature:
            raise ValueError(f"model {name!r} must define signature")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"invalid checkpoint URL for {name}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"invalid checkpoint SHA-256 for {name}")
        entry = {"model": metadata.get("model", name), "url": url,
                 "sha256": digest.lower(), "signature": signature,
                 "path": artifact.get("path", Path(url).name),
                 "license": metadata.get("license", config.get("metadata", {}).get("license", "")),
                 "provenance": metadata.get("provenance", config.get("metadata", {}).get("provenance", "")),
                 "source_revision": metadata.get("source_revision", config.get("metadata", {}).get("source_revision", "")),
                 "updated_at": metadata.get("updated_at", config.get("metadata", {}).get("updated_at", ""))}
        entries[signature] = entry
    return {"raw": config, "entries": entries}


_CONFIG = _parse_config()
_ENTRIES = _CONFIG["entries"]
CHECKPOINT_CATALOG = MappingProxyType({k: MappingProxyType(v) for k, v in _ENTRIES.items()})


def get_checkpoint_metadata(key):
    """Return a mutable copy of pinned metadata for a signature or model name."""
    if key in _ENTRIES:
        return dict(_ENTRIES[key])
    for entry in _ENTRIES.values():
        if entry["model"] == key:
            return dict(entry)
    return None


def checkpoint_catalog():
    """Return a copy of all package-owned checkpoint metadata."""
    return {key: dict(value) for key, value in _ENTRIES.items()}


def validate_checkpoint_config(path=None):
    """Parse and validate a config file, returning its raw TOML mapping."""
    return _parse_config(path)["raw"]


__all__ = ["CHECKPOINT_CATALOG", "checkpoint_catalog", "checkpoint_config_path",
           "get_checkpoint_metadata", "validate_checkpoint_config"]
