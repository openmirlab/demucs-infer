"""Parse the package-owned checkpoint registry and project compatibility views.

Schema v2 separates immutable physical artifacts from named model recipes.
Runtime code consumes those records, while the historical catalog mapping and
metadata lookup remain read-only compatibility surfaces for callers.

Reads: config/checkpoints.toml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
import typing as tp

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.8-3.10 fallback
    import tomli as tomllib


_ARTIFACT_FIELDS = {
    "id", "signature", "path", "urls", "sha256", "size", "license",
    "provenance", "source_revision", "updated_at",
}
_MODEL_FIELDS = {
    "name", "aliases", "components", "format", "weights", "segment",
    "stems", "display_name", "architecture",
}
_FORMATS = {"demucs_package", "state_dict"}

_HTDEMUCS_BOOL_FIELDS = {
    "cac", "emb_smooth", "rewrite", "t_auto_sparsity", "t_cape_augment",
    "t_cape_mean_normalize", "t_cross_first", "t_gelu", "t_group_norm",
    "t_layer_scale", "t_norm_first", "t_norm_in", "t_norm_in_group",
    "t_norm_out", "t_sparse_cross_attn", "t_sparse_self_attn",
    "wiener_residual",
}
_HTDEMUCS_INT_FIELDS = {
    "audio_channels", "bottom_channels", "channels", "context", "context_enc",
    "dconv_comp", "dconv_depth", "dconv_mode", "depth", "emb_scale",
    "end_iters", "growth", "kernel_size", "multi_freqs_depth", "nfft",
    "norm_groups", "norm_starts", "samplerate", "segment", "stride",
    "t_global_window", "t_heads", "t_layers", "t_mask_random_seed",
    "t_max_positions", "t_sin_random_shift", "t_sparse_attn_window",
    "time_stride", "wiener_iters",
}
_HTDEMUCS_FLOAT_FIELDS = {
    "dconv_init", "freq_emb", "rescale", "t_dropout", "t_hidden_scale",
    "t_max_period", "t_sparsity", "t_weight_decay", "t_weight_pos_embed",
}
_HTDEMUCS_STRING_FIELDS = {"t_emb", "t_mask_type"}
_HTDEMUCS_LIST_FIELDS = {"multi_freqs", "sources", "t_cape_glob_loc_scale"}
_HTDEMUCS_FIELDS = (
    _HTDEMUCS_BOOL_FIELDS | _HTDEMUCS_INT_FIELDS | _HTDEMUCS_FLOAT_FIELDS
    | _HTDEMUCS_STRING_FIELDS | _HTDEMUCS_LIST_FIELDS
)


@dataclass(frozen=True)
class CheckpointArtifact:
    id: str
    signature: str
    path: str
    urls: tp.Tuple[str, ...]
    sha256: str
    size: tp.Optional[int]
    license: str
    provenance: str
    source_revision: str
    updated_at: str


@dataclass(frozen=True)
class ArchitectureRecipe:
    name: str
    kwargs: tp.Mapping[str, tp.Any]


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    aliases: tp.Tuple[str, ...]
    components: tp.Tuple[str, ...]
    format: str
    weights: tp.Optional[tp.Tuple[tp.Tuple[float, ...], ...]]
    segment: tp.Optional[float]
    stems: tp.Tuple[str, ...]
    display_name: tp.Optional[str]
    architecture: tp.Optional[ArchitectureRecipe]


@dataclass(frozen=True)
class CheckpointConfig:
    artifacts: tp.Mapping[str, CheckpointArtifact]
    signatures: tp.Mapping[str, CheckpointArtifact]
    recipes: tp.Mapping[str, ModelRecipe]
    aliases: tp.Mapping[str, ModelRecipe]


def checkpoint_config_path():
    return Path(__file__).with_name("config") / "checkpoints.toml"


def _load_toml(path):
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc


def _required_string(record, field, owner):
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must define non-empty {field}")
    return value


def _string_list(record, field, owner, *, required=False):
    value = record.get(field, [])
    if not isinstance(value, list) or (required and not value):
        raise ValueError(f"{owner} must define {'non-empty ' if required else ''}{field} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{owner} has invalid {field}")
    return value


def _parse_artifact(record, index):
    owner = f"artifact #{index}"
    if not isinstance(record, dict):
        raise ValueError(f"invalid {owner}")
    unknown = set(record) - _ARTIFACT_FIELDS
    if unknown:
        raise ValueError(f"{owner} has unsupported fields: {sorted(unknown)}")
    artifact_id = _required_string(record, "id", owner)
    signature = _required_string(record, "signature", owner)
    cache_path = _required_string(record, "path", owner)
    path = Path(cache_path)
    windows_path = PureWindowsPath(cache_path)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or cache_path in {".", ".."}
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"invalid checkpoint path for {artifact_id}")
    urls = _string_list(record, "urls", owner, required=True)
    if any(not url.startswith("https://") for url in urls):
        raise ValueError(f"invalid checkpoint URLs for {artifact_id}")
    digest = _required_string(record, "sha256", owner)
    if len(digest) != 64 or digest != digest.lower() or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"invalid checkpoint SHA-256 for {artifact_id}")
    if "-" in path.stem:
        embedded = path.stem.split("-", 1)[1]
        if not digest.startswith(embedded):
            raise ValueError(f"checkpoint hash prefix mismatch for {artifact_id}")
    size = record.get("size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError(f"invalid checkpoint size for {artifact_id}")
    return CheckpointArtifact(
        id=artifact_id,
        signature=signature,
        path=cache_path,
        urls=tuple(urls),
        sha256=digest,
        size=size,
        license=_required_string(record, "license", owner),
        provenance=_required_string(record, "provenance", owner),
        source_revision=_required_string(record, "source_revision", owner),
        updated_at=_required_string(record, "updated_at", owner),
    )


def _parse_weights(value, component_count, owner):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != component_count:
        raise ValueError(f"{owner} weights must match component count")
    width = None
    rows = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError(f"{owner} has invalid weights")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise ValueError(f"{owner} has non-numeric weights")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{owner} weight rows must have equal dimensions")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _parse_architecture(value, owner):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"name", "kwargs"}:
        raise ValueError(f"{owner} has invalid architecture recipe")
    if value["name"] != "HTDemucs":
        raise ValueError(f"unsupported architecture for {owner}: {value['name']}")
    kwargs = value["kwargs"]
    if not isinstance(kwargs, dict):
        raise ValueError(f"{owner} architecture kwargs must be a table")
    missing = _HTDEMUCS_FIELDS - set(kwargs)
    unknown = set(kwargs) - _HTDEMUCS_FIELDS
    if missing or unknown:
        raise ValueError(
            f"{owner} HTDemucs constructor fields differ: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    normalized = {}
    for field in _HTDEMUCS_BOOL_FIELDS:
        item = kwargs[field]
        if type(item) is not bool:
            raise ValueError(f"{owner} HTDemucs field {field} must be bool")
        normalized[field] = item
    for field in _HTDEMUCS_INT_FIELDS:
        item = kwargs[field]
        if type(item) is not int:
            raise ValueError(f"{owner} HTDemucs field {field} must be int")
        normalized[field] = item
    for field in _HTDEMUCS_FLOAT_FIELDS:
        item = kwargs[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{owner} HTDemucs field {field} must be numeric")
        normalized[field] = float(item)
    for field in _HTDEMUCS_STRING_FIELDS:
        item = kwargs[field]
        if not isinstance(item, str) or not item:
            raise ValueError(f"{owner} HTDemucs field {field} must be non-empty string")
        normalized[field] = item
    sources = kwargs["sources"]
    if not isinstance(sources, list) or not sources or any(
        not isinstance(item, str) or not item for item in sources
    ):
        raise ValueError(f"{owner} HTDemucs sources must be non-empty strings")
    normalized["sources"] = tuple(sources)
    for field in ("multi_freqs", "t_cape_glob_loc_scale"):
        items = kwargs[field]
        if not isinstance(items, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in items
        ):
            raise ValueError(f"{owner} HTDemucs field {field} must be numeric list")
        normalized[field] = tuple(float(item) for item in items)
    if len(normalized["t_cape_glob_loc_scale"]) != 3:
        raise ValueError(f"{owner} HTDemucs t_cape_glob_loc_scale must have three values")
    return ArchitectureRecipe("HTDemucs", MappingProxyType(normalized))


def _parse_recipe(record, index, artifact_ids):
    owner = f"model recipe #{index}"
    if not isinstance(record, dict):
        raise ValueError(f"invalid {owner}")
    unknown = set(record) - _MODEL_FIELDS
    if unknown:
        raise ValueError(f"{owner} has unsupported fields: {sorted(unknown)}")
    name = _required_string(record, "name", owner)
    aliases = _string_list(record, "aliases", owner)
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"duplicate aliases for model {name}")
    if name in aliases:
        raise ValueError(f"model name collides with alias: {name}")
    components = _string_list(record, "components", owner, required=True)
    missing = [component for component in components if component not in artifact_ids]
    if missing:
        raise ValueError(f"model {name} references missing artifacts: {missing}")
    model_format = _required_string(record, "format", owner)
    if model_format not in _FORMATS:
        raise ValueError(f"unsupported checkpoint format for model {name}: {model_format}")
    weights = _parse_weights(record.get("weights"), len(components), owner)
    architecture = _parse_architecture(record.get("architecture"), owner)
    if model_format == "state_dict":
        if architecture is None or len(components) != 1 or weights is not None:
            raise ValueError(
                f"state_dict model {name} requires one component, architecture, and no weights"
            )
    elif architecture is not None:
        raise ValueError(f"demucs_package model {name} cannot define architecture")
    segment = record.get("segment")
    if segment is not None and (
        isinstance(segment, bool) or not isinstance(segment, (int, float)) or segment <= 0
    ):
        raise ValueError(f"invalid segment for model {name}")
    stems = _string_list(record, "stems", owner)
    if len(set(stems)) != len(stems):
        raise ValueError(f"duplicate stems for model {name}")
    if weights is not None and stems and len(weights[0]) != len(stems):
        raise ValueError(f"model {name} weight dimensions do not match stems")
    display_name = record.get("display_name")
    if display_name is not None and (not isinstance(display_name, str) or not display_name):
        raise ValueError(f"invalid display_name for model {name}")
    return ModelRecipe(
        name=name,
        aliases=tuple(aliases),
        components=tuple(components),
        format=model_format,
        weights=weights,
        segment=float(segment) if segment is not None else None,
        stems=tuple(stems),
        display_name=display_name,
        architecture=architecture,
    )


def _parse_config(path=None):
    path = Path(path) if path else checkpoint_config_path()
    raw = _load_toml(path)
    if set(raw) - {"schema", "package", "artifacts", "models"}:
        raise ValueError("unsupported top-level checkpoint config fields")
    if raw.get("schema", {}).get("version") != 2:
        raise ValueError("unsupported checkpoint config schema version")
    records = raw.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint config must define artifacts")
    artifacts = {}
    signatures = {}
    for index, record in enumerate(records, 1):
        artifact = _parse_artifact(record, index)
        if artifact.id in artifacts:
            raise ValueError(f"duplicate artifact ID: {artifact.id}")
        if artifact.signature in signatures:
            raise ValueError(f"duplicate artifact signature: {artifact.signature}")
        artifacts[artifact.id] = artifact
        signatures[artifact.signature] = artifact
    model_records = raw.get("models")
    if not isinstance(model_records, list) or not model_records:
        raise ValueError("checkpoint config must define model recipes")
    recipes = {}
    aliases = {}
    occupied = set()
    for index, record in enumerate(model_records, 1):
        recipe = _parse_recipe(record, index, artifacts)
        names = (recipe.name,) + recipe.aliases
        collision = next((name for name in names if name in occupied), None)
        if collision is not None:
            raise ValueError(f"duplicate model name or alias: {collision}")
        occupied.update(names)
        recipes[recipe.name] = recipe
        for alias in recipe.aliases:
            aliases[alias] = recipe
    config = CheckpointConfig(
        artifacts=MappingProxyType(artifacts),
        signatures=MappingProxyType(signatures),
        recipes=MappingProxyType(recipes),
        aliases=MappingProxyType(aliases),
    )
    return raw, config


_RAW_CONFIG, _CONFIG = _parse_config()


def checkpoint_config():
    """Return the immutable internal schema-v2 registry."""
    return _CONFIG


def get_checkpoint_artifact(key):
    """Return an immutable artifact by stable ID or physical signature."""
    return _CONFIG.artifacts.get(key) or _CONFIG.signatures.get(key)


def get_model_recipe(key):
    """Return an immutable named recipe by public name or alias."""
    return _CONFIG.recipes.get(key) or _CONFIG.aliases.get(key)


def _artifact_model_name(artifact_id):
    for recipe in _CONFIG.recipes.values():
        if artifact_id in recipe.components:
            return recipe.name
    return artifact_id


def _compatibility_entry(artifact, *, model=None):
    return {
        "model": model or _artifact_model_name(artifact.id),
        "url": artifact.urls[0],
        "sha256": artifact.sha256,
        "signature": artifact.signature,
        "path": artifact.path,
        "license": artifact.license,
        "provenance": artifact.provenance,
        "source_revision": artifact.source_revision,
        "updated_at": artifact.updated_at,
    }


CHECKPOINT_CATALOG = MappingProxyType({
    artifact.signature: MappingProxyType(_compatibility_entry(artifact))
    for artifact in _CONFIG.artifacts.values()
})


def get_checkpoint_metadata(key):
    """Return a mutable compatibility copy for a signature or model name."""
    artifact = get_checkpoint_artifact(key)
    if artifact is not None:
        return _compatibility_entry(artifact)
    recipe = get_model_recipe(key)
    if recipe is None:
        return None
    artifact = _CONFIG.artifacts[recipe.components[0]]
    return _compatibility_entry(artifact, model=recipe.name)


def checkpoint_catalog():
    """Return a mutable copy of package-owned compatibility metadata."""
    return {key: dict(value) for key, value in CHECKPOINT_CATALOG.items()}


def validate_checkpoint_config(path=None):
    """Parse and validate a config file, returning its raw TOML mapping."""
    raw, _ = _parse_config(path)
    return raw


__all__ = [
    "CHECKPOINT_CATALOG", "checkpoint_catalog", "checkpoint_config_path",
    "get_checkpoint_metadata", "validate_checkpoint_config",
]
