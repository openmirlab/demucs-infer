"""Resolve, materialize, and load package-owned checkpoint recipes.

The immutable schema-v2 registry is the only authority for migrated model
names, aliases, physical signatures, cache paths, and bag assembly. Both the
clean facade and legacy default resolver enter through this module; unknown
names alone fall back to the historical repository chain.

Reads: checkpoint_catalog, states, htdemucs.HTDemucs, apply.BagOfModels,
api.Separator.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import typing as tp
from urllib.request import urlopen

from .checkpoint_catalog import (
    CheckpointArtifact,
    ModelRecipe,
    checkpoint_config,
    get_checkpoint_artifact,
    get_checkpoint_metadata,
    get_model_recipe,
)


_IO_CHUNK_SIZE = 1024 * 1024


class RegistryModelNotFound(LookupError):
    """Raised when a name is intentionally outside the schema-v2 registry."""


@dataclass(frozen=True)
class _Resolution:
    mode: str
    paths: tp.Tuple[Path, ...]
    artifacts: tp.Tuple[CheckpointArtifact, ...] = ()
    recipe: tp.Optional[ModelRecipe] = None
    signature: tp.Optional[str] = None
    expected_sha256: tp.Optional[str] = None
    reported_sha256: tp.Optional[str] = None
    checkpoint_url: tp.Optional[str] = None


class CheckpointRuntime:
    """Package-internal owner of checkpoint resolution and loading policy."""

    def __init__(
        self,
        model_name="htdemucs",
        *,
        checkpoint_path=None,
        checkpoint_url=None,
        checkpoint_sha256=None,
        cache_dir=None,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _verify(self, path, expected):
        digest = sha256()
        with path.open("rb") as checkpoint:
            while True:
                chunk = checkpoint.read(_IO_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")

    def _cache_root(self, *, create=True):
        root = self.cache_dir or (Path.home() / ".cache" / "demucs-infer")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _download_checkpoint(self, urls, path, expected):
        if path.exists():
            try:
                self._verify(path, expected)
                return path
            except ValueError:
                path.unlink()

        path.parent.mkdir(parents=True, exist_ok=True)
        candidates = (urls,) if isinstance(urls, str) else tuple(urls)
        last_error = None
        for url in candidates:
            temporary_path = None
            try:
                with urlopen(url) as source:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=path.parent,
                        prefix=f".{path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as target:
                        temporary_path = Path(target.name)
                        while True:
                            chunk = source.read(_IO_CHUNK_SIZE)
                            if not chunk:
                                break
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                self._verify(temporary_path, expected)
                os.replace(temporary_path, path)
                return path
            except BaseException as error:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                last_error = error
        if last_error is not None:
            raise last_error
        raise ValueError("checkpoint artifact has no download URLs")

    def resolve(self):
        """Describe the exact current load paths without network or writes."""
        metadata = get_checkpoint_metadata(self.model_name)
        if self.checkpoint_path is not None:
            signature = self.checkpoint_path.stem.split("-", 1)[0]
            expected = self.checkpoint_sha256 or (
                get_checkpoint_metadata(signature) or {}
            ).get("sha256")
            return _Resolution(
                "checkpoint_path",
                (self.checkpoint_path,),
                signature=signature,
                expected_sha256=expected,
                reported_sha256=self.checkpoint_sha256 or (metadata or {}).get("sha256"),
                checkpoint_url=self.checkpoint_url or (metadata or {}).get("url"),
            )
        if self.checkpoint_url:
            path = self._cache_root(create=False) / Path(self.checkpoint_url).name
            return _Resolution(
                "checkpoint_url",
                (path,),
                signature=path.stem.split("-", 1)[0],
                expected_sha256=self.checkpoint_sha256,
                reported_sha256=self.checkpoint_sha256 or (metadata or {}).get("sha256"),
                checkpoint_url=self.checkpoint_url,
            )

        recipe = get_model_recipe(self.model_name)
        if recipe is not None:
            registry = checkpoint_config()
            artifacts = tuple(registry.artifacts[item] for item in recipe.components)
            root = self._cache_root(create=False)
            return _Resolution(
                "named_recipe",
                tuple(root / artifact.path for artifact in artifacts),
                artifacts=artifacts,
                recipe=recipe,
                signature=recipe.name,
                expected_sha256=artifacts[0].sha256,
                reported_sha256=artifacts[0].sha256,
                checkpoint_url=artifacts[0].urls[0],
            )
        artifact = get_checkpoint_artifact(self.model_name)
        if artifact is not None:
            root = self._cache_root(create=False)
            return _Resolution(
                "named_artifact",
                (root / artifact.path,),
                artifacts=(artifact,),
                signature=artifact.signature,
                expected_sha256=artifact.sha256,
                reported_sha256=artifact.sha256,
                checkpoint_url=artifact.urls[0],
            )
        return _Resolution("legacy", ())

    def materialize_override(self, resolution=None):
        """Materialize only an explicit path/URL override, preserving its tuple seam."""
        resolution = resolution or self.resolve()
        if resolution.mode == "checkpoint_url":
            if not resolution.expected_sha256:
                raise ValueError("checkpoint_sha256 is required with checkpoint_url")
            path = self._cache_root() / resolution.paths[0].name
            self._download_checkpoint(
                resolution.checkpoint_url,
                path,
                resolution.expected_sha256,
            )
            return path, resolution.signature
        if resolution.mode == "checkpoint_path":
            path = resolution.paths[0]
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {path}")
            if resolution.expected_sha256:
                self._verify(path, resolution.expected_sha256)
            return path, resolution.signature
        return None

    def _materialize_registry(self, resolution):
        if resolution.mode not in {"named_recipe", "named_artifact"}:
            raise RegistryModelNotFound(self.model_name)
        root = self._cache_root()
        paths = []
        for artifact in resolution.artifacts:
            path = root / artifact.path
            paths.append(self._download_checkpoint(artifact.urls, path, artifact.sha256))
        return tuple(paths)

    @staticmethod
    def _build_architecture(architecture):
        if architecture is None or architecture.name != "HTDemucs":
            name = None if architecture is None else architecture.name
            raise ValueError(f"unsupported checkpoint architecture: {name}")
        from .htdemucs import HTDemucs

        kwargs = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in architecture.kwargs.items()
        }
        return HTDemucs(**kwargs)

    def _load_recipe_component(self, path, recipe):
        if recipe.format == "demucs_package":
            from .states import load_model

            return load_model(path)
        if recipe.format != "state_dict":
            raise ValueError(f"unsupported checkpoint format: {recipe.format}")
        import torch

        model = self._build_architecture(recipe.architecture)
        state = torch.load(path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"strict checkpoint load mismatch: {incompatible}")
        return model

    def load_registered_model(self, resolution=None):
        """Load a schema-v2 artifact or assemble its named recipe directly."""
        resolution = resolution or self.resolve()
        paths = self._materialize_registry(resolution)
        if resolution.mode == "named_artifact":
            from .states import load_model

            return load_model(paths[0]).eval()
        recipe = resolution.recipe
        models = [self._load_recipe_component(path, recipe) for path in paths]
        if recipe.format == "state_dict":
            return models[0].eval()
        from .apply import BagOfModels

        model = BagOfModels(
            models,
            [list(row) for row in recipe.weights] if recipe.weights is not None else None,
            recipe.segment,
        )
        return model.eval()

    def load_separator(self, separator_options):
        """Construct the existing Separator around the resolved model."""
        from .api import Separator

        resolution = self.resolve()
        if resolution.mode in {"checkpoint_path", "checkpoint_url"}:
            path, signature = self.materialize_override(resolution)
            from .states import load_model

            model = load_model(path)
            repo = path.parent
        elif resolution.mode in {"named_recipe", "named_artifact"}:
            model = self.load_registered_model(resolution)
            signature = resolution.signature
            repo = self._cache_root(create=False)
        else:
            return Separator(model=self.model_name, **separator_options)

        separator = Separator.__new__(Separator)
        separator._name = signature
        separator._repo = repo
        separator._model = model
        separator._audio_channels = model.audio_channels
        separator._samplerate = model.samplerate
        import inspect

        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(Separator).parameters.items()
            if name not in {"model", "repo"} and parameter.default is not inspect.Parameter.empty
        }
        defaults.update(separator_options)
        separator.update_parameter(**defaults)
        return separator

    def cache_info(self, *, loaded, status):
        """Report the same read-only resolution used by loading."""
        resolution = self.resolve()
        path = resolution.paths[0] if resolution.paths else None
        return {
            "model": self.model_name,
            "checkpoint_path": str(path) if path is not None else None,
            "checkpoint_paths": [str(item) for item in resolution.paths],
            "checkpoint_url": resolution.checkpoint_url,
            "sha256": resolution.reported_sha256,
            "cached": bool(resolution.paths) and all(item.is_file() for item in resolution.paths),
            "loaded": loaded,
            "status": status,
        }
