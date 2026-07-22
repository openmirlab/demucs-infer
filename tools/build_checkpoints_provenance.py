#!/usr/bin/env python3
"""Build and locally verify the schema-v2 checkpoint provenance record.

The package registry owns full hashes and ordered HTTPS sources. This tool
projects those facts into the durable JSON audit record and verifies any
matching local cache file against the configured hash. It never downloads.

Reads: checkpoint_catalog, community compatibility metadata.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demucs_infer.checkpoint_catalog import checkpoint_config  # noqa: E402
from demucs_infer.community import COMMUNITY_MODELS, DEFAULT_CACHE_DIR  # noqa: E402

TORCH_HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    registry = checkpoint_config()
    out = {"official": {}, "community": {}, "third_party": {}}
    cached_verified = {"official": 0, "community": 0, "third_party": 0}

    for artifact in sorted(registry.artifacts.values(), key=lambda item: item.signature):
        if artifact.signature in COMMUNITY_MODELS:
            continue
        if artifact.provenance != "facebookresearch/demucs":
            entry = {
                "path": artifact.path,
                "urls": list(artifact.urls),
                "sha256": artifact.sha256,
                "license": artifact.license,
                "provenance": artifact.provenance,
                "source_revision": artifact.source_revision,
                "updated_at": artifact.updated_at,
            }
            for root in (TORCH_HUB_CACHE, DEFAULT_CACHE_DIR):
                cached = root / artifact.path
                if cached.exists():
                    if sha256_of(cached) != artifact.sha256:
                        raise ValueError(f"cached checkpoint hash mismatch: {cached}")
                    cached_verified["third_party"] += 1
            out["third_party"][artifact.id] = entry
            continue
        embedded_prefix = Path(artifact.path).stem.split("-", 1)[1]
        entry = {
            "url": artifact.urls[0],
            "embedded_prefix": embedded_prefix,
            "sha256": artifact.sha256,
            "prefix_matches_embedded": artifact.sha256.startswith(embedded_prefix),
        }
        for root in (TORCH_HUB_CACHE, DEFAULT_CACHE_DIR):
            cached = root / artifact.path
            if cached.exists():
                if sha256_of(cached) != artifact.sha256:
                    raise ValueError(f"cached checkpoint hash mismatch: {cached}")
                cached_verified["official"] += 1
        out["official"][artifact.signature] = entry

    for sig, meta in COMMUNITY_MODELS.items():
        cached = DEFAULT_CACHE_DIR / f"{sig}.th"
        artifact = registry.signatures[sig]
        entry = {
            "name": meta["name"],
            "origin": meta["origin"],
            "gdrive_id": meta["gdrive_id"],
            "sha256": artifact.sha256,
            "verified_https_url": artifact.urls[0],
        }
        if cached.exists():
            if sha256_of(cached) != artifact.sha256:
                raise ValueError(f"cached checkpoint hash mismatch: {cached}")
            cached_verified["community"] += 1
        out["community"][sig] = entry

    out_path = REPO_ROOT / "docs" / "checkpoints_provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    n_official = len(out["official"])
    n_mismatch = sum(
        1 for e in out["official"].values()
        if e["sha256"] and not e.get("prefix_matches_embedded", True)
    )
    print(f"wrote {out_path}")
    print(
        f"official: {n_official}/{n_official} full hashes recorded; "
        f"{cached_verified['official']} cached files verified; "
        f"{n_mismatch} prefix mismatches"
    )
    n_community = len(out["community"])
    print(
        f"community: {n_community}/{n_community} full hashes recorded; "
        f"{cached_verified['community']} cached files verified"
    )
    n_third_party = len(out["third_party"])
    print(
        f"third-party: {n_third_party}/{n_third_party} full hashes recorded; "
        f"{cached_verified['third_party']} cached files verified"
    )


if __name__ == "__main__":
    main()
