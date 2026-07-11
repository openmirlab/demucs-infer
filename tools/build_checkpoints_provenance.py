#!/usr/bin/env python3
"""ADOPT P3 -- build a checkpoints provenance record.

Every model demucs_infer can load already has *some* integrity check:
  - Meta-hosted models (demucs_infer/remote/files.txt + *.yaml bag configs):
    the URL is ``ROOT_URL + root + filename``, and the filename embeds an
    8-hex-char sha256 prefix (e.g. ``955717e8-8726e21a.th`` -> prefix
    ``8726e21a``). ``RemoteRepo.get_model`` downloads via
    ``torch.hub.load_state_dict_from_url(url, check_hash=True)``, which
    verifies the downloaded bytes' sha256 against that filename prefix.
    ``LocalRepo`` separately verifies the same prefix via
    ``repo.check_checksum`` when loading from a local folder.
  - Community models (demucs_infer/community.py's GDriveRepo): until this
    phase, there was NO integrity check at all on the gdown download --
    a real gap, since ``states.load_model`` calls
    ``torch.load(..., weights_only=False)`` (pickle deserialization) on
    whatever bytes land in the cache.

This tool does not duplicate either existing mechanism; it extends
coverage by recording the FULL 64-char sha256 of every checkpoint this
machine already has cached (no re-download), for two purposes:
  1. A durable, auditable provenance record (docs/checkpoints_provenance.json)
     going beyond the 8-char prefix torch.hub checks.
  2. The source of the sha256 wired into community.py's COMMUNITY_MODELS
     registry (see P3 commit), so GDriveRepo can finally verify its
     downloads via the existing repo.check_checksum() helper.

Only checkpoints already present in the local torch hub cache
(~/.cache/torch/hub/checkpoints/) or the demucs-infer community cache
(~/.cache/demucs-infer/) are hashed; nothing is downloaded by this tool.

Reads: demucs_infer.pretrained (_parse_remote_files, REMOTE_ROOT, ROOT_URL),
       demucs_infer.community (COMMUNITY_MODELS)
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demucs_infer.pretrained import _parse_remote_files, REMOTE_ROOT, ROOT_URL  # noqa: E402
from demucs_infer.community import COMMUNITY_MODELS, DEFAULT_CACHE_DIR  # noqa: E402

TORCH_HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    models = _parse_remote_files(REMOTE_ROOT / "files.txt")
    out = {"official": {}, "community": {}}

    for sig, url in sorted(models.items()):
        filename = url.rsplit("/", 1)[-1]
        embedded_prefix = filename.split("-", 1)[1].rsplit(".", 1)[0] if "-" in filename else None
        cached = TORCH_HUB_CACHE / filename
        entry = {"url": url, "embedded_prefix": embedded_prefix}
        if cached.exists():
            full = sha256_of(cached)
            entry["sha256"] = full
            entry["prefix_matches_embedded"] = (
                embedded_prefix is not None and full.startswith(embedded_prefix)
            )
        else:
            entry["sha256"] = None
            entry["note"] = "not cached locally; not downloaded by this tool"
        out["official"][sig] = entry

    for sig, meta in COMMUNITY_MODELS.items():
        cached = DEFAULT_CACHE_DIR / f"{sig}.th"
        entry = {
            "name": meta["name"],
            "origin": meta["origin"],
            "gdrive_id": meta["gdrive_id"],
        }
        if cached.exists():
            entry["sha256"] = sha256_of(cached)
        else:
            entry["sha256"] = None
            entry["note"] = "not cached locally; not downloaded by this tool"
        out["community"][sig] = entry

    out_path = REPO_ROOT / "docs" / "checkpoints_provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    n_official = len(out["official"])
    n_hashed = sum(1 for e in out["official"].values() if e["sha256"])
    n_mismatch = sum(
        1 for e in out["official"].values()
        if e["sha256"] and not e.get("prefix_matches_embedded", True)
    )
    print(f"wrote {out_path}")
    print(f"official: {n_hashed}/{n_official} hashed locally, {n_mismatch} prefix mismatches")
    print(f"community: {sum(1 for e in out['community'].values() if e['sha256'])}/"
          f"{len(out['community'])} hashed locally")


if __name__ == "__main__":
    main()
