"""ADOPT P3 -- checkpoint URL liveness (network test, deselected by default).

HEADs every official checkpoint URL (built the same way
demucs_infer.pretrained._parse_remote_files does, from
demucs_infer/remote/files.txt + ROOT_URL) plus every community model's
Google Drive download URL, to catch upstream hosting rot (Meta or a
community author moving/deleting a file) before a user hits it.

Deselected by default (see pyproject.toml's `addopts = -m "not network"` and
the registered `network` marker). Run explicitly with:

    pytest -m network tests/test_checkpoints_liveness.py
"""
import pytest
import urllib.request
import urllib.error

from demucs_infer.pretrained import _parse_remote_files, REMOTE_ROOT
from demucs_infer.community import COMMUNITY_MODELS

pytestmark = pytest.mark.network


def _head_ok(url: str, timeout: float = 10.0) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as e:
        # Some hosts (e.g. Drive) don't support HEAD cleanly; treat
        # "method not allowed"/redirect-adjacent codes as reachable.
        return e.code in (405, 302, 303)
    except Exception:
        return False


def _official_urls():
    models = _parse_remote_files(REMOTE_ROOT / "files.txt")
    return sorted(models.items())


@pytest.mark.parametrize("sig,url", _official_urls())
def test_official_checkpoint_url_is_live(sig, url):
    assert _head_ok(url), f"{sig}: {url} did not respond with a live status"


@pytest.mark.parametrize("sig,entry", sorted(COMMUNITY_MODELS.items()))
def test_community_checkpoint_gdrive_is_live(sig, entry):
    url = f"https://drive.google.com/uc?id={entry['gdrive_id']}"
    assert _head_ok(url), f"{sig} ({entry['name']}): {url} did not respond with a live status"
