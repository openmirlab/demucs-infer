"""Check every schema-v2 artifact's primary HTTPS source for liveness.

The package registry is the only URL authority. Each artifact's first ordered
URL is probed with HEAD, then a one-byte ranged GET when the host rejects or
mishandles HEAD; only a successful HTTP response counts as live.

Deselected by default (see pyproject.toml's `addopts = -m "not network"` and
the registered `network` marker). Run explicitly with:

    pytest -m network tests/test_checkpoints_liveness.py

Reads: checkpoint_catalog.checkpoint_config.
"""
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from demucs_infer.checkpoint_catalog import checkpoint_config

pytestmark = pytest.mark.network


def _request_ok(request, timeout):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if request.get_method() == "GET":
                response.read(1)
            return 200 <= response.status < 400, f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return False, f"URL error: {error.reason}"
    except TimeoutError:
        return False, "timeout"


def _probe_primary_url(url, timeout=20.0):
    headers = {"User-Agent": "demucs-infer-checkpoint-liveness/1"}
    head_ok, head_detail = _request_ok(
        urllib.request.Request(url, headers=headers, method="HEAD"),
        timeout,
    )
    if head_ok:
        return True, f"HEAD {head_detail}"
    range_headers = dict(headers)
    range_headers["Range"] = "bytes=0-0"
    get_ok, get_detail = _request_ok(
        urllib.request.Request(url, headers=range_headers, method="GET"),
        timeout,
    )
    return get_ok, f"HEAD {head_detail}; ranged GET {get_detail}"


def _primary_urls():
    registry = checkpoint_config()
    return tuple(
        (artifact.id, artifact.urls[0])
        for artifact in sorted(registry.artifacts.values(), key=lambda item: item.id)
    )


PRIMARY_URLS = _primary_urls()


@pytest.mark.parametrize(
    "artifact_id,url",
    PRIMARY_URLS,
    ids=[f"{urlsplit(url).netloc}-{artifact_id}" for artifact_id, url in PRIMARY_URLS],
)
def test_primary_checkpoint_url_is_live(artifact_id, url):
    live, detail = _probe_primary_url(url)
    host = urlsplit(url).netloc
    assert live, f"{host}: {artifact_id}: {url} is not live ({detail})"
