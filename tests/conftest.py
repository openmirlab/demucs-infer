"""Regression guard: the default (offline) test suite must never touch a
real network socket.

An autouse fixture monkeypatches ``socket.socket.connect``/``connect_ex`` for
the duration of every test NOT marked ``network`` or ``realweights`` (both
already deselected by default via pyproject.toml's ``addopts``), raising a
clear error instead of silently reaching the network -- the failure mode a
Phonon-hosting audit found in this repo (an un-mocked checkpoint download
inside a test with no marker at all). Loopback/unix-domain connections stay
allowed since nothing in this suite currently needs them, but blocking only
non-local destinations keeps the door open without weakening the guard.

Reads: (nothing internal -- pytest/stdlib socket only)
"""

from __future__ import annotations

import socket

import pytest

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test not marked network/realweights tries to open a
    real network socket."""


def _is_local(address) -> bool:
    if isinstance(address, str):
        # AF_UNIX: the address is a path (or an abstract-namespace string),
        # never a (host, port) tuple -- always allowed.
        return True
    try:
        host = address[0]
    except (TypeError, IndexError):
        return False
    return host in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def _block_real_network(request, monkeypatch):
    marker = request.node.get_closest_marker("network") or request.node.get_closest_marker(
        "realweights"
    )
    if marker is not None:
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_local(address):
            raise NetworkAccessBlocked(
                f"blocked real network connection to {address!r} from a test not "
                "marked @pytest.mark.network or @pytest.mark.realweights"
            )
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if not _is_local(address):
            raise NetworkAccessBlocked(
                f"blocked real network connection to {address!r} from a test not "
                "marked @pytest.mark.network or @pytest.mark.realweights"
            )
        return real_connect_ex(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
