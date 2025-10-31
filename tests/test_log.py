#!/usr/bin/env python3
"""Test the custom log module (dora.log replacement)."""
import pytest
from demucs_infer import log


def test_bold_formatting():
    """Test that bold() returns ANSI-formatted string."""
    result = log.bold("test")
    assert result == "\033[1mtest\033[0m"
    assert result.startswith("\033[1m")
    assert result.endswith("\033[0m")


def test_bold_empty_string():
    """Test bold() with empty string."""
    result = log.bold("")
    assert result == "\033[1m\033[0m"


def test_fatal_exits(capsys):
    """Test that fatal() prints to stderr and exits."""
    with pytest.raises(SystemExit) as exc_info:
        log.fatal("test error")

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error: test error" in captured.err
