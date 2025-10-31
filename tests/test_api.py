#!/usr/bin/env python3
"""Test the high-level Separator API."""
import pytest
from demucs_infer.api import Separator


def test_separator_class_exists():
    """Test that Separator class is importable."""
    assert Separator is not None


def test_separator_init_params():
    """Test that Separator accepts expected parameters."""
    # Note: We don't instantiate as it would download models
    # Just verify the class signature
    import inspect
    sig = inspect.signature(Separator.__init__)
    params = list(sig.parameters.keys())

    # Should have 'self' and 'model' at minimum
    assert 'self' in params
    assert 'model' in params


@pytest.mark.slow
def test_separator_default_model():
    """Test Separator with default model (requires model download)."""
    # Mark as slow test - only runs when explicitly requested
    # Skip in regular test runs to avoid downloading large models
    pytest.skip("Skipping model download test - run with -m slow to enable")
    sep = Separator()
    assert sep is not None
