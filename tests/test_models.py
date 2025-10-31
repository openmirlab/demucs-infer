#!/usr/bin/env python3
"""Test model loading and pretrained model functionality."""
import pytest
from demucs_infer import pretrained


def test_default_model():
    """Test that DEFAULT_MODEL is set correctly."""
    assert pretrained.DEFAULT_MODEL == "htdemucs"


def test_get_model_from_args():
    """Test get_model_from_args with basic arguments."""
    # Create a simple namespace object to simulate argparse args
    class Args:
        def __init__(self):
            self.name = "htdemucs"
            self.repo = None
            self.sig = None

    args = Args()
    # This should not raise an error
    # Note: We don't actually load the model as it requires downloading
    # Just verify the function is callable
    assert callable(pretrained.get_model_from_args)


def test_model_loading_error():
    """Test that ModelLoadingError exists and is importable."""
    assert hasattr(pretrained, 'ModelLoadingError')
    assert issubclass(pretrained.ModelLoadingError, Exception)
