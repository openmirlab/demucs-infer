#!/usr/bin/env python3
"""
Basic audio separation example for demucs-infer.

This script demonstrates the simplest way to separate audio sources
using the demucs-infer package.

Usage:
    uv run python examples/basic_separation.py
"""

from pathlib import Path
import torch
import torchaudio
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio


def separate_audio(
    input_file: str,
    output_dir: str = "output",
    model_name: str = "htdemucs_ft",
    device: str = "auto"
):
    """
    Separate audio into stems (drums, bass, other, vocals).
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save separated stems
        model_name: Model to use (default: htdemucs_ft - best quality)
        device: 'cuda', 'cpu', or 'auto' for automatic detection
    """
    print(f"Loading model: {model_name}")
    
    # Auto-detect device if requested
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(model_name)
    model = model.to(device)
    model.eval()
    
    # Load audio
    print(f"Loading audio: {input_file}")
    wav, sr = torchaudio.load(input_file)
    
    # Add batch dimension
    wav = wav.unsqueeze(0).to(device)
    
    # Separate sources
    print("Separating audio sources...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save separated stems
    print(f"\nSaving stems to: {output_dir}/")
    for i, source_name in enumerate(model.sources):
        source = sources[0, i].cpu()  # Remove batch dim and move to CPU
        output_file = output_path / f"{source_name}.wav"
        save_audio(source, str(output_file), sr)
        print(f"  ✓ {source_name}.wav")
    
    print("\n✅ Separation complete!")
    return output_path


def main():
    """Example usage with a test file."""
    # Check if we have a test audio file
    test_files = [
        "test.wav",
        "audio.wav",
        "song.wav",
    ]
    
    input_file = None
    for test_file in test_files:
        if Path(test_file).exists():
            input_file = test_file
            break
    
    if input_file:
        print(f"Found test audio: {input_file}\n")
        separate_audio(
            input_file=input_file,
            output_dir="output",
            model_name="htdemucs_ft",  # Best quality model
            device="auto"
        )
    else:
        print("No test audio file found.")
        print("\nTo use this example:")
        print("1. Place an audio file (test.wav, audio.wav, or song.wav) in the current directory")
        print("2. Run: uv run python examples/basic_separation.py")
        print("\nOr call the function directly:")
        print("  from examples.basic_separation import separate_audio")
        print("  separate_audio('your_audio.wav')")


if __name__ == "__main__":
    main()

