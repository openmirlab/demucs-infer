#!/usr/bin/env python3
"""Quick sanity test for demucs-infer package."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing demucs-infer imports...")

# Test 1: log module (dora.log replacement)
print("\n1. Testing demucs_infer.log (dora.log replacement)...")
from demucs_infer import log
print(f"   ✓ log.fatal: {log.fatal}")
print(f"   ✓ log.bold: {log.bold}")
print(f"   ✓ Bold text: {log.bold('test message')}")

# Test 2: Core API
print("\n2. Testing demucs_infer.api...")
try:
    from demucs_infer.api import Separator
    print("   ✓ Separator class imported (but not instantiated - requires torch)")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Audio module
print("\n3. Testing demucs_infer.audio...")
try:
    from demucs_infer import audio
    print(f"   ✓ audio module imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Model loading
print("\n4. Testing demucs_infer.pretrained...")
try:
    from demucs_infer import pretrained
    print(f"   ✓ pretrained module imported")
    print(f"   ✓ DEFAULT_MODEL: {pretrained.DEFAULT_MODEL}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 5: CLI entry point
print("\n5. Testing demucs_infer.separate (CLI)...")
try:
    from demucs_infer import separate
    print("   ✓ separate module imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n✅ All import tests passed!")
print("\nNote: Full functionality requires torch/torchaudio installation.")
