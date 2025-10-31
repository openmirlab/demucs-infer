# Development Principles

## Package Philosophy

**demucs-infer** is a maintenance fork of Demucs focused on inference-only capabilities with modern PyTorch support.

### Core Tenets

1. **Inference-Only**: No training code ever. Keep the package lean and focused.
2. **Minimal Dependencies**: Only include what's absolutely necessary for inference.
3. **100% Model Fidelity**: Never modify separation algorithms or model architectures.
4. **PyTorch 2.x Compatibility**: Always support the latest stable PyTorch versions.
5. **Attribution First**: Always credit Alexandre Défossez and Meta AI Research prominently.

## Package Management

### Use UV for Environment Management

This project uses [uv](https://github.com/astral-sh/uv) as the package and virtual environment manager.

**Key Rules:**

1. **Virtual Environment Management**: Always use `uv` to manage the virtual environment
   ```bash
   # Install dependencies
   uv sync

   # Add new dependencies
   uv add <package-name>
   ```

2. **Running Python Scripts**: Always use `uv run` instead of `python` directly
   ```bash
   # ✅ Correct
   uv run python examples/basic_separation.py
   uv run python tests/test_imports.py

   # ❌ Incorrect
   python examples/basic_separation.py
   ```

## Maintenance Rules

### 1. Never Modify Core Algorithms

The separation quality and model behavior must remain **100% identical** to original Demucs.

**What you CAN change:**
- ✅ Dependency compatibility layers
- ✅ Import statements and module organization
- ✅ Logging and error messages
- ✅ CLI argument parsing
- ✅ Documentation

**What you CANNOT change:**
- ❌ Model architectures (`demucs.py`, `hdemucs.py`, `htdemucs.py`, `transformer.py`)
- ❌ Separation algorithms (`apply.py`, `spec.py`)
- ❌ Audio processing logic (`audio.py` - except lazy imports)
- ❌ Model loading (`states.py`, `pretrained.py` - except dependency replacements)

### 2. Keep Dependencies Minimal

**Current core dependencies (7 packages):**
- `torch>=2.0.0`
- `torchaudio>=2.0.0`
- `einops`
- `julius>=0.2.3`
- `openunmix`
- `pyyaml`
- `tqdm`

**Before adding ANY new dependency:**
1. Ask: Is this absolutely required for inference?
2. Check: Can we make it optional or lazy-imported?
3. Verify: Does it work with PyTorch 2.x?
4. Document: Add to CHANGELOG.md with clear justification

### 3. Test Before Every Commit

**Minimum testing checklist:**
```bash
# 1. Run full test suite (unit + integration tests)
uv run pytest tests/ -v

# 2. Integration test (verify actual inference works)
uv run python examples/basic_separation.py

# 3. CLI test (verify command-line interface)
uv run demucs-infer --help
uv run demucs-infer --list-models
```

**Note**: The test suite includes:
- Unit tests for logging, API, models
- Integration tests (marked as `@pytest.mark.slow` - skipped by default)
- Import validation tests

### 4. Update CHANGELOG.md

For **every user-facing change**, update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format:

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes

### 5. Semantic Versioning

Follow [Semantic Versioning](https://semver.org/):

- **Patch (4.1.x)**: Bug fixes, dependency updates, documentation
- **Minor (4.x.0)**: New features, new model support, new PyTorch version support
- **Major (x.0.0)**: Breaking API changes (avoid if possible!)

**Version prefix (4.x.x)**: Indicates compatibility with Demucs v4 model architectures.

## Development Workflow

### Setting Up Development Environment

```bash
# Clone and enter directory
cd demucs-infer

# Install dependencies with UV
uv sync

# Run sanity tests
uv run python tests/test_imports.py
```

### Adding a New Feature

1. **Check if it's inference-related**
   - If it involves training: ❌ Don't add it
   - If it's pure inference: ✅ Proceed

2. **Create a branch** (if using git)
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes** following the principles above

4. **Test thoroughly**
   ```bash
   uv run python tests/test_imports.py
   uv run python examples/basic_separation.py
   ```

5. **Update documentation**
   - Update `CHANGELOG.md`
   - Update `README.md` if user-facing
   - Add example if needed

6. **Commit and test build**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   uv build  # Verify package builds
   ```

### Updating Dependencies

**For PyTorch updates:**
```bash
# Check latest stable PyTorch
uv add "torch>=2.x.0" "torchaudio>=2.x.0"

# Test thoroughly - PyTorch updates can break things
uv run python tests/test_imports.py
uv run python examples/basic_separation.py
```

**For other dependencies:**
```bash
# Update single package
uv add "package-name>=x.x.x"

# Update all (careful!)
uv lock --upgrade
```

### Testing Before Release

**Pre-release checklist:**

1. ✅ All import tests pass
2. ✅ Integration examples work
3. ✅ CLI command works
4. ✅ CHANGELOG.md updated
5. ✅ README.md reflects changes (if needed)
6. ✅ Version bumped in `pyproject.toml`
7. ✅ Package builds: `uv build`
8. ✅ Clean install works: `uv pip install dist/*.whl`

## File Organization

```
demucs-infer/
├── demucs_infer/              # Source code (17 files)
│   ├── __init__.py
│   ├── api.py                 # High-level API
│   ├── separate.py            # CLI entry point
│   ├── pretrained.py          # Model loading
│   ├── apply.py               # Inference engine
│   ├── audio.py               # Audio I/O
│   ├── log.py                 # Logging (dora.log replacement)
│   ├── compat.py              # PyTorch compatibility
│   ├── demucs.py              # Base model
│   ├── hdemucs.py             # Hybrid models
│   ├── htdemucs.py            # Transformer models
│   ├── wdemucs.py             # Alias
│   ├── transformer.py         # Transformer components
│   ├── spec.py                # Spectogram operations
│   ├── states.py              # Model state handling
│   ├── repo.py                # Model repositories
│   ├── utils.py               # Utilities
│   └── remote/                # Pretrained configs (DO NOT MODIFY)
├── docs/
│   ├── README.md              # User documentation
│   ├── MIGRATION.md           # Migration from original Demucs
│   └── dev/
│       ├── PRINCIPLES.md      # This file
│       ├── IMPLEMENTATION_NOTES.md  # Technical details
│       └── MAINTENANCE.md     # Maintenance checklist
├── examples/                  # Working code examples
│   ├── basic_separation.py
│   └── batch_processing.py
├── tests/                     # Test suite
│   └── test_imports.py
├── pyproject.toml             # Package metadata
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT license
├── README.md                  # User-facing docs
├── uv.lock                    # UV lockfile
└── .gitignore                 # Git ignore rules
```

## Common Maintenance Tasks

### Updating for New PyTorch Version

1. Update `pyproject.toml`:
   ```toml
   dependencies = [
       "torch>=2.x.0",
       "torchaudio>=2.x.0",
       # ... others
   ]
   ```

2. Test compatibility:
   ```bash
   uv sync
   uv run python tests/test_imports.py
   uv run python examples/basic_separation.py
   ```

3. Update README.md badges if needed

4. Update CHANGELOG.md:
   ```markdown
   ## [4.x.x] - YYYY-MM-DD
   ### Changed
   - Updated PyTorch support to 2.x.0+
   ```

### Fixing a Bug

1. **Reproduce the bug** locally
2. **Identify the cause** - check if it's:
   - Dependency incompatibility → Update compat.py
   - Import error → Check lazy imports
   - Logic error → Fix (but verify against original Demucs!)
3. **Test the fix** thoroughly
4. **Document in CHANGELOG.md**:
   ```markdown
   ### Fixed
   - Fixed [specific issue] with [specific scenario]
   ```

### Adding Support for New Model

If original Demucs adds a new model:

1. **Add config** to `demucs_infer/remote/` (copy from original)
2. **Test loading**: 
   ```python
   from demucs_infer.pretrained import get_model
   model = get_model("new_model_name")
   ```
3. **Update README.md** model table
4. **Update CHANGELOG.md**:
   ```markdown
   ### Added
   - Added support for `new_model_name` model
   ```

## When to Say No

### ❌ Don't Accept:

1. **Training features** - "Can you add model fine-tuning?"
   - Response: demucs-infer is inference-only. Use original Demucs for training.

2. **Algorithm modifications** - "Can you improve separation quality?"
   - Response: We maintain 100% fidelity to original. Submit to Meta's Demucs instead.

3. **Heavy dependencies** - "Can you add feature X that requires Y?"
   - Response: If Y is large/complex, evaluate if truly necessary for inference.

4. **Platform-specific hacks** - "Can you add workaround for platform Z?"
   - Response: Only if it doesn't affect other platforms or add dependencies.

## Quick Reference

```bash
# Setup project
uv sync

# Run tests
uv run python tests/test_imports.py

# Run example
uv run python examples/basic_separation.py

# Add dependency (think twice!)
uv add <package-name>

# Update dependencies (carefully!)
uv lock --upgrade

# Build package
uv build

# Test installation
uv pip install dist/*.whl
```

## Resources

- **Original Demucs**: https://github.com/facebookresearch/demucs
- **UV Documentation**: https://github.com/astral-sh/uv
- **Keep a Changelog**: https://keepachangelog.com/
- **Semantic Versioning**: https://semver.org/

## Philosophy Summary

> **demucs-infer exists to maintain access to excellent research, not to improve upon it.**

We are **custodians**, not innovators. Our job is to:
- ✅ Keep it working with modern tools
- ✅ Make it easy to install and use
- ✅ Document it clearly
- ❌ NOT modify the science
- ❌ NOT add training features
- ❌ NOT change the output quality

Credit always goes to Alexandre Défossez and Meta AI Research. We just keep the lights on.

