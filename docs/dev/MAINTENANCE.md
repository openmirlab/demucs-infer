# Maintenance Checklist

Quick reference guide for maintaining demucs-infer.

## Pre-Commit Checklist

Before committing any changes:

- [ ] Code changes follow [PRINCIPLES.md](PRINCIPLES.md)
- [ ] No modifications to core algorithms (models, separation logic)
- [ ] Full test suite passes: `uv run pytest tests/ -v`
- [ ] Integration example works: `uv run python examples/basic_separation.py`
- [ ] CLI commands work: `uv run demucs-infer --help` and `--list-models`
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Documentation updated (if needed)

## Pre-Release Checklist

Before releasing a new version:

### 1. Version & Documentation
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md has entry for this version with date
- [ ] README.md reflects any new features or changes
- [ ] All docs are accurate

### 2. Testing
- [ ] Full test suite passes: `uv run pytest tests/ -v`
- [ ] Integration examples work:
  - [ ] `uv run python examples/basic_separation.py`
- [ ] CLI commands work:
  - [ ] `uv run demucs-infer --help`
  - [ ] `uv run demucs-infer --list-models`
- [ ] Clean environment test:
  ```bash
  uv venv --python 3.10 test-env
  source test-env/bin/activate
  uv pip install .
  demucs-infer --help
  deactivate
  rm -rf test-env
  ```

### 3. Build & Distribution
- [ ] Package builds: `uv build`
- [ ] No errors in build output
- [ ] `.whl` and `.tar.gz` created in `dist/`
- [ ] Test wheel installation:
  ```bash
  uv pip install dist/demucs_infer-*.whl
  ```

### 4. Git
- [ ] All changes committed
- [ ] Tagged with version: `git tag v4.x.x`
- [ ] Pushed: `git push && git push --tags`

## Monthly Maintenance

Run these checks monthly to catch issues early:

### Dependency Health Check
```bash
# Check for outdated dependencies
uv pip list --outdated

# Check for security issues (if using safety or similar)
# pip install safety
# safety check
```

### Compatibility Check
```bash
# Test with latest PyTorch
uv add "torch>=2.0.0" "torchaudio>=2.0.0" --upgrade
uv sync
uv run pytest tests/ -v
uv run python examples/basic_separation.py
```

### CI/CD Check
- [ ] GitHub Actions workflow passing
- [ ] No failing tests in CI
- [ ] Review any dependabot alerts

### Documentation Review
- [ ] README.md still accurate
- [ ] Links not broken
- [ ] Examples still work
- [ ] Installation instructions current

## Quarterly Review

Every 3 months, do a deeper review:

### Code Quality
- [ ] Remove any accumulated dead code
- [ ] Check for deprecated PyTorch APIs
- [ ] Review TODO comments in code
- [ ] Update type hints if needed

### Dependencies Audit
- [ ] Review all dependencies - still needed?
- [ ] Check for lighter alternatives
- [ ] Verify licenses compatible (MIT)
- [ ] Update version constraints if needed

### Community Check
- [ ] Check original Demucs repo for updates
- [ ] Review any issues/PRs in demucs-infer repo
- [ ] Update from upstream if beneficial (carefully!)

## Emergency Hotfix Process

For critical bugs requiring immediate fix:

1. **Create hotfix branch**:
   ```bash
   git checkout -b hotfix/issue-description
   ```

2. **Make minimal fix** (only what's necessary)

3. **Test thoroughly**:
   ```bash
   uv run python tests/test_imports.py
   uv run python examples/basic_separation.py
   ```

4. **Bump patch version** in `pyproject.toml`

5. **Update CHANGELOG.md**:
   ```markdown
   ## [4.x.x] - YYYY-MM-DD
   ### Fixed
   - Critical: [description of bug fix]
   ```

6. **Release immediately**:
   ```bash
   uv build
   # Test wheel
   uv pip install dist/demucs_infer-*.whl
   # If good, tag and push
   git tag v4.x.x
   git push origin hotfix/issue-description
   git push --tags
   ```

## Common Maintenance Scenarios

### Scenario 1: PyTorch Breaking Change

**Symptom**: Import errors or runtime errors after PyTorch update

**Fix**:
1. Check original Demucs issues for similar problems
2. Add compatibility layer in `compat.py`
3. Test thoroughly with old and new PyTorch
4. Document in CHANGELOG.md as "Fixed"

### Scenario 2: Dependency Becomes Unmaintained

**Symptom**: Dependency no longer available or incompatible

**Fix**:
1. Check if truly needed for inference
2. Options (in order of preference):
   - Remove if not critical
   - Make optional/lazy import
   - Vendor minimal required code
   - Fork and maintain separately (last resort)
3. Document in CHANGELOG.md

### Scenario 3: Model Download URL Changes

**Symptom**: Models fail to download

**Fix**:
1. Update URLs in `pretrained.py` or config YAMLs
2. Test model loading:
   ```python
   from demucs_infer.pretrained import get_model
   model = get_model("htdemucs_ft")
   ```
3. Update CHANGELOG.md as "Fixed"

### Scenario 4: User Requests Training Feature

**Response Template**:
> demucs-infer is an inference-only package to maintain PyTorch 2.x compatibility. 
> For training features, please use the original Demucs repository or fork this package
> for your specific needs. We maintain a strict inference-only policy to keep the 
> package minimal and maintainable.

## Tools & Commands Reference

```bash
# Development
uv sync                              # Install dependencies
uv add <package>                     # Add dependency
uv remove <package>                  # Remove dependency
uv run python <script>               # Run script in venv

# Testing
uv run python tests/test_imports.py  # Import test
uv run python examples/*.py          # Integration tests
uv run demucs-infer --help          # CLI test

# Building
uv build                             # Build package
uv pip install dist/*.whl            # Test wheel

# Cleanup
rm -rf build/ dist/ *.egg-info/      # Clean build artifacts
find . -name "__pycache__" -exec rm -rf {} +  # Clean cache
```

## Contact & Escalation

If you encounter issues beyond routine maintenance:

1. **Check original Demucs**: https://github.com/facebookresearch/demucs/issues
2. **Review past fixes**: Check git history and CHANGELOG.md
3. **Document thoroughly**: Add to docs/dev/ for future reference

## Success Metrics

Good maintenance means:

- ✅ Package installs cleanly with `uv add demucs-infer`
- ✅ Works with latest stable PyTorch
- ✅ Import tests pass
- ✅ Examples run without errors
- ✅ Dependencies stay minimal (7 core packages)
- ✅ Zero algorithm changes
- ✅ Clear documentation

If all these are true, you're doing great! 🎉

