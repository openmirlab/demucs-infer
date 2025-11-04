# Publishing demucs-infer to PyPI

This guide explains how to publish your package to PyPI using the modern uv workflow.

## What's Been Set Up

1. **pyproject.toml** - Updated to use hatchling (modern, fast build backend)
2. **GitHub Remote** - Connected to `https://github.com/openmirlab/demucs-infer.git`
3. **GitHub Actions** - Automated publishing workflow in `.github/workflows/publish.yml`
4. **Build Artifacts** - Package successfully built in `dist/` folder

## Publishing Options

### Option A: Automated Publishing via GitHub Actions with Trusted Publishing (Recommended)

**This is the modern, secure way - no API tokens needed!**

1. **Set up Trusted Publishing on PyPI** (one-time setup):
   - Go to https://pypi.org → Account settings → Publishing
   - Click "Add a new pending publisher"
   - Fill in:
     - PyPI Project Name: `demucs-infer`
     - Owner: `openmirlab`
     - Repository name: `demucs-infer`
     - Workflow name: `publish.yml`
     - Environment name: (leave blank)
   - Click "Add"

   📖 **See TRUSTED_PUBLISHING_SETUP.md for detailed step-by-step guide with screenshots**

2. **Commit and push your code**:
   ```bash
   git add .
   git commit -m "Initial commit: Setup for PyPI publishing"
   git push -u origin main
   ```

3. **Create and push a version tag**:
   ```bash
   git tag v4.1.0
   git push origin v4.1.0
   ```

   GitHub Actions will automatically:
   - Build your package
   - Publish to PyPI
   - Make it available at `pip install demucs-infer`

### Option B: Manual Publishing (Alternative)

If you prefer to publish manually:

```bash
# Get PyPI token from https://pypi.org/manage/account/token/
export UV_PUBLISH_TOKEN=pypi-xxxxxxxxxxxxxxxx

# Build and publish
uv build
uv publish
```

### Option C: Install Directly from GitHub (No PyPI needed)

Users can install directly from your GitHub repository:
```bash
pip install git+https://github.com/openmirlab/demucs-infer.git
```

## Step-by-Step First Time Publishing

1. **First, push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit: Setup for PyPI publishing"
   git push -u origin main
   ```

2. **Set up PyPI token in GitHub** (see Option B step 1 above)

3. **Create and push a release tag**:
   ```bash
   git tag v4.1.0
   git push origin v4.1.0
   ```

4. **Watch the Action run**:
   - Go to `https://github.com/openmirlab/demucs-infer/actions`
   - You'll see the "Publish to PyPI" workflow running
   - Once complete, your package will be live on PyPI!

5. **Test installation**:
   ```bash
   pip install demucs-infer
   ```

## Future Releases

For future versions:

1. Update version in `pyproject.toml`:
   ```toml
   version = "4.1.1"  # or whatever new version
   ```

2. Commit changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 4.1.1"
   git push
   ```

3. Tag and push:
   ```bash
   git tag v4.1.1
   git push origin v4.1.1
   ```

GitHub Actions will automatically build and publish!

## Package Structure

Your package is built with:
- **Name**: demucs-infer
- **Version**: 4.1.0
- **Command**: `demucs-infer` (available after installation)
- **Module**: `demucs_infer` (importable in Python)

## Current Build Output

```
dist/
├── demucs_infer-4.1.0-py3-none-any.whl (64K)
└── demucs_infer-4.1.0.tar.gz (48K)
```

Both files will be uploaded to PyPI when you publish.

## Testing Locally

Before publishing, test your package locally:

```bash
# Install in editable mode
uv pip install -e .

# Test the command
demucs-infer --help

# Test importing
python -c "import demucs_infer; print(demucs_infer.__version__)"
```

## Troubleshooting

- **Authentication error**: Make sure your PyPI token is correctly set
- **Package name taken**: You may need to choose a different name in pyproject.toml
- **Build fails**: Run `uv build` locally first to catch errors
- **GitHub Action fails**: Check the workflow logs at github.com/openmirlab/demucs-infer/actions
