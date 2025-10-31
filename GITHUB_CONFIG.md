# GitHub Configuration Guide

**Repository**: ChenPaulYu/demucs-infer
**Date**: 2025-10-31

This guide covers all GitHub configurations needed for the package to work properly.

---

## 🎯 Required Configurations

### 1. ✅ GitHub Actions - Already Working

**Current Status**: The test workflow (`.github/workflows/python-package.yml`) should work **out of the box** after you push.

**No configuration needed!** GitHub Actions is enabled by default on public repos.

**Verify after push:**
1. Go to: `https://github.com/ChenPaulYu/demucs-infer/actions`
2. You should see workflows running automatically
3. Check that tests pass

---

### 2. ⚙️ Branch Protection (Optional but Recommended)

**Why**: Prevent accidental force pushes and require CI checks before merging.

**Setup**:
1. Go to: `Settings` → `Branches`
2. Click `Add branch protection rule`
3. Branch name pattern: `main`
4. Enable:
   - ☑️ Require a pull request before merging
   - ☑️ Require status checks to pass before merging
     - Add check: `test` (from python-package.yml)
   - ☑️ Require branches to be up to date before merging
5. Save changes

**Priority**: Optional (but recommended for safety)

---

### 3. 🔐 PyPI Publishing - Required for Automated Releases

The `.github/workflows/python-publish.yml` workflow can automatically publish to PyPI when you create a GitHub Release.

#### Option A: Trusted Publishing (Recommended - Modern, Secure)

**Advantages**:
- ✅ No API tokens to manage
- ✅ More secure (OIDC-based)
- ✅ No expiration

**Setup Steps:**

1. **On PyPI** (https://pypi.org):
   - Log in to your PyPI account
   - Go to: `Account Settings` → `Publishing`
   - Click `Add a new pending publisher`
   - Fill in:
     - **PyPI Project Name**: `demucs-infer`
     - **Owner**: `ChenPaulYu`
     - **Repository name**: `demucs-infer`
     - **Workflow name**: `python-publish.yml`
     - **Environment name**: `pypi`
   - Click `Add`

2. **On GitHub** (https://github.com/ChenPaulYu/demucs-infer):
   - Go to: `Settings` → `Environments`
   - Click `New environment`
   - Name: `pypi`
   - Add protection rules (optional):
     - ☑️ Required reviewers (yourself)
     - ☑️ Wait timer (0 minutes)
   - Save

**That's it!** No secrets needed.

#### Option B: API Token (Traditional Method)

**If trusted publishing doesn't work:**

1. **Generate PyPI API Token**:
   - Go to: https://pypi.org/manage/account/token/
   - Click `Add API token`
   - Token name: `github-actions-demucs-infer`
   - Scope: `Project: demucs-infer` (after first upload) or `Entire account` (for first upload)
   - Copy the token (starts with `pypi-...`)

2. **Add to GitHub Secrets**:
   - Go to: `Settings` → `Secrets and variables` → `Actions`
   - Click `New repository secret`
   - Name: `PYPI_API_TOKEN`
   - Value: (paste your token)
   - Click `Add secret`

3. **Update workflow** (`.github/workflows/python-publish.yml`):
   ```yaml
   # Change line 67-68 from:
   - name: Publish release distributions to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1

   # To:
   - name: Publish release distributions to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

---

### 4. 📝 Repository Settings (Recommended)

**Go to**: `Settings` → `General`

#### About Section
- **Description**: "Inference-only distribution of Demucs for PyTorch 2.x - Music source separation"
- **Website**: `https://github.com/ChenPaulYu/demucs-infer`
- **Topics**: Add tags
  - `audio`
  - `music`
  - `source-separation`
  - `demucs`
  - `pytorch`
  - `inference`
  - `music-processing`

#### Features
- ☑️ Issues (enabled)
- ☑️ Projects (optional)
- ☑️ Discussions (optional - for community support)

#### Pull Requests
- ☑️ Allow squash merging
- ☑️ Allow merge commits
- ☑️ Automatically delete head branches

---

## 📋 Configuration Checklist

### Immediately After Push (No Setup Needed)
- [ ] Verify GitHub Actions runs automatically
- [ ] Check that tests pass in Actions tab
- [ ] Verify CI badge shows status

### For Publishing to PyPI (When Ready)
- [ ] Create PyPI account if you don't have one
- [ ] Reserve package name on PyPI (upload first version manually or via trusted publishing)
- [ ] Set up Trusted Publishing on PyPI ⭐ **RECOMMENDED**
  - [ ] Add pending publisher on PyPI
  - [ ] Create `pypi` environment on GitHub
- [ ] **OR** Set up API token (alternative)
  - [ ] Generate token on PyPI
  - [ ] Add to GitHub secrets

### Optional Improvements
- [ ] Enable branch protection on `main`
- [ ] Configure repository topics and description
- [ ] Enable Dependabot (Settings → Security → Dependabot)
- [ ] Add CODEOWNERS file (optional)

---

## 🚀 How to Publish a Release

Once PyPI publishing is configured:

### 1. Via GitHub UI (Recommended)

```bash
# 1. Make sure everything is committed and pushed
git add .
git commit -m "feat: Add testing and improvements"
git push origin main

# 2. Create and push a tag
git tag v4.1.0
git push origin v4.1.0
```

Then on GitHub:
1. Go to: `Releases` → `Create a new release`
2. Choose tag: `v4.1.0`
3. Release title: `v4.1.0`
4. Description: Copy from CHANGELOG.md
5. Click `Publish release`
6. GitHub Actions will automatically build and publish to PyPI!

### 2. Via Command Line

```bash
# 1. Create tag
git tag -a v4.1.0 -m "Release v4.1.0: Add testing and improvements"
git push origin v4.1.0

# 2. Create release via gh CLI
gh release create v4.1.0 \
  --title "v4.1.0" \
  --notes-file CHANGELOG.md

# GitHub Actions will automatically publish to PyPI
```

---

## 🔍 Verification Steps

### After First Push
1. Check Actions: https://github.com/ChenPaulYu/demucs-infer/actions
   - Should show "Tests" workflow
   - Status should be ✅ passing

2. Check CI Badge in README:
   - Should show current build status

### After Release (if publishing configured)
1. Check PyPI: https://pypi.org/project/demucs-infer/
   - Package should appear
   - Version should match release

2. Test installation:
   ```bash
   pip install demucs-infer
   demucs-infer --help
   ```

---

## 🎯 Minimal Setup (Start Here)

**To get started with just testing (no publishing yet):**

1. ✅ Push your changes
2. ✅ Check GitHub Actions runs
3. ✅ Verify tests pass

**That's it!** Everything else can be configured later when you're ready to publish.

---

## ⚠️ Common Issues

### Issue: GitHub Actions doesn't run
**Solution**: Check that Actions are enabled in `Settings` → `Actions` → `General`

### Issue: Test workflow fails
**Solution**: Check the logs in the Actions tab for specific errors

### Issue: PyPI publish fails with authentication error
**Solution**:
- If using trusted publishing: Verify environment name matches exactly (`pypi`)
- If using token: Verify secret name is correct and token is valid

### Issue: Badge shows "unknown" status
**Solution**: Wait a few minutes for first workflow to complete, then hard refresh

---

## 📞 Support

If you encounter issues:
1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Check the [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/)
3. Open an issue in the repository

---

## 🎉 Summary

**Minimum to work right now**: Nothing! Just push.

**For automated PyPI publishing**: Set up Trusted Publishing (5 minutes)

**Optional but nice**: Branch protection, repository settings

The package is already configured to work - you just need to enable PyPI publishing when you're ready to release! 🚀
