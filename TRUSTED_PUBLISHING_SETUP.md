# Trusted Publishing Setup Guide

This guide will help you set up **Trusted Publishing** for demucs-infer on PyPI. With Trusted Publishing, you don't need to manage API tokens - GitHub authenticates directly with PyPI!

---

## 🎯 What is Trusted Publishing?

Trusted Publishing uses OpenID Connect (OIDC) to let GitHub Actions authenticate with PyPI securely. Benefits:

- ✅ **No tokens to manage** - No need to create, store, or rotate API tokens
- ✅ **More secure** - Uses GitHub's identity verification
- ✅ **Never expires** - No need to update expired tokens
- ✅ **Recommended by PyPI** - Modern best practice

---

## 📋 Prerequisites

1. **PyPI Account**: You need an account on https://pypi.org
2. **GitHub Repository**: Already set up at https://github.com/openmirlab/demucs-infer
3. **Workflow File**: Already configured in `.github/workflows/publish.yml`

---

## 🚀 Setup Steps (5 minutes)

### Step 1: Configure PyPI Trusted Publisher

1. **Log in to PyPI**:
   - Go to https://pypi.org and log in

2. **Navigate to Trusted Publishers**:
   - Click your username (top right) → "Account settings"
   - Scroll down to "Publishing" section
   - Click "Add a new pending publisher"

3. **Fill in the form**:
   ```
   PyPI Project Name:    demucs-infer
   Owner:                openmirlab
   Repository name:      demucs-infer
   Workflow name:        publish.yml
   Environment name:     (leave blank)
   ```

4. **Click "Add"**

   **Important Notes:**
   - The project name must be **exactly** `demucs-infer` (as in your pyproject.toml)
   - The workflow name must be **exactly** `publish.yml` (the filename in .github/workflows/)
   - Owner and repository must match your GitHub repo exactly
   - Environment name can be left blank for now

### Step 2: Push Your Code to GitHub

```bash
# Add all changes
git add .

# Commit
git commit -m "Setup Trusted Publishing for PyPI"

# Push to GitHub
git push -u origin main
```

### Step 3: Create a Release Tag

When you're ready to publish:

```bash
# Create and push a version tag
git tag v4.1.0
git push origin v4.1.0
```

**That's it!** GitHub Actions will automatically:
1. Build your package
2. Authenticate with PyPI using Trusted Publishing
3. Upload your package to PyPI

---

## 🔍 Verification

### After pushing the tag:

1. **Watch the workflow run**:
   - Go to: https://github.com/openmirlab/demucs-infer/actions
   - You should see "Publish to PyPI" workflow running
   - It should complete successfully in ~2-3 minutes

2. **Check PyPI**:
   - Go to: https://pypi.org/project/demucs-infer/
   - Your package should appear!

3. **Test installation**:
   ```bash
   pip install demucs-infer
   demucs-infer --help
   ```

---

## 🎨 Visual Guide for PyPI Setup

### Finding the Trusted Publishers Section

```
PyPI Homepage
  └─ Click your username (top right)
      └─ Account settings
          └─ Scroll to "Publishing" section
              └─ Click "Add a new pending publisher"
```

### Form Fields Screenshot Reference

The form looks like this:

```
┌─────────────────────────────────────────────┐
│ Add a new pending publisher                 │
├─────────────────────────────────────────────┤
│ PyPI Project Name:    demucs-infer          │
│ Owner:                openmirlab            │
│ Repository name:      demucs-infer          │
│ Workflow name:        publish.yml           │
│ Environment name:     [optional - blank]    │
│                                             │
│            [Add] [Cancel]                   │
└─────────────────────────────────────────────┘
```

---

## 🆘 Troubleshooting

### Issue: "Publisher verification failed"

**Cause**: The information in PyPI doesn't match your GitHub setup exactly.

**Solution**: Double-check that:
- Repository owner is exactly: `openmirlab`
- Repository name is exactly: `demucs-infer`
- Workflow filename is exactly: `publish.yml`
- PyPI project name is exactly: `demucs-infer`

### Issue: "Project name already exists"

**Cause**: Someone else has already registered this package name on PyPI.

**Solution**: Choose a different package name:
1. Update `name` in `pyproject.toml`
2. Update the PyPI project name in Trusted Publisher settings

### Issue: Workflow runs but fails at publish step

**Cause**: Trusted Publisher not configured correctly on PyPI.

**Solution**:
1. Go to PyPI → Account Settings → Publishing
2. Verify your pending publisher is there
3. Check all fields match exactly

### Issue: "id-token: write permission denied"

**Cause**: GitHub Actions permissions issue.

**Solution**: The workflow is already configured correctly with:
```yaml
permissions:
  contents: read
  id-token: write
```
This should work by default. If it doesn't, check your repository settings.

---

## 🔐 Security Notes

### Why is Trusted Publishing more secure?

1. **No long-lived secrets**: API tokens can be stolen or leaked. Trusted Publishing uses short-lived tokens that GitHub creates on-demand.

2. **Scoped authentication**: Only your specific GitHub workflow can publish, not anyone with a token.

3. **Audit trail**: All publishes are tied to specific GitHub Actions runs with full logs.

4. **No credential management**: You don't need to store, rotate, or secure API tokens.

### What permissions does it use?

The workflow only needs:
- `contents: read` - To check out your code
- `id-token: write` - To get the OIDC token for PyPI authentication

These are the minimal permissions needed and are scoped to the workflow run only.

---

## 📅 Publishing Workflow

### For your first release (v4.1.0):

```bash
# 1. Make sure everything is committed
git status

# 2. Push to GitHub
git push origin main

# 3. Create and push the tag
git tag v4.1.0
git push origin v4.1.0

# 4. Watch it publish automatically!
# Go to: https://github.com/openmirlab/demucs-infer/actions
```

### For future releases:

```bash
# 1. Update version in pyproject.toml
# Change: version = "4.1.1"

# 2. Commit and push
git add pyproject.toml
git commit -m "Bump version to 4.1.1"
git push

# 3. Tag and push
git tag v4.1.1
git push origin v4.1.1

# Done! GitHub Actions handles the rest.
```

---

## ✅ Quick Checklist

Before your first publish, make sure:

- [ ] You have a PyPI account
- [ ] You're logged into PyPI
- [ ] You've added the pending publisher on PyPI with these exact values:
  - [ ] Project name: `demucs-infer`
  - [ ] Owner: `openmirlab`
  - [ ] Repository: `demucs-infer`
  - [ ] Workflow: `publish.yml`
- [ ] Your code is pushed to GitHub
- [ ] You've created and pushed a tag (e.g., `v4.1.0`)

---

## 🎉 Success!

Once set up, publishing is as simple as:

```bash
git tag v4.1.0
git push origin v4.1.0
```

GitHub Actions + Trusted Publishing handles everything else automatically!

---

## 📚 Additional Resources

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPA Publish Action](https://github.com/pypa/gh-action-pypi-publish)

---

## 💡 Pro Tips

1. **Test in TestPyPI first** (optional):
   - Create a separate workflow for TestPyPI
   - Use `https://test.pypi.org` to practice
   - Verify everything works before publishing to real PyPI

2. **Use GitHub Releases**:
   - After pushing a tag, create a GitHub Release
   - Add release notes from your CHANGELOG.md
   - Makes it easy for users to see what's new

3. **Automate version bumping**:
   - Use tools like `bump2version` or `poetry version`
   - Keep version in sync between git tags and pyproject.toml

---

**Need help?** Open an issue at https://github.com/openmirlab/demucs-infer/issues
