# Developer Documentation

This directory contains documentation for **maintaining and developing** `demucs-infer`.

## Documents

### 📋 [PRINCIPLES.md](PRINCIPLES.md)
**Core development philosophy and rules**

Read this FIRST before making any changes. Covers:
- Package philosophy (inference-only, minimal deps, 100% fidelity)
- Package management with UV
- Maintenance rules
- Development workflow
- What to accept/reject

### 🔧 [MAINTENANCE.md](MAINTENANCE.md)
**Practical checklists and procedures**

Your quick reference for:
- Pre-commit checklist
- Pre-release checklist
- Monthly/quarterly maintenance tasks
- Common maintenance scenarios
- Tools & commands reference

### 🏗️ [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
**Technical implementation details**

Historical record of:
- What was created and why
- Key modifications from original Demucs
- Dependency replacement strategies
- Architecture decisions
- Technical debt and known limitations

## Quick Start for Maintainers

### First Time Setup
```bash
# Navigate to project
cd /home/worzpro/Desktop/dev/patched_modules/demucs-infer

# Install dependencies
uv sync

# Verify everything works
uv run python tests/test_imports.py
```

### Before Every Commit
1. Read relevant section in [PRINCIPLES.md](PRINCIPLES.md)
2. Make your changes
3. Follow [MAINTENANCE.md](MAINTENANCE.md) pre-commit checklist
4. Test: `uv run python tests/test_imports.py`
5. Update CHANGELOG.md

### Before Every Release
Follow the complete pre-release checklist in [MAINTENANCE.md](MAINTENANCE.md)

## File Organization

```
docs/
├── README.md                  # User documentation (installation, usage)
├── MIGRATION.md              # User guide for migrating from original Demucs
└── dev/                      # Developer documentation (YOU ARE HERE)
    ├── README.md             # This file
    ├── PRINCIPLES.md         # Core development philosophy ⭐
    ├── MAINTENANCE.md        # Checklists and procedures ⭐
    └── IMPLEMENTATION_NOTES.md  # Technical details
```

## Philosophy

> **We are custodians, not innovators.**

`demucs-infer` exists to:
- ✅ Keep excellent research accessible
- ✅ Maintain PyTorch 2.x compatibility
- ✅ Provide easy installation
- ❌ NOT modify algorithms
- ❌ NOT add training features
- ❌ NOT change output quality

All credit for the science goes to **Alexandre Défossez** and **Meta AI Research**.

## Key Principles

1. **Inference-Only**: Never add training code
2. **Minimal Dependencies**: Currently 7 core packages - keep it that way
3. **100% Model Fidelity**: Never modify separation algorithms
4. **Use UV**: For all package management
5. **Test Everything**: Before every commit and release
6. **Document Changes**: Update CHANGELOG.md for user-facing changes

## Getting Help

1. **Check existing docs** in this directory
2. **Review git history** for similar changes
3. **Check original Demucs** for upstream solutions
4. **Document your findings** to help future maintainers

## Contributing

When making changes:

1. **Understand the philosophy** (read PRINCIPLES.md)
2. **Follow the checklist** (use MAINTENANCE.md)
3. **Test thoroughly** (imports + integration)
4. **Document well** (update CHANGELOG.md, README.md)
5. **Keep it minimal** (resist feature creep)

## Resources

- **Original Demucs**: https://github.com/facebookresearch/demucs
- **UV Documentation**: https://github.com/astral-sh/uv
- **Our Package**: https://github.com/ChenPaulYu/demucs-infer

---

**Remember**: Simple is better than complex. When in doubt, do less, not more.

