#!/bin/bash
# Script to bump version for bug fixes or new releases
# Usage: ./scripts/bump_version.sh [patch|minor|major]
#
# Note: demucs-infer uses semantic versioning (MAJOR.MINOR.PATCH)
# Version prefix (4.x.x) indicates compatibility with Demucs v4 model architectures.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 [patch|minor|major]"
    echo "  patch: 4.1.0 -> 4.1.1 (bug fixes)"
    echo "  minor: 4.1.0 -> 4.2.0 (new features)"
    echo "  major: 4.1.0 -> 5.0.0 (breaking changes)"
    exit 1
fi

BUMP_TYPE=$1
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

case $BUMP_TYPE in
    patch)
        PATCH=$((PATCH + 1))
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    *)
        echo "Error: Invalid bump type. Use patch, minor, or major"
        exit 1
        ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"

echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION"

# Update pyproject.toml
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml

# Update demucs_infer/__init__.py
sed -i "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" demucs_infer/__init__.py

# Update README.md if it has version references (optional, may not exist)
if grep -q "Current Version\|Version:" README.md 2>/dev/null; then
    sed -i "s/\*\*Current Version:\*\* $CURRENT_VERSION/\*\*Current Version:\*\* $NEW_VERSION/" README.md || true
    sed -i "s/Version: $CURRENT_VERSION/Version: $NEW_VERSION/" README.md || true
fi

echo "✅ Updated pyproject.toml, demucs_infer/__init__.py, and README.md to version $NEW_VERSION"
echo ""
echo "Next steps:"
echo "1. Update CHANGELOG.md with your changes"
echo "2. Commit: git add -A && git commit -m \"Bump version to $NEW_VERSION\""
echo "3. Tag: git tag v$NEW_VERSION"
echo "4. Push: git push origin main && git push origin v$NEW_VERSION"
echo "5. Create release using GitHub CLI:"
echo "   gh release create v$NEW_VERSION --title \"v$NEW_VERSION\" --notes-file .github/release-notes-v$NEW_VERSION.md"
echo ""
echo "   This will trigger the GitHub Actions workflow to publish to PyPI automatically."
