#!/usr/bin/env bash
# Install git hooks for the onnx-genai repository.
#
# Usage:
#   scripts/install-hooks.sh        # install all hooks
#   scripts/install-hooks.sh --dry  # show what would be installed
#
# Safe to re-run — overwrites only hooks shipped in scripts/hooks/.
# Does NOT clobber hooks that are not in this repository's scripts/hooks/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [[ ! -d "$HOOKS_SRC" ]]; then
    echo "error: scripts/hooks/ not found at $HOOKS_SRC" >&2
    exit 1
fi

if [[ ! -d "$HOOKS_DST" ]]; then
    echo "error: .git/hooks/ not found — are you in a git repo?" >&2
    exit 1
fi

DRY=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY=true
fi

installed=0
for hook in "$HOOKS_SRC"/*; do
    [[ -f "$hook" ]] || continue
    name="$(basename "$hook")"
    dst="$HOOKS_DST/$name"

    if [[ "$DRY" == true ]]; then
        echo "[dry-run] would install: $name → $dst"
    else
        cp "$hook" "$dst"
        chmod +x "$dst"
        echo "✓ installed: $name"
    fi
    installed=$((installed + 1))
done

if [[ $installed -eq 0 ]]; then
    echo "warning: no hooks found in scripts/hooks/" >&2
    exit 1
fi

if [[ "$DRY" == false ]]; then
    echo ""
    echo "Done — $installed hook(s) installed."
    echo "To bypass temporarily: git commit --no-verify"
fi
