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

# Resolve the destination hooks directory via git's own plumbing rather than
# assuming "$REPO_ROOT/.git/hooks". Inside a linked worktree, .git is a FILE
# (a gitdir pointer), not a directory, so the old assumption always failed with
# "are you in a git repo?". --git-common-dir yields the *shared* gitdir whose
# hooks/ is used by the main checkout AND every linked worktree.
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)"
if [[ -z "$GIT_COMMON_DIR" ]]; then
    echo "error: not inside a git repository (git rev-parse --git-common-dir failed)" >&2
    exit 1
fi
# --git-common-dir may be relative (commonly ".git"); resolve it to an absolute
# path so this script works regardless of the caller's working directory.
case "$GIT_COMMON_DIR" in
    /* | [A-Za-z]:* | \\*) ;;                       # already absolute (POSIX or Windows)
    *) GIT_COMMON_DIR="$(cd "$REPO_ROOT" && cd "$GIT_COMMON_DIR" && pwd)" ;;
esac
HOOKS_DST="$GIT_COMMON_DIR/hooks"

if [[ ! -d "$HOOKS_SRC" ]]; then
    echo "error: scripts/hooks/ not found at $HOOKS_SRC" >&2
    exit 1
fi

mkdir -p "$HOOKS_DST"

DRY=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY=true
fi

installed=0
echo "Hooks directory: $HOOKS_DST"
echo "(This is the shared gitdir — hooks apply to the main checkout and every linked worktree.)"
echo ""
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
    echo "Done — $installed hook(s) installed into the shared gitdir."
    echo "They now run for this checkout and all linked worktrees of this repo."
    echo "To bypass temporarily: git commit --no-verify"
fi
