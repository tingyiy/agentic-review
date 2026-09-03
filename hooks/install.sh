#!/usr/bin/env bash
# Install the commit-msg hook into a checkout. Idempotent.
#
#   hooks/install.sh              # this repository
#   hooks/install.sh ~/src/your-repo
#
# A COPY, not a symlink: a symlink into this repo breaks the target's commits
# the moment somebody moves or deletes it, and a hook that fails to execute is
# reported by git as a refused commit with no explanation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$(cd "$HERE/.." && pwd)}"

GIT_DIR="$(git -C "$TARGET" rev-parse --git-dir 2>/dev/null)" || {
  echo "not a git checkout: $TARGET" >&2; exit 1; }
# `rev-parse --git-dir` is relative when run from inside the checkout.
case "$GIT_DIR" in /*) ;; *) GIT_DIR="$TARGET/$GIT_DIR" ;; esac

# `core.hooksPath` wins over .git/hooks when it is set, so installing into
# .git/hooks would silently do nothing. This is a real configuration in
# repositories using husky or pre-commit.
CONFIGURED="$(git -C "$TARGET" config --get core.hooksPath || true)"
if [ -n "$CONFIGURED" ]; then
  case "$CONFIGURED" in /*) DEST="$CONFIGURED" ;; *) DEST="$TARGET/$CONFIGURED" ;; esac
  echo "note: core.hooksPath is set — installing into $DEST"
else
  DEST="$GIT_DIR/hooks"
fi

mkdir -p "$DEST"
if [ -e "$DEST/commit-msg" ] && ! grep -q "agentic-review" "$DEST/commit-msg" 2>/dev/null; then
  echo "refusing to overwrite an existing commit-msg hook at $DEST" >&2
  echo "move it aside, or chain to hooks/commit-msg from it" >&2
  exit 1
fi
cp "$HERE/commit-msg" "$DEST/commit-msg"
chmod +x "$DEST/commit-msg"
echo "installed $DEST/commit-msg"
