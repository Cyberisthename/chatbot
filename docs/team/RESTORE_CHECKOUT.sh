#!/usr/bin/env bash
# RESTORE_CHECKOUT.sh — rebuild the shared checkout's git metadata after the
# volatile volume (/var/tmp, which shares the 3.1G overlay with /) is wiped.
#
# Tested 2026-10-09 on a real wipe (PR #150 / commit a407687 was in flight when
# it happened). This script was authored from that recovery.
#
# DESIGN RULE — this script is NON-DESTRUCTIVE:
#   It never runs `git clean`, `git reset --hard`, `git checkout -f`, or
#   `git stash`. Those would delete untracked files, which are the ONLY copy of
#   un-pushed work. On 2026-10-09 the untracked files it protected were the sole
#   surviving copy of a member's benchmark work.
#
# Usage:  bash /home/team/shared/RESTORE_CHECKOUT.sh
set -euo pipefail

CHECKOUT=/home/team/shared/chatbot
GITSTORE=/var/tmp/chatbot-fresh
REMOTE=https://github.com/Cyberisthename/chatbot.git

echo "=== 1. is the checkout actually broken? ==="
if git -C "$CHECKOUT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "git works already: $(git -C "$CHECKOUT" log --oneline -1)"
  echo "Nothing to do. (Refusing to run, so we never risk a healthy tree.)"
  exit 0
fi
echo "checkout git is broken (expected after a /var/tmp wipe). Proceeding."

echo
echo "=== 2. INVENTORY THE WORKING TREE BEFORE TOUCHING ANYTHING ==="
echo "Files under $CHECKOUT survive wipes (it is on the separate /home fs)."
if [ -f "$CHECKOUT/.git" ]; then
  echo "stale .git pointer: $(cat "$CHECKOUT/.git")"
fi
echo "top-level entries: $(ls -A "$CHECKOUT" | wc -l)"
echo "If members have work in flight, note it now: git log/status are unavailable,"
echo "so any unpushed COMMITS are already gone. Unpushed commits live in the object"
echo "store that was just wiped. Uncommitted FILES still exist and are protected below."

echo
echo "=== 3. rebuild the git store in /var/tmp (roomy, but VOLATILE) ==="
echo "note: /home has only ~500M free, so the object store must live in /var/tmp."
df -h / /home | tail -3
if [ ! -d "$GITSTORE/.git" ]; then
  rm -rf "$GITSTORE"
  git clone --no-checkout "$REMOTE" "$GITSTORE"
else
  echo "reusing existing clone at $GITSTORE"
fi

echo
echo "=== 4. wire the checkout to the store WITHOUT changing any file ==="
git -C "$GITSTORE" config core.bare false
git -C "$GITSTORE" config core.worktree "$CHECKOUT"
printf 'gitdir: %s/.git\n' "$GITSTORE" > "$CHECKOUT/.git"

echo
echo "=== 5. rebuild the index from HEAD (touches the index only, never files) ==="
git -C "$CHECKOUT" fetch origin -q
git -C "$CHECKOUT" reset --mixed HEAD >/dev/null

echo
echo "=== 6. materialise tracked files that the new HEAD adds and the tree lacks ==="
echo "(\`git checkout -- .\` restores tracked files from the index;"
echo " it never deletes untracked files, which is why it is safe here.)"
git -C "$CHECKOUT" checkout -- .

echo
echo "=== 7. RESULT — verify, and PRESERVE the untracked files listed ==="
git -C "$CHECKOUT" log --oneline -1
git -C "$CHECKOUT" rev-parse --abbrev-ref HEAD
echo "--- untracked files (DO NOT CLEAN — these may be the only copy) ---"
git -C "$CHECKOUT" status --porcelain | grep '^??' || echo "(none)"
echo "--- remaining status ---"
git -C "$CHECKOUT" status --short
echo
echo "DONE. Remind the team: never run git clean / reset --hard / stash in this"
echo "checkout while untracked work is present. And push branches early — a branch"
echo "that exists only on this disk does not exist."
