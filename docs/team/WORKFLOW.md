<!-- managed:linked-repos -->
## Linked Repositories
- Cyberisthename/chatbot
<!-- /managed:linked-repos -->

# Code Workflow

## Linked Repositories
- Cyberisthename/chatbot

## Default Process
1. Members push code to feature branches and create pull requests
2. The team lead reviews and merges PRs
3. Before starting new work, members should pull the latest default branch so they branch from up-to-date code

## Shared Checkout Discipline (added 2026-09-30 after PR #143 / PR #144 branch race)
The canonical checkout `/home/team/shared/chatbot` is a **detached-HEAD worktree SHARED BY ALL
MEMBERS** (its `.git` metadata lives at `/var/tmp/chatbot-fresh/.git`). When several members
branch/commit/push through the SAME working directory concurrently, race conditions occur:
- 2026-09-30: a branch `fix/jarvis-artifact-sync` was created in the shared checkout for PR #143
  (jarvis sync) while another member's brand/voice work was staged; GitHub re-attached the open PR
  to a same-named pushed branch, and a commit briefly landed on the wrong branch (repaired, zero
  data loss). The creative engineer's brand/voice PR had to be re-filed as PR #144.
Rules going forward:
1. **One git operation at a time in the shared checkout.** Before branching, `cd
   /home/team/shared/chatbot`, `git fetch origin`, `git checkout --detach origin/main`, and
   `git status --porcelain` — confirm the tree is clean BEFORE you branch. If `git status` shows
   staged/unstaged changes you did NOT create, STOP: another member is mid-edit. Do not commit
   their work, do not reset it; tell the lead and wait.
2. **Never create a branch with a generic name that another task might also use** (e.g.
   `fix/jarvis-artifact-sync`). Name it after THE TASK (e.g. `docs/factcheck-verified`). If the
   name already exists on origin, branch with a unique suffix.
3. **State `--head` explicitly when creating a PR** (`gh pr create --head <branch>`) so the PR is
   pinned to YOUR branch; verify the PR's file list matches your branch's diff before announcing.
   To inspect a PR's true contents at any time use `gh pr view <n> --json files,headRefName` and
   `gh pr diff <n> --name-only` — trust those, not remembered titles.
4. If a PR number turns out to be attached to the wrong branch (same-name collision): do NOT force
   anything in the shared checkout. Open a NEW PR from the correct branch with `--head`, retitle
   the collided PR to describe what it actually contains, and notify the lead.
5. Preferred alternative when running multi-step local work: use a **per-member scratch worktree**
   (`git -C /var/tmp/chatbot-fresh worktree add --detach /var/tmp/<member>-wt origin/main`), work
   there, push from there, and remove it when done. This completely isolates your branch operations
   from the shared checkout. Remember /var/tmp can be wiped; the worktree is re-creatable, and
   origin is the durable store — nothing of value should live only in a scratch worktree.

## Disk Hygiene (added 2026-09-24 after disk-full incident)
The `/home` filesystem is only **2 GB**. It filled to 100% when several members each kept a full
clone of the repo in their home dir (`~agent-*/chatbot`, each with a 110–450 MB `.git` + 56 MB
`releases/`). That blocked the team DB and corrupted the canonical repo's git metadata mid-write.
Rules going forward:
1. **Do NOT clone the repo into your home directory.** The canonical checkout is
   `/home/team/shared/chatbot` (a git worktree, HEAD detached at origin/main; its git metadata
   lives at `/var/tmp/chatbot-fresh/.git`). Work there or operate on it via `git -C`.
2. If you must branch, branch inside `/home/team/shared/chatbot` or use a shallow clone in
   `/var/tmp`; never leave a second full `.git` in `/home`.
3. Check `df -h /home` before any large write. Keep `/home` under ~80%.
4. If `/home` ever fills again: the DB (`team-db`) and git both fail with "no space". Free space
   first (delete re-creatable build output/caches; move stale clones to `/var/tmp`), then retry.
5. Backups from the 2026-09-24 recovery live in `/var/tmp`: `chatbot-old-backup` (pre-swap tree
   incl. uncommitted extras), `chatbot-fresh` (full clone, source of the worktree's git dir),
   `team-stale-clones/` (two moved member clones). Origin on GitHub is the source of truth for
   everything committed; if `/var/tmp` is ever wiped, re-clone from origin and re-add the worktree:
   `cd /var/tmp && git clone https://github.com/Cyberisthename/chatbot.git chatbot-fresh &&
   git -C chatbot-fresh worktree add --detach /home/team/shared/chatbot origin/main`.

## Artifact Durability (added 2026-09-24 after second loss in disk-full aftermath)
The 2026-09-24 disk-full event went deeper than git metadata: files in `/home/team/shared` that
were verified present on 2026-09-22 are gone from disk (directory skeletons kept, contents lost;
mechanism not confirmed — observed loss, not diagnosis).
Lost from disk: `jarvis/INDEX.md`, the approved trainer (`jarvis/trainer/`), the Validation
Track 3/3 artifacts (`validation/`), QLM and anyon artifacts (`qml/`, `anyon/`),
`fbsc_efficiency_report.json`, `seedopt_results.json`, base `owner_quantum_seed.json`.
The owner seed VALUES survive (git: `docs/artifacts/nitrogenase_qudit/artifacts/owner_quantum_seed_d{2,3,4}.json`
and hardcoded in `compression_specialist.py`); every approved deliverable's spec + measured
numbers survive in the team DB task results, so regeneration is fully specified.
Rules going forward:
1. **Any approved deliverable (report, results JSON, INDEX, script) must be committed to git
   (origin/main via PR) — never leave approved work only in `/home/team/shared`.**
2. `/home/team/shared` is a working/staging area, NOT an archive. Treat anything there as
   deletable; treat origin/main as the only durable store.
3. Before finishing a task, verify the artifact exists in the repo (or the PR diff), not just
   in the shared dir.
4. Do not `git gc` expecting to shrink packs — big binary blobs are reachable (2026-09-24 lesson).

## Volatile-volume wipe + PUSH EARLY (added 2026-10-09 after third /var/tmp wipe)
The 3.1G overlay shared by `/` and `/var/tmp` was reset again (`/var/tmp` came back empty,
`/` at 1% used). `/home` is a separate filesystem and was untouched. Consequences observed:

1. **The shared checkout's git metadata died.** `/home/team/shared/chatbot/.git` is a pointer
   file (`gitdir: /var/tmp/chatbot-fresh/.git`) into the volatile volume. When that went, every
   `git` command in the shared checkout failed with
   `fatal: not a git repository: /var/tmp/chatbot-fresh/.git/worktrees/chatbot`.
   **Fix — one command, tested, non-destructive:**
   `bash /home/team/shared/RESTORE_CHECKOUT.sh`
   It refuses to run if git already works, and it never runs `git clean`, `git reset --hard`,
   `git checkout -f`, or `git stash`. Read the top of that script before editing it.
2. **Unpushed branches do NOT survive — this is the real lesson.** A member's branch
   `bench/braid-vs-gd-structured-0b990cee` was never pushed; after the wipe GitHub returned 404
   for it and its commits were unrecoverable. Their *files* survived (untracked files live in
   `/home`, which does not get wiped), but the branch and its history did not.
   **Rule: push your branch the moment it exists — `git push -u origin <branch>` before you
   write the second commit.** A branch that exists only on this machine does not exist.
   Assume every session can lose `/var/tmp` without warning.
3. **Never destroy untracked files in the shared checkout.** After a wipe, untracked files are
   often the only surviving copy of a member's work. Do not run `git clean`, `git stash`,
   `git reset --hard`, or `git checkout .` in `/home/team/shared/chatbot` while untracked work
   is present. `git checkout -- .` is safe (it restores tracked files from the index and never
   deletes untracked ones) — that is the only checkout form used by the restore script.
4. **The public site is unaffected by these wipes.** Both the working and live site URLs are
   served by the platform, not by a local process; they stayed at HTTP 200 throughout. The R&D
   battery server (`:8777`) and any local dev server (`:3000`) DO die, because their venvs and
   processes live on the volatile volume — recreate them, they are re-creatable.
5. **Local SQLite provenance survives** (it is in `/home`), but treat origin/main as the only
   durable store for approved work. Same rule as the Artifact Durability section above.

## Notes
- The team lead can update this file to reflect the owner's preferences
- If the owner provides specific instructions about code review, branch strategy, or merge policies, update this document accordingly

## Owner Preference Updates (2026-05-08)
- On request, create a full **checkpoint snapshot** to GitHub that includes current code and artifact outputs.
- Keep using feature branch + PR flow for checkpoint snapshots unless the owner explicitly asks for a direct push to default branch.
- For large artifacts, include them in-repo when feasible; if any file exceeds platform limits, store the artifact in an approved alternative and commit a manifest/index file with exact paths and hashes.
- After checkpoint snapshot creation, active experiment work continues without interruption.
