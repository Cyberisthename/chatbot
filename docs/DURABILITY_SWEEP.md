# Durability Sweep — Inventory of local-only approved work (2026-10-05)

**Task:** 20548ae6-e6fa-4968-95a7-e76ea23be443 (engineer) · **PR:** https://github.com/Cyberisthename/chatbot/pull/153
**Method:** content-hash comparison of every file under `/home/team/shared/` (and member homes, bounded scan) against `origin/main` blobs (`git ls-tree -r origin/main`, `git hash-object` on candidates). Excluded per task: `docs/validation/` (owned by a839d0af), the repo checkout itself, `node_modules/`, `.git/`, `dist/`, caches, zero-byte files. All verdicts re-verifiable by fresh run of the scan script.
**Baseline:** `origin/main` @ `db56770` (after fetch; includes PRs #149/#150/#151 landed during the sweep — re-fetched and re-scanned; numbers below are against the final ref).

## Summary (counts per verdict bucket)
- **Scanned candidates:** 730 (non-empty files under /home/team/shared, excluding the repo tree, validation/, and heavy dirs)
- **ALREADY DURABLE (content identical to a blob in origin/main):** 112
- **STRANDED — committed in this sweep's PR:** 2 (below)
- **NOT-AN-ARTIFACT (superseded / duplicate / generated / working state / third-party vendored / team infra):** 615
- **OPEN (another member's in-flight deliverable or needs lead routing):** 1
- **Third-party vendored corpus (excluded from commit, kept on disk):** 605 of the above (indus/corpus_raw)
- **Nothing approved was deleted.** Everything stays on disk; this sweep is additive only.

## COMMITTED this sweep (2 files — clearly stranded, clearly approved)

| Local file | Repo destination | Why approved |
|---|---|---|
| `/home/team/shared/nitrogenase_seed_hygiene_verdict.md` | `docs/artifacts/nitrogenase_qudit/nitrogenase_seed_hygiene_verdict.md` | Deliverable of task 7a8d1b21 (done): seed hygiene verdict — scratch OUT_DIR only. Companion to already-tracked nitrogenase artifacts. |
| `/home/team/shared/jarvis_rnd_server_restart_smoke.txt` | `docs/artifacts/jarvis_server_restart_smoke.txt` | Re-verification smoke evidence from the approved server-restart work (2026-10-05, after /var/tmp wipe); sibling of tracked `docs/artifacts/jarvis_server_smoke_test.txt` from PR #147. |

## NOT-AN-ARTIFACT (615) — kept on disk, nothing to commit

### Superseded by already-durable tracked versions (blob-in-origin/main exists)
- `quantum_likeness/QUDIT_LHV_WRAPUP.md` — interim status wrap-up (2026-09-28) for the crown-jewel task; the task's durable deliverable `QUDIT_LHV_REPORT.md` + probe + results are tracked (PR #146).
- `qml/bell_chsh_prob.py`, `qml/bell_chsh_results.json` (2026-09-24) — earlier Bell-CHSH probe; superseded by `likeness_probes.py` (tracked).

### Content-equal regenerated duplicates (same approved numbers, no new information)
- `nitrogenase_qudit_report.json`, `NITROGENASE_QUDIT_REPORT.md` (top level) and the identical copies under `nitrogenase_artifacts/` — verified: energy barrier **100.7657** identical to tracked `docs/artifacts/nitrogenase_qudit/*`; only the `date` field differs (2026-09-28 vs 2026-09-24).
- `nitrogenase_artifacts/owner_quantum_seed_d2.json` — same seed as tracked; differs only by added metadata fields (`fold_depth`, `qudit_dim`).

### Site working-tree state (tracked paths already in origin/main; local copies differ)
- `site/SITE.md`, `site/bun.lock` — tracked files whose local working copy has since evolved (site is the live Quantum Compressor Lab; working-state diffs are not stranded artifacts).
- `site/src/routeTree.gen.ts` — generated file (TanStack route tree), not source-of-truth.

### Team infrastructure (lives in shared by design)
- `skills/jarvis-rnd-battery-server/SKILL.md` — team skill, auto-discovered from `/home/team/shared/skills/`; not a repo artifact.

### Third-party vendored input data (NOT ours; never commit wholesale)
- `indus/corpus_raw/` (605 files) — a clone of the third-party "Corpus of Indus Seals and Inscriptions digitization" repo (contains its own `.git`, LICENSE, Cargo.toml). Input data only; the team's own Indus artifacts (H1–H4 harness, NORMALIZE_RB) are already tracked under `docs/artifacts/indus/`.

## OPEN (needs lead routing; not committed — would race another member's in-flight work or awaits a routing decision)
1. `compliance/GOVERNANCE_GATE.md` (8258 B, dated 2026-10-05, authored by agent-legal-manager, task b2008e8c = done) — the fresh Governance Gate closeout memo ("GO-WITH-CONDITIONS"), created *after* PR #149 merged (17:57). Companion docs (`GOVERNANCE_GATE_BIO_QUANTUM.md`, `ORIGINALITY_AUDIT.md`) are tracked; the closeout memo itself is not. **Recommendation:** the legal manager or lead should file this (likely their imminent follow-up PR). I did not commit it to avoid racing it.

## Already durable — notable confirmations
- `WORKFLOW.md`, `RESTORE_CHECKOUT.sh` — now durable via PR #151 (landed during the sweep; verified by re-scan: stranding cleared for both).
- All Validation Track 3/3 artifacts — durable via PR #150 (`docs/validation/`).
- Governance compliance docs — durable via PR #149.
- `docs/artifacts/jarvis/`, `anyon/`, `indus/`, `qml/`, `quantum_likeness/`, `original_assets/`, `branding_voice/`, `showoff/`, `seedopt/`, `nitrogenase_qudit/`, `voynich/` (tracked), `site/` (tracked source) — all matched content-hash.
- `New folder/` (repo path) = owner-uploaded content (commit f556e96, 45 files) — confirmed tracked in origin/main; untouched, per instructions.

## Working-tree note (shared checkout)
- `git status --porcelain` at the shared checkout shows one untracked file: `scripts/braid_vs_gd_benchmark.py` — not part of this sweep (another member's in-progress work, likely the braid-vs-GD trainer); left untouched.

## Re-verification
- Scan script (reproducible): `/home/agent-engineer/durability_scan2.py` + `durability_scan2.json` (private workspace; run `python durability_scan2.py` after `git -C /home/team/shared/chatbot fetch origin main`).
- Every verdict above is (a) measured against origin/main content hashes; none are from memory.