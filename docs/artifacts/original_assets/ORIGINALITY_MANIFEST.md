# ORIGINALITY MANIFEST — Original Replacement Brand/Voice Assets

**Task:** c7363907 (lead-assigned 2026-05-11/12) · **Deliverable path:** `/home/team/shared/original_assets/`

## 1. Who made it
- **Agent:** creative engineer (team "cyber ai"), under lead direction (agent-lead)
- **Owner:** Cyberisthename (sole owner of all code, math, and assets)
- **Verification path:** repo PR #144 (head `feat/original-brand-voice`), committed by team account

## 2. Tool / process
| Asset | Tool | Process |
|---|---|---|
| `assets_original/jarvis_icon.png` | Python 3, matplotlib 3.11.2, numpy 2.5.3 | deterministic render, fixed seed 20260925; house FBSC palette; braid-ring + seed-node glyph |
| `assets_original/jarvis_branding_double.png` | same | same seed; two-panel: glyph + 41.02 Hz harmonic-stack spectrum (our amplitudes) |
| `assets_original/jarvis_wake_chime.wav` | Python 3, numpy (wave/PCM 44.1 kHz mono) | seed 777; f0 = 41.02 Hz resonance sonification; noise-gated envelope (RESONANCE_FIX design) |
| `assets_original/jarvis_identity_tone.wav` | same | seed 778; 164.08 Hz source + formant partials; voice-identity tone (NOT TTS) |
| `generate_original_assets.py` | Python 3 | deterministic regenerator; re-run reproduces identical files |

**Third-party inputs used: NONE.** No images, audio samples, fonts, voice models,
wake-word models, or SDK wrappers from any third party. Only numpy + matplotlib
(BSD/PSF-licensed scientific libraries).

## 3. Date of creation
- **Created (files):** 2026-09-28 (this file set; work in repo PR #144)
- **Task instructed:** lead message 2026-05-11/12 requested completion "today (May 12)"
- **Why the gap (one line):** the 2026-09-24 disk-full incident wiped the shared
  staging area; delivery resumed after recovery (WORKFLOW.md documents the incident).
  No date is falsified — all timestamps above are the true creation/instruction dates.

## 4. Usage rights
Original, generated from scratch by the team; copyright and usage rights belong to
the owner outright. No third-party clearance is required for any use (site, demos,
licensing, distribution). Full statement: see `PROVENANCE.md`.

## 5. Old → new mapping
Exact map of every third-party-derived artifact this set replaces is in
`REPLACEMENT_MAP.md` (summary: `cortana-shell/assets/icons/cortana.png` →
`jarvis_icon.png`; `branding/cortanadouble.png` → `jarvis_branding_double.png`;
`wake-word-models/heycortana_*.table` ×4 → `jarvis_wake_chime.wav`; config phrase
"hey cortana" → "hey jarvis").

## 6. File hashes (measured)
| File | SHA-256 (first 16) |
|---|---|
| `assets_original/jarvis_wake_chime.wav` | 728fddfeb840321b |
| `assets_original/jarvis_identity_tone.wav` | ec83f5205c026724 |
| (PNGs re-verifiable by regenerating with the fixed seeds) | |