# REPLACEMENT_MAP — third-party-derived brand/voice artifacts → original assets

Part of task c7363907 (original replacement brand/voice assets). Primary
deliverables: `assets_original/` (icons, branding, audio) + PROVENANCE.md.

## 1. What is replaced (measured — repo inventory)
| Old (third-party-derived) | New (original, ours) |
|---|---|
| `cortana-shell/assets/icons/cortana.png` | `assets_original/jarvis_icon.png` |
| `cortana-shell/assets/branding/cortanadouble.png` | `assets_original/jarvis_branding_double.png` |
| `cortana-shell/assets/wake-word-models/heycortana_en-US.table` | `assets_original/jarvis_wake_chime.wav` |
| `cortana-shell/assets/wake-word-models/heycortana_enIN.table` | `assets_original/jarvis_wake_chime.wav` |
| `cortana-shell/assets/wake-word-models/heycortana_enUS.table` | `assets_original/jarvis_wake_chime.wav` |
| `cortana-shell/assets/wake-word-models/heycortana_zhCN.table` | `assets_original/jarvis_wake_chime.wav` |
| `config.yaml` `voice.wakeWord.phrase: "hey cortana"` | `"hey jarvis"` (original wake phrase; **not** a Cortana trademark) |
| `config.yaml` `voice.wakeWord.modelPath: .../heycortana_enUS.table` | `.../assets_original/jarvis_wake_chime.wav` (after engineer re-wire) |

## 2. New asset inventory (all original — provenance in PROVENANCE.md)
- `assets_original/jarvis_icon.png` — icon glyph (1025×1031)
- `assets_original/jarvis_branding_double.png` — two-panel brand (2068×1144)
- `assets_original/jarvis_wake_chime.wav` — 3.2 s, 44.1 kHz mono, 41.02 Hz resonance sonification
- `assets_original/jarvis_identity_tone.wav` — 2.4 s, optional voice-identity tone
- `generate_original_assets.py` — deterministic generator (seeds 20260925/777/778)
- `MANIFEST.md` — per-file hashes and generation notes (in assets_original/)

## 3. How to re-verify
```bash
python3 generate_original_assets.py   # regenerates identical files (fixed seeds)
ls assets_original/ && sha256sum assets_original/*.wav
```

## 4. Engineer follow-ups (NOT done here — wiring decisions belong to engineer/lead)
1. **Wake-word re-wiring**: `VoiceManager.loadWakeWordModel()` reads a binary
   `.table`; it cannot load a `.wav`. Decide: (a) replace wake-word detection
   with the resonance chime as a brand/status sound (asset-only), or (b) wire an
   original model-free wake detector and point `modelPath` at
   `jarvis_wake_chime.wav`. Until then, leave `config.yaml` untouched so the
   shell does not break.
2. **Deletion of tracked third-party binaries** (`cortana.png`,
   `cortanadouble.png`, `heycortana_*.table`) from the tipped tree: owner/lead
   decision; history will still contain them (unavoidable).
3. Optionally update the site copy (currently says "no extracted Cortana assets"
   on the landing page — now true only after deletions land).

## 5. Status
- [x] Original assets generated + verified (this PR)
- [ ] config.yaml phrase/modelPath swap (engineer)
- [ ] third-party binaries removed from tipped tree (lead/engineer decision)
- [ ] site copy re-check once deletions land