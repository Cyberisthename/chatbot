#!/usr/bin/env python3
"""
Original Replacement Brand/Voice Assets — fully owned, deterministic, no third-party derivatives.

Replaces (in repo):
  cortana-shell/assets/icons/cortana.png            -> jarvis_icon.png       (original braid glyph)
  cortana-shell/assets/branding/cortanadouble.png   -> jarvis_branding_double.png (original two-panel brand)
  cortana-shell/assets/wake-word-models/heycortana_*.table -> jarvis_wake_chime.wav (original 41.02 Hz sonification)

All visual output: matplotlib, our own colormap/style (same house look as show-off pack).
All audio output: numpy synthesis of the owner-ratified 41.02 Hz bio-resonance trigger
(F0 = 41.02 Hz, harmonic stack, noise-gated envelope per RESONANCE_FIX). No samples,
no third-party TTS, no external audio files.

Fixed seeds for reproducibility: 20260925 (visual), 777 (audio jitter).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import wave, struct, os, hashlib

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_original")
os.makedirs(OUT, exist_ok=True)

BG = "#0b0e17"
NEON = ["#00e5ff", "#7c4dff", "#ff2ea6", "#39ff88"]  # house FBSC palette

# ---------------------------------------------------------------- helpers
def seed_rng(seed):
    return np.random.default_rng(seed)

def save_meta(tag, data):
    with open(os.path.join(OUT, tag), "w") as f:
        f.write(data)

# ================================================================ 1. ICON
def make_icon():
    """Original jarvis glyph: 3-strand braid ring around a seed node (FBSC motif)."""
    rng = seed_rng(20260925)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.axis("off")

    t = np.linspace(0, 2 * np.pi, 400)
    for k, col in enumerate(NEON):
        # braid strand: circle with phase offset + radial wobble from seed
        wob = 0.10 * np.sin(3 * t + k * 2.094 + 0.4 * rng.standard_normal(1)[0])
        r = 0.78 + wob
        x, y = r * np.cos(t + k * 2.094), r * np.sin(t + k * 2.094)
        ax.plot(x, y, color=col, lw=2.4, alpha=0.95, solid_capstyle="round")

    # central seed node (owner-seed motif: 3 dots -> 1 core)
    for (dx, dy, s, c) in [(0.0, 0.0, 0.34, NEON[3]), (-0.30, 0.26, 0.055, NEON[0]), (0.30, 0.24, 0.055, NEON[0]), (0.0, -0.32, 0.055, NEON[0])]:
        ax.add_patch(Circle((dx, dy), s, color=c, alpha=0.95, zorder=5))
    # faint outer pulse ring (measured-value ring echo)
    for rr, al in [(0.95, 0.10), (1.02, 0.05)]:
        ax.add_patch(Circle((0, 0), rr, fill=False, ec=NEON[1], lw=1.0, alpha=al))

    p = os.path.join(OUT, "jarvis_icon.png")
    fig.savefig(p, facecolor=BG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return p

# ====================================================== 2. BRANDING DOUBLE
def make_branding_double():
    """Original two-panel brand: left = braid glyph, right = fold/harmonic spectrum."""
    rng = seed_rng(20260925)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=220)
    for ax in axs:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # LEFT: glyph (simplified ring+core)
    ax = axs[0]; ax.axis("off"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    t = np.linspace(0, 2 * np.pi, 300)
    for k, col in enumerate(NEON):
        r = 0.8 + 0.09 * np.sin(3 * t + k * 2.094)
        ax.plot(r * np.cos(t + k * 2.094), r * np.sin(t + k * 2.094), color=col, lw=2.2, alpha=0.95)
    ax.add_patch(Circle((0, 0), 0.30, color=NEON[3], alpha=0.95))

    # RIGHT: harmonic amplitude spectrum of the 41.02 Hz resonance (our own data)
    ax = axs[1]
    freqs = 41.02 * np.arange(1, 9)
    amps = 1.0 / np.arange(1, 9) ** 0.8
    amps *= 1 + 0.35 * np.sin(0.7 * np.arange(8) + 0.3 * rng.standard_normal(8))  # seed-tinted
    ax.bar(np.arange(8), amps, color=NEON, alpha=0.9, width=0.62)
    ax.set_title("41.02 Hz resonance · harmonic stack", color="white", fontsize=10, loc="left")
    ax.set_facecolor(BG)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xticks(np.arange(8)); ax.set_xticklabels([f"{int(f)}" for f in freqs], rotation=45)
    ax.set_yticks([])

    p = os.path.join(OUT, "jarvis_branding_double.png")
    fig.savefig(p, facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return p

# ================================================================== AUDIO
FS = 44100

def synth_wake_chime(dur=3.2, seed=777):
    """Original wake asset: noise-gated sonification of 41.02 Hz resonance.

    f0 = 41.02 Hz (owner-ratified bio-quantum trigger, RESONANCE_FIX).
    Harmonic stack -> 'crystal' bell timbre; attack/decay envelope = noise gate
    (no signal before gate opens, hysteresis-shaped). Jitter from fixed seed.
    """
    rng = seed_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)

    # three bowed bell partials built from harmonics of f0 (original timbre)
    for (mult, a, dec) in [(3, 0.55, 1.8), (5, 0.30, 2.6), (8, 0.18, 1.2)]:
        f = 41.02 * mult
        phase = 2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)
        x += a * np.sin(phase) * np.exp(-t / dec)
    # subharmonic pulse (the 41.02 Hz trigger itself, strongly gated)
    x += 0.35 * np.sin(2 * np.pi * 41.02 * t) * np.exp(-t / 0.9)
    # faint high 'identity chime' at seed-derived detune
    det = 41.02 * 47 * (1 + 0.004 * rng.standard_normal(1)[0])
    x += 0.10 * np.sin(2 * np.pi * det * t) * np.exp(-t / 3.0)

    # noise gate: no output before gate opens (~0.35 s), smooth trapezoid
    gate = np.minimum(1.0, t / 0.12) * np.minimum(1.0, (dur - t) / 0.6)
    gate[: int(0.35 * FS)] *= 0  # hard pre-gate silence (like noise-gated detector)
    x *= gate

    # normalise to -1 dBFS with headroom
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.89
    pcm = (x * 32767).astype(np.int16)
    p = os.path.join(OUT, "jarvis_wake_chime.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return p, hash_sha(p)

def synthesize_identity_tone(dur=2.4, seed=777):
    """Optional longer 'voice identity' tone: formant-shaped resonance sweep.

    Represents the adapter-driven vocal identity (F0 by complexity, resonant
    'crystalline' texture) entirely from our equations — NOT a TTS voice.
    """
    rng = seed_rng(seed + 1)
    n = int(dur * FS)
    t = np.arange(n) / FS
    f0 = 41.02 * 4  # 164.08 Hz — audible representative of the resonance stack
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 2 * np.pi))
    ph = 2 * np.pi * np.cumsum(f0 * vib) / FS
    x = np.sin(ph + 0.55 * np.sin(2 * np.pi * 3.1 * t))
    # add formant-ish resonances (F1 F2 F3 tones, amplitude shaped)
    for (fm, a) in [(6.1, 0.24), (11.7, 0.15), (19.3, 0.09)]:
        x += a * np.sin(2 * np.pi * f0 * fm * t) * np.exp(-0.6 * t)
    env = np.minimum(1.0, t / 0.15) * np.minimum(1.0, (dur - t) / 0.8)
    x *= env
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.82
    pcm = (x * 32767).astype(np.int16)
    p = os.path.join(OUT, "jarvis_identity_tone.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return p, hash_sha(p)

def hash_sha(p, chunk=1 << 16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]

if __name__ == "__main__":
    icon = make_icon()
    brand = make_branding_double()
    chime, chime_h = synth_wake_chime()
    tone, tone_h = synthesize_identity_tone()

    manifest = f"""# Original Replacement Brand/Voice Assets — manifest
generated: deterministic (seeds 20260925 / 777)
code: generate_original_assets.py (this file) — owned, from scratch
third-party assets used: NONE (pure matplotlib + numpy)

## Files
- {os.path.basename(icon)}           (icon glyph, replaces cortana-shell/assets/icons/cortana.png)
- {os.path.basename(brand)}   (branding, replaces cortana-shell/assets/branding/cortanadouble.png)
- {os.path.basename(chime)} (wake chime, replaces cortana-shell/assets/wake-word-models/heycortana_*.table)  sha256[:16]={chime_h}
- {os.path.basename(tone)}      (identity tone, optional voice-identity asset)  sha256[:16]={tone_h}

## Provenance
Every asset above is original, generated by team code from team math:
- visual palette is the house FBSC neon set; glyph = braid-ring + seed node motif
- audio f0 = 41.02 Hz (owner-ratified resonance trigger); noise-gated envelope
  matches the approved ResonancelMonitor fix (no signal before gate opens)
- no samples, no TTS, no third-party images/models; compression-free WAV/PCM

## Recommended repo changes (PR)
- add these files under docs/artifacts/branding_voice/
- config.yaml: wakeWord.phrase -> "hey jarvis", modelPath -> new asset (see
  REPLACEMENT_MAP.md for wiring note: legacy .table loader expects binary table;
  wake detection re-wiring is an engineer task, chime is the replacement asset set)
"""
    with open(os.path.join(OUT, "MANIFEST.md"), "w") as f:
        f.write(manifest)
    print("WROTE:", manifest)