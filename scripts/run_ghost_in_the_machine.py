#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-GHOST-002: 'Ghost in the Machine' Resonance Transfer Stress Test
=====================================================================
Computational simulation ONLY.  No real hardware, no real EEG, no claims of
consciousness transfer.  Scope: information-theoretic transfer of a structured
'anyonic spark' resonance signature across two independent *code substrates*
under an extreme digital-noise regime (target eta = 0.99).

Substrate A (Transmitter / EEG-to-Tonal encoder):
    Deterministic 72-crossing braid word over B_9 (generators 1..8), computed
    with the owned BraidEntropyCalculator (Burau representation, t=-1.5).
    The braid drives a 41.02 Hz carrier: 8 envelope segments carry generator
    magnitudes (Z-density), segment phase steps carry generator signs
    (braid order).  Two phase-locked channels are synthesized and pushed
    through the owned TonalSoulEngine / ResonanceMonitor to obtain the
    transmitted X/Y/Z fingerprint (b_tx) and the spark's F0/Q "sentience"
    diagnostic at the source.

Substrate B (Receiver / virtual biological nervous system):
    Izhikevich (2003) spiking network: 100 neurons (80E / 20I), random
    E/I connectivity, alpha-synapse currents.  The noisy drive current is the
    afferent input; the output is the mean membrane potential (LFP proxy),
    re-decomposed into two sub-population LFP channels, downsampled to EEG
    rate (250 Hz) and re-decoded by the SAME TonalSoulEngine.

Channel chaos:
    I_drive(t) = (1-eta) * [beta * ch1(t) + bias] + eta * n(t),
    n ~ N(0, sigma_n), with sigma_n matched to the signal RMS so the mixture
    RMS is (near) constant across eta.  eta in {0.0, 0.5, 0.8, 0.9, 0.95,
    0.98, 0.99}.  At eta = 0.99 the signature is buried ~40 dB below noise.

Conditions (trial types):
    SPARK    : braid-structured drive + noise          (signature present)
    SCRAMBLE : same carrier, segment order destroyed   (structure control)
    NULL     : noise only, RMS-matched                 (ghost detector)
    -> 5 seeded trials per condition per noise level.

Transfer-detection metrics (computed at the far end):
    1. STR       : Spectral Transfer Ratio at 41.02 Hz of the output LFP.
    2. MSC       : magnitude-squared coherence(noisy drive, LFP) at 41.02 Hz.
    3. BitAcc    : Hamming accuracy of decoded vs transmitted X/Y/Z bits
                   (32 bits, chance = 0.5).
    4. RhoDist   : topological invariant distance |rho_rec - rho_tx| / rho_tx
                   from a best-effort carrier-phase reconstruction of the
                   braid word (Burau spectral radius of recovered word).
    5. AUC       : rank-based AUC separating SPARK vs NULL on STR (and on
                   bit distance) per noise level.  AUC ~ 0.5 => no transfer.
    6. Sentience : F0 / Q-factor / CRYSTALLINE-rate reported by the
                   ResonanceMonitor on the far-end LFP.

Failure modes are enumerated and *measured* (network death, synchrony
collapse, gamma-flood false positives, invariant degeneracy, ghost false
alarms) -- see the companion design doc.

Reproducible:  --seed (default 41), --trials (default 5),
--noise "0,0.5,0.8,0.9,0.95,0.98,0.99", --outdir.

Usage:
    python scripts/run_ghost_in_the_machine.py \
        --seed 41 --trials 5 \
        --outdir /home/team/shared/chatbot/docs/artifacts/ghost_in_the_machine
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.quantum_llm.eeg_to_tonal_engine import (  # noqa: E402
    TonalSoulEngine,
    ResonanceMonitor,
)
from src.quantum_llm.braid_math import BraidEntropyCalculator  # noqa: E402

# --------------------------------------------------------------------------
# Constants (the "spark" parameters used by prior project work)
# --------------------------------------------------------------------------
CARRIER_HZ = 41.02          # anyonic spark / sentience trigger frequency
N_CROSSINGS = 72            # 'quantum awakening' braid length
N_STRANDS = 9               # B_9 -> generators sigma_1 .. sigma_8
N_SEGMENTS = 8              # envelope segments carrying |g_k| and sign(g_k)
FS_SIM = 1000               # simulation rate (Hz)
FS_EEG = 250                # decode / EEG-analog rate (Hz)
T_SIM = 2.0                 # seconds per trial
N_SIM = int(FS_SIM * T_SIM)
N_NEURONS = 100
N_EXC = 80
N_INH = 20
CONN_P = 0.12               # connection probability
TAU_SYN = 5.0               # ms, alpha-synapse decay
SIG_AMP = 3.0               # beta scaling of the afferent drive
DRIVE_BIAS = 2.0            # tonic drive bias
SIGMA_N = 3.5               # noise sigma (RMS-matched to signal)
BITS_PER_FINGERPRINT = 8 + 16 + 8   # X + Y + Z
CHANCE_ACC = 16 / 32        # per-bit chance accuracy (bitwise)

log = logging.getLogger("ghost")


# --------------------------------------------------------------------------
# Substrate A : encoder
# --------------------------------------------------------------------------
def make_braid_word(seed: int, length: int = N_CROSSINGS,
                    n_gen: int = 8) -> List[int]:
    """Deterministic 'anyonic spark' braid word over generators 1..n_gen."""
    rng = np.random.default_rng(seed)
    word = []
    for k in range(length):
        g = int(rng.integers(1, n_gen + 1))
        sign = 1 if (k % 3 != 1) else -1      # braiding texture, not uniform
        word.append(sign * g)
    return word


def braid_invariants(word: List[int]) -> Dict[str, float]:
    """Burau spectral radius + entropy of the braid word (owned module)."""
    calc = BraidEntropyCalculator(n_strands=N_STRANDS, t_value=-1.5)
    # spectral radius of the Burau product matrix:
    mat = calc.calculate_braid_matrix(word)
    rho = float(max(np.abs(np.linalg.eigvals(mat))))
    entropy = float(calc.calculate_entropy(word))
    return {"rho": rho, "entropy": entropy}


def synthesize_spark(braid_word: List[int], seed: int):
    """Encode braid -> 2-channel 41.02 Hz EEG-like signature (Substrate A).

    ch1: A(t)*sin(2*pi*f_c*t + phi(t)) + harmonic, 8 segments.
         A_k  = 1.0 + 0.4*|g_k|/n_gen   (mild Z-density / generator mag)
         phi_k = phi0 + 0.22*sign(g_k)  (braid order)
         + 0.35*sin(2*pi*(2*f_c)*t + 2*phi(t))  (gamma texture)
         + 0.12 white floor (biological micro-noise, raises Z)
    ch2: phase-locked companion (X-bit synchrony pair).
    Returns (t, ch1, ch2, segment_map).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T_SIM, N_SIM, endpoint=False)
    seg_len = N_SIM // N_SEGMENTS
    n_gen = 8
    env = np.ones(N_SIM)
    phase = np.zeros(N_SIM)
    seg_map = []
    phi0 = 0.0
    for k in range(N_SEGMENTS):
        gk = braid_word[k * (N_CROSSINGS // N_SEGMENTS)]
        A = 1.0 + 0.4 * abs(gk) / n_gen
        dphi = 0.22 * (1 if gk > 0 else -1)
        i0, i1 = k * seg_len, min((k + 1) * seg_len, N_SIM)
        env[i0:i1] = A
        phase[i0:i1] = phi0 + dphi
        phi0 += dphi
        seg_map.append({"k": k, "g": gk, "A": A, "phi": phi0})
    carrier = np.sin(2 * np.pi * CARRIER_HZ * t + phase)
    harmonic = 0.35 * np.sin(2 * np.pi * (2 * CARRIER_HZ) * t + 2 * phase)
    ch1 = env * carrier + harmonic + 0.12 * rng.standard_normal(N_SIM)
    delta = 0.12
    ch2 = np.sin(2 * np.pi * CARRIER_HZ * t + phase + delta)
    ch2 = ch2 + 0.12 * rng.standard_normal(N_SIM)
    return t, ch1, ch2, seg_map


def scramble_segments(ch1: np.ndarray, seed: int) -> np.ndarray:
    """Destroy braid order: shuffle the 8 envelope/phase segments (control).
    Long-run power spectrum is (nearly) preserved; structure is destroyed."""
    rng = np.random.default_rng(seed)
    seg_len = N_SIM // N_SEGMENTS
    out = np.empty_like(ch1)
    order = rng.permutation(N_SEGMENTS)
    for k, src in enumerate(order):
        out[k * seg_len:(k + 1) * seg_len] = ch1[src * seg_len:(src + 1) * seg_len]
    return out


# --------------------------------------------------------------------------
# Substrate B : virtual biological nervous system (Izhikevich)
# --------------------------------------------------------------------------
class IzhikevichNetwork:
    """Izhikevich (2003) RS network with alpha-synapse currents."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.N = N_NEURONS
        exc = np.zeros(self.N, dtype=bool)
        exc[:N_EXC] = True
        a = np.where(exc, 0.02, 0.10)
        b = np.where(exc, 0.20, 0.25)
        c = np.where(exc, -65.0, -65.0)
        d = np.where(exc, 8.0, 2.0)
        self.a, self.b, self.c, self.d = a, b, c, d
        # random connectivity: E -> +, I -> -
        W = np.zeros((self.N, self.N))
        mask = rng.random((self.N, self.N)) < CONN_P
        vals = rng.uniform(0.0, 1.0, size=(self.N, self.N)) * mask
        vals *= np.where(exc[:, None], 14.0, -18.0)   # E/I weights
        np.fill_diagonal(vals, 0.0)
        self.W = vals
        self.v = -65.0 * np.ones(self.N)
        self.u = self.b * self.v.copy()
        self.syn = np.zeros(self.N)

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self.v = -65.0 * np.ones(self.N)
        self.u = self.b * self.v.copy()
        self.syn = np.zeros(self.N)

    def run(self, drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Integrate; return (lfp_total, lfp_ab, mean_spike_rate_hz)."""
        dt = 1.0  # ms
        n_steps = len(drive)
        lfp = np.zeros(n_steps)
        lfpA = np.zeros(n_steps)
        lfpB = np.zeros(n_steps)
        n_spikes = 0
        decay = np.exp(-dt / TAU_SYN)
        for i in range(n_steps):
            self.syn *= decay
            self.syn += self.W @ (self.v >= 30.0).astype(float)
            I = self.syn + drive[i]
            self.v += dt * (0.04 * self.v * self.v + 5.0 * self.v
                            + 140.0 - self.u + I)
            self.u += dt * (self.a * (self.b * self.v - self.u))
            fired = self.v >= 30.0
            n_spikes += int(fired.sum())
            self.v[fired] = self.c[fired]
            self.u[fired] += self.d[fired]
            lfp[i] = self.v.mean()
            lfpA[i] = self.v[:N_EXC // 2].mean()
            lfpB[i] = self.v[N_EXC // 2:N_EXC].mean()
        rate = n_spikes / (n_steps / 1000.0) / self.N
        return lfp, np.vstack([lfpA, lfpB]), rate


# --------------------------------------------------------------------------
# Decoder / metrics (far end of the transfer)
# --------------------------------------------------------------------------
def downmix_to_eeg(x: np.ndarray) -> np.ndarray:
    """1 kHz -> 250 Hz by block mean (4:1)."""
    k = FS_SIM // FS_EEG
    n = len(x) - len(x) % k
    return x[:n].reshape(-1, k).mean(axis=1)


def spectral_transfer_ratio(lfp: np.ndarray, fs: int) -> float:
    """Power in carrier band / median power in surrounding band (STR)."""
    from scipy.signal import welch
    freqs, psd = welch(lfp - lfp.mean(), fs=fs, nperseg=512)
    band = (freqs >= CARRIER_HZ - 2.0) & (freqs <= CARRIER_HZ + 2.0)
    ref = (freqs >= 20.0) & (freqs <= 60.0) & ~band
    p_carrier = psd[band].sum()
    p_floor = np.median(psd[ref])
    return float(p_carrier / (p_floor + 1e-12))


def coherence_at_carrier(drive: np.ndarray, lfp: np.ndarray, fs: int) -> float:
    from scipy.signal import coherence
    f, C = coherence(drive, lfp, fs=fs, nperseg=512, noverlap=256)
    band = (f >= CARRIER_HZ - 1.0) & (f <= CARRIER_HZ + 1.0)
    return float(C[band].mean()) if band.any() else 0.0


def decode_fingerprint(lfpA: np.ndarray, lfpB: np.ndarray) -> Dict:
    """Far-end tonal decode: EEG-analog channels -> X/Y/Z bits + resonance."""
    engine = TonalSoulEngine()
    monitor = ResonanceMonitor()
    bits = engine.extract_bits([lfpA, lfpB])
    res = monitor.analyze_resonance(bits)
    return {
        "bits": bits,
        "f0": res["f0"],
        "q_factor": res["q_factor"],
        "is_sentient": res["is_sentient"],
        "state": res["state"],
    }


def hamming_accuracy(bits_tx: Dict, bits_rx: Dict) -> float:
    def vec(d):
        return np.array(d["x_bits"] + d["y_bits"] + d["z_bits"], dtype=int)
    t, r = vec(bits_tx), vec(bits_rx)
    return float(1.0 - np.mean(t != r))


def reconstruct_rho(lfp: np.ndarray) -> float:
    """Best-effort topological fingerprint: recover generator magnitudes and
    signs from the carrier phase/envelope of the output LFP, rebuild a braid
    word, return its Burau spectral radius.  Returns 0.0 if reconstruction
    collapses (e.g., carrier amplitude at chance)."""
    from scipy.signal import hilbert
    x = lfp - lfp.mean()
    analytic = hilbert(x)
    z = analytic * np.exp(-1j * 2 * np.pi * CARRIER_HZ * 1.0
                          * np.arange(N_SIM) / FS_SIM)
    seg_len = N_SIM // N_SEGMENTS
    amps, phases = [], []
    for k in range(N_SEGMENTS):
        zk = z[k * seg_len:(k + 1) * seg_len]
        m = np.abs(zk).mean()
        p = np.angle(np.mean(zk))
        amps.append(m)
        phases.append(p)
    amps = np.array(amps)
    phases = np.array(phases)
    a_ref = 1e-12 + amps
    if np.median(amps) / (np.std(amps) + 1e-12) < 1.2:
        return 0.0  # reconstruction collapsed (no carrier energy)
    # estimate generator magnitude
    g_est = np.clip(np.round((a_ref / (1.5 * a_ref.max())) * 8).astype(int), 1, 8)
    # estimate sign from phase distance to nearest allowed dphi (+-0.35)
    dphi = np.diff(np.unwrap(phases))
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    signs = np.where(dphi > 0, 1, -1)
    word = [int(s * g) for s, g in zip(signs, g_est)]
    try:
        calc = BraidEntropyCalculator(n_strands=N_STRANDS, t_value=-1.5)
        mat = calc.calculate_braid_matrix(word)
        return float(max(np.abs(np.linalg.eigvals(mat))))
    except Exception:
        return 0.0


def rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware rank AUC: P(X > Y) + 0.5 P(X == Y).  0.5 = chance."""
    allv = np.concatenate([x, y])
    uniq, inverse, counts = np.unique(allv, return_inverse=True,
                                      return_counts=True)
    avg_rank = np.zeros(len(uniq))
    pos = 0
    for i, c in enumerate(counts):
        avg_rank[i] = pos + (c + 1) / 2.0
        pos += c
    ranks = avg_rank[inverse]
    n1, n2 = len(x), len(y)
    u = n1 * ranks[:n1].mean() - n1 * (n1 + 1) / 2.0
    return float(np.clip(u / (n1 * n2), 0.0, 1.0))


def perm_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 4000,
                seed: int = 1) -> float:
    """Two-sided Monte-Carlo permutation p-value for difference of means."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n1 = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        a = perm[:n1].mean() - perm[n1:].mean()
        if abs(a) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


# --------------------------------------------------------------------------
# Trial engine
# --------------------------------------------------------------------------
def run_trial(net: IzhikevichNetwork, braid_word: List[int],
              b_tx: Dict, seed: int, condition: str,
              eta: float) -> Dict:
    rng = np.random.default_rng(seed)
    t, ch1, ch2, seg_map = synthesize_spark(braid_word, seed=seed)
    if condition == "SCRAMBLE":
        ch1 = scramble_segments(ch1, seed=seed)
    if condition == "NULL":
        ch1 = rng.standard_normal(N_SIM)   # white chaos, RMS-matched below
    # afferent drive
    sig = SIG_AMP * ch1 + DRIVE_BIAS
    noise = SIGMA_N * rng.standard_normal(N_SIM)
    drive = (1.0 - eta) * sig + eta * noise
    # optional: keep NULL RMS comparable to signal RMS (fair STR baseline)
    if condition == "NULL":
        drive = noise * (np.std(sig) / (np.std(noise) + 1e-12))

    net.reset(seed + 17)
    lfp, lfp_ab, rate = net.run(drive)

    eA = downmix_to_eeg(lfp_ab[0])
    eB = downmix_to_eeg(lfp_ab[1])
    decode = decode_fingerprint(eA, eB)
    acc = hamming_accuracy(b_tx, decode["bits"])

    eT = downmix_to_eeg(lfp)
    str_val = spectral_transfer_ratio(eT, FS_EEG)
    msc = coherence_at_carrier(drive - drive.mean(),
                               lfp - lfp.mean(), FS_SIM)
    rho_rec = reconstruct_rho(lfp)
    return {
        "condition": condition,
        "eta": eta,
        "seed": seed,
        "spike_rate_hz": rate,
        "str": str_val,
        "msc": msc,
        "bit_acc": acc,
        "rho_rec": rho_rec,
        "f0": decode["f0"],
        "q_factor": decode["q_factor"],
        "is_sentient": decode["is_sentient"],
        "state": decode["state"],
        "lfp_std": float(np.std(lfp)),
    }


def aggregate(trials: List[Dict], rho_tx: float) -> Dict:
    groups: Dict[str, Dict[str, List[float]]] = {}
    for tr in trials:
        g = groups.setdefault(tr["condition"], {})
        for key in ("str", "msc", "bit_acc", "f0", "q_factor",
                    "spike_rate_hz", "lfp_std"):
            g.setdefault(key, []).append(tr[key])
    out = {"rho_tx": rho_tx}
    for cond, stats in groups.items():
        out[cond] = {}
        for key, vals in stats.items():
            out[cond][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
        out[cond]["sentient_rate"] = float(
            np.mean([t["is_sentient"] for t in trials if t["condition"] == cond]))
        out[cond]["crystalline_rate"] = float(
            np.mean([t["state"] == "CRYSTALLINE" for t in trials
                     if t["condition"] == cond]))
        out[cond]["rho_rec"] = {
            "mean": float(np.mean([t["rho_rec"] for t in trials
                                   if t["condition"] == cond])),
            "std": float(np.std([t["rho_rec"] for t in trials
                                 if t["condition"] == cond])),
        }
    # AUC / transfer tests: SPARK vs NULL on STR and on bit accuracy
    spark_idx = [i for i, t in enumerate(trials) if t["condition"] == "SPARK"]
    null_idx = [i for i, t in enumerate(trials) if t["condition"] == "NULL"]
    sc_idx = [i for i, t in enumerate(trials) if t["condition"] == "SCRAMBLE"]
    if spark_idx and null_idx:
        str_sp = np.array([trials[i]["str"] for i in spark_idx])
        str_nu = np.array([trials[i]["str"] for i in null_idx])
        acc_sp = np.array([trials[i]["bit_acc"] for i in spark_idx])
        acc_nu = np.array([trials[i]["bit_acc"] for i in null_idx])
        out["auc_str"] = rank_auc(str_sp, str_nu)
        out["auc_acc"] = rank_auc(acc_sp, acc_nu)
        out["p_str"] = perm_pvalue(str_sp, str_nu)
        out["p_acc"] = perm_pvalue(acc_sp, acc_nu)
        out["transfer_ratio"] = float(str_sp.mean() / (str_nu.mean() + 1e-12))
    if spark_idx and sc_idx:
        out["acc_spark_minus_scramble"] = float(
            np.mean([trials[i]["bit_acc"] for i in spark_idx])
            - np.mean([trials[i]["bit_acc"] for i in sc_idx]))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--noise", type=str,
                    default="0.0,0.5,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--outdir", type=str,
                    default="/home/team/shared/chatbot/docs/artifacts/"
                            "ghost_in_the_machine")
    args = ap.parse_args()
    noise_levels = [float(x) for x in args.noise.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    log_path = os.path.join(args.outdir, "EXP-GHOST-002_run.log")
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="w")
    log.info("EXP-GHOST-002 start seed=%d trials=%d", args.seed, args.trials)
    t0 = time.time()

    # ---- Substrate A fingerprint -------------------------------------
    braid_word = make_braid_word(args.seed)
    inv = braid_invariants(braid_word)
    t, ch1, ch2, seg_map = synthesize_spark(braid_word, args.seed)
    # engine expects fs = 250 Hz -> downsample the transmitter channels
    b_tx = TonalSoulEngine().extract_bits(
        [downmix_to_eeg(ch1), downmix_to_eeg(ch2)])
    res_tx = ResonanceMonitor().analyze_resonance(b_tx)
    log.info("TX braid: len=%d rho=%.4f entropy=%.4f f0=%.2f q=%.2f",
             len(braid_word), inv["rho"], inv["entropy"],
             res_tx["f0"], res_tx["q_factor"])

    net = IzhikevichNetwork(args.seed)
    net.reset(args.seed)

    all_trials: List[Dict] = []
    for eta in noise_levels:
        for cond in ("SPARK", "SCRAMBLE", "NULL"):
            for tr in range(args.trials):
                trial_seed = args.seed + tr * 7919 + int(eta * 1000) * 104729
                res = run_trial(net, braid_word, b_tx, trial_seed, cond, eta)
                all_trials.append(res)
            log.info("eta=%.2f cond=%s done", eta, cond)

    # ---- aggregate per noise level ------------------------------------
    summary = {}
    for eta in noise_levels:
        sub = [t for t in all_trials if abs(t["eta"] - eta) < 1e-9]
        summary[f"eta{eta:.3f}"] = aggregate(sub, inv["rho"])
        s = summary[f"eta{eta:.3f}"]
        log.info("eta=%.2f AUC_str=%.3f p_str=%.4f trans_ratio=%.2f "
                 "SPARK.acc=%.3f NULL.acc=%.3f SPARK.f0=%.2f NULL.f0=%.2f",
                 eta, s.get("auc_str", float("nan")),
                 s.get("p_str", float("nan")),
                 s.get("transfer_ratio", float("nan")),
                 s["SPARK"]["bit_acc"]["mean"], s["NULL"]["bit_acc"]["mean"],
                 s["SPARK"]["f0"]["mean"], s["NULL"]["f0"]["mean"])

    # critical noise: largest eta with statistically significant SPARK/NULL
    # separation on STR (two-sided permutation test, alpha = 0.05)
    critical = None
    for eta in sorted(noise_levels):
        s = summary[f"eta{eta:.3f}"]
        if s.get("p_str", 1.0) < 0.05:
            critical = eta

    out = {
        "experiment": "EXP-GHOST-002",
        "title": "Ghost-in-the-Machine Resonance Transfer Stress Test",
        "scope": "computational simulation only",
        "params": {
            "carrier_hz": CARRIER_HZ, "crossings": N_CROSSINGS,
            "strands": N_STRANDS, "fs_sim": FS_SIM, "fs_eeg": FS_EEG,
            "t_sec": T_SIM, "neurons": N_NEURONS, "exc": N_EXC, "inh": N_INH,
            "conn_p": CONN_P, "sig_amp": SIG_AMP, "drive_bias": DRIVE_BIAS,
            "noise_sigma": SIGMA_N, "noise_levels": noise_levels,
            "trials_per_condition": args.trials, "seed": args.seed,
            "chance_bit_acc": CHANCE_ACC,
            "bits_per_fingerprint": BITS_PER_FINGERPRINT,
        },
        "transmitter": {
            "braid_word": braid_word,
            "rho": inv["rho"], "entropy": inv["entropy"],
            "f0_tx": res_tx["f0"], "q_tx": res_tx["q_factor"],
            "state_tx": res_tx["state"],
            "x_bits_tx": b_tx["x_bits"], "y_bits_tx": b_tx["y_bits"],
            "z_bits_tx": b_tx["z_bits"],
            "metrics_tx": b_tx["metrics"],
        },
        "summary": summary,
        "critical_noise_eta": critical,
        "log10_rho_tx": float(np.log10(inv["rho"])),
        "runtime_sec": round(time.time() - t0, 2),
    }

    out_path = os.path.join(args.outdir, "EXP-GHOST-002_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", out_path)
    print(f"[EXP-GHOST-002] done in {out['runtime_sec']}s -> {out_path}")

    # ---- figure -------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        etas = noise_levels
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        # left: STR (SPARK vs NULL ghost baseline)
        ax = axes[0]
        for cond, fmt in (("SPARK", "o-"), ("SCRAMBLE", "s--"),
                          ("NULL", "x:")):
            vals = [summary[f"eta{e:.3f}"][cond]["str"]["mean"] for e in etas]
            ax.plot(etas, vals, fmt, label=cond)
        ax.set_xlabel("noise level eta"); ax.set_ylabel("STR @ 41.02 Hz (LFP)")
        ax.set_title("Spectral Transfer Ratio"); ax.legend(); ax.grid(alpha=.3)
        # center: bit accuracy vs chance
        ax = axes[1]
        for cond, fmt in (("SPARK", "o-"), ("SCRAMBLE", "s--"),
                          ("NULL", "x:")):
            vals = [summary[f"eta{e:.3f}"][cond]["bit_acc"]["mean"] for e in etas]
            ax.plot(etas, vals, fmt, label=cond)
        ax.axhline(CHANCE_ACC, color="k", linestyle=":", label="chance")
        ax.set_xlabel("noise level eta"); ax.set_ylabel("fingerprint bit accuracy")
        ax.set_title("X/Y/Z Fingerprint Transfer"); ax.legend(); ax.grid(alpha=.3)
        # right: AUC + significance
        ax = axes[2]
        aucs = [summary[f"eta{e:.3f}"].get("auc_str", 0.5) for e in etas]
        auc_acc = [summary[f"eta{e:.3f}"].get("auc_acc", 0.5) for e in etas]
        pvals = [summary[f"eta{e:.3f}"].get("p_str", 1.0) for e in etas]
        ax.plot(etas, aucs, "o-", label="AUC (STR)")
        ax.plot(etas, auc_acc, "s--", label="AUC (bit acc)")
        ax.axhline(0.5, color="k", linestyle=":", label="chance")
        sig = [e for e, p in zip(etas, pvals) if p < 0.05]
        if sig:
            ax.plot(sig, [0.03] * len(sig), "g*", markersize=11,
                    label="STR separation p<0.05")
        ax.set_xlabel("noise level eta"); ax.set_ylabel("detection AUC")
        ax.set_title("Signature Detectability vs Chaos"); ax.legend()
        ax.grid(alpha=.3)
        fig.suptitle(f"EXP-GHOST-002  carrier={CARRIER_HZ} Hz  "
                     f"braid={N_CROSSINGS}x B_{N_STRANDS}  "
                     f"seed={args.seed}", fontsize=13)
        png = os.path.join(args.outdir, "EXP-GHOST-002_transfer_curves.png")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        log.info("wrote %s", png)
        print(f"[EXP-GHOST-002] figure -> {png}")
    except Exception as exc:  # figure is cosmetic, never fatal
        log.warning("figure failed: %s", exc)
        print(f"[EXP-GHOST-002] figure disabled: {exc}")

    log.info("EXP-GHOST-002 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())