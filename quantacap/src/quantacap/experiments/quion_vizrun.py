"""Scenarios for generating Quion++ visual experiments."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.experiments.chsh_ybit import run_chsh_y
from quantacap.quion.gates import H5, PHASE5, RANDOM_UNITARY5, YG_BIAS, apply, invert
from quantacap.quion.material_time import MaterialClock
from quantacap.quion.state import EXPECTED_DIM, entropy_from_mags, fidelity, mags_phases, normalize
from quantacap.viz.quion_viz import animate_series, plot_quion_frame

DEFAULT_PREFIX = "artifacts/quion"


@dataclass
class FrameRecord:
    t: int
    psi: np.ndarray
    mags: np.ndarray
    phases: np.ndarray
    entropy: float
    fidelity: float
    frame_path: str
    summary: Dict[str, object] | None = None


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_state(rng: np.random.Generator, dtype: str) -> np.ndarray:
    raw = rng.normal(size=EXPECTED_DIM) + 1j * rng.normal(size=EXPECTED_DIM)
    return normalize(raw.astype(dtype))


def _random_gate(rng: np.random.Generator, dtype: str) -> np.ndarray:
    pick = rng.integers(0, 3)
    if pick == 0:
        angles = rng.normal(scale=0.2, size=EXPECTED_DIM)
        return PHASE5(angles, dtype=dtype)
    if pick == 1:
        weights = rng.normal(size=EXPECTED_DIM)
        return YG_BIAS(0.05, weights, dtype=dtype)
    seed = int(rng.integers(0, 2**32))
    return RANDOM_UNITARY5(seed=seed, dtype=dtype)


def _record_state(
    *,
    records: List[FrameRecord],
    psi: np.ndarray,
    t: int,
    psi0: np.ndarray,
    frame_dir: Path,
    scenario: str,
    history: Sequence[np.ndarray],
    meta: Dict[str, float] | None = None,
) -> None:
    mags, phases = mags_phases(psi)
    ent = entropy_from_mags(mags)
    fid = fidelity(psi0, psi)
    frame_path = frame_dir / f"frame_t{t:04d}.png"
    summary = plot_quion_frame(psi, frame_path, title=f"{scenario} t={t}", history=history, meta=meta)
    records.append(
        FrameRecord(
            t=t,
            psi=psi.copy(),
            mags=mags,
            phases=phases,
            entropy=ent,
            fidelity=fid,
            frame_path=str(frame_path),
            summary=summary,
        )
    )


def _serialise_series(
    scenario: str,
    seed: int,
    frames: Sequence[FrameRecord],
    metrics: Dict[str, float | int | Dict[str, float]],
    out_path: Path,
) -> Path:
    payload = {
        "scenario": scenario,
        "seed": int(seed),
        "frames": [
            {
                "t": rec.t,
                "mags": rec.mags.tolist(),
                "phases": rec.phases.tolist(),
                "fidelity": rec.fidelity,
                "S": rec.entropy,
                "frame": rec.frame_path,
                "summary": rec.summary,
            }
            for rec in frames
        ],
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def _video_path(prefix: str, prefer: str) -> Path:
    base = Path(prefix + "_timelapse")
    if prefer == "gif":
        return base.with_suffix(".gif")
    return base.with_suffix(".mp4")


def _series_path(prefix: str) -> Path:
    return Path(prefix + "_state_series.json")


def _frame_dir(prefix: str) -> Path:
    return Path(prefix + "_frames")


def run_quion_viz(
    *,
    scenario: str,
    steps: int,
    stride: int = 10,
    tau: float = 1e-4,
    jitter: float = 1e-3,
    delta: float = 0.03,
    lam: float = 0.85,
    eps: float = 0.02,
    dtype: str = "complex128",
    seed: int = 424242,
    seed_id: str = "demo.ybit",
    out_prefix: str = DEFAULT_PREFIX,
    video_format: str = "auto",
) -> Dict[str, object]:
    rng = _rng(seed)
    psi0 = _random_state(rng, dtype)
    psi = psi0.copy()
    frames: List[FrameRecord] = []
    history: List[np.ndarray] = [psi.copy()]

    prefix = out_prefix
    frame_dir = _frame_dir(prefix)
    frame_dir.mkdir(parents=True, exist_ok=True)
    records_logged = 0
    committed_count = 0

    def capture(t: int, meta: Dict[str, float] | None = None, commit: bool = True) -> None:
        nonlocal records_logged, committed_count
        _record_state(
            records=frames,
            psi=psi,
            t=t,
            psi0=psi0,
            frame_dir=frame_dir,
            scenario=scenario,
            history=history,
            meta=meta,
        )
        records_logged += 1
        if commit:
            committed_count += 1

    capture(0, commit=True)

    if scenario == "reverse":
        gates: List[np.ndarray] = []
        for step in range(1, steps + 1):
            gate = _random_gate(rng, dtype)
            gates.append(gate)
            psi = apply(psi, gate)
            history.append(psi.copy())
            if step % stride == 0 or step == steps:
                capture(step, commit=True)
        for idx, gate in enumerate(reversed(gates), 1):
            psi = apply(psi, invert(gate))
            history.append(psi.copy())
            t = steps + idx
            if idx % stride == 0 or idx == len(gates):
                capture(t, commit=True)
        metrics = {
            "F_final": fidelity(psi0, psi),
            "frames_committed": committed_count,
            "total_steps": steps,
        }
    elif scenario == "freeze":
        clock = MaterialClock(tau=tau, guard=False)
        for step in range(1, steps + 1):
            psi = apply(psi, _random_gate(rng, dtype))
            history.append(psi.copy())
            tick = clock.observe(psi)
            if tick.commit:
                capture(step, meta={"entropy": tick.entropy}, commit=True)
            elif step % stride == 0 or step == steps:
                capture(step, meta={"entropy": tick.entropy}, commit=False)
        metrics = {
            "frames_committed": committed_count,
            "total_steps": steps,
            "tau": tau,
        }
    elif scenario == "noise":
        clock = MaterialClock(tau=0.0, guard=True)
        coherence_open = []
        coherence_guard = []
        psi_guard = psi.copy()
        for step in range(1, steps + 1):
            jitter_vec = rng.normal(scale=jitter, size=EXPECTED_DIM)
            noise = np.exp(1j * jitter_vec)
            psi = normalize(psi * noise, dtype=dtype)
            candidate = normalize(psi_guard * noise, dtype=dtype)
            tick = clock.observe(candidate)
            if tick.commit:
                psi_guard = candidate
            history.append(psi_guard.copy())
            coherence_open.append(fidelity(psi0, psi))
            coherence_guard.append(fidelity(psi0, psi_guard))
            if tick.commit or step % stride == 0 or step == steps:
                capture(step, meta={"entropy": tick.entropy}, commit=tick.commit)
        metrics = {
            "coherence_open_final": coherence_open[-1] if coherence_open else 1.0,
            "coherence_guard_final": coherence_guard[-1] if coherence_guard else 1.0,
            "violations": clock.violations,
            "frames_committed": committed_count,
            "jitter": jitter,
        }
    elif scenario == "chshy":
        result = run_chsh_y(
            shots=max(20000, steps * 50),
            depol=0.0,
            seed=seed,
            seed_id=seed_id,
            lam=lam,
            eps=eps,
            delta=delta,
            adapter_id=None,
        )
        mod = result["mod"]
        angles = result["angles"]
        S_value = float(result["S"])
        mags_base = np.array(
            [
                0.18 + 0.32 * ((angles["A"] % (2 * math.pi)) / (2 * math.pi)),
                0.18 + 0.32 * ((angles["B"] % (2 * math.pi)) / (2 * math.pi)),
                0.16 + 0.28 * mod["etaA"],
                0.16 + 0.28 * mod["etaB"],
                0.32 + 0.2 * (S_value / 3.0),
            ]
        )
        mags_base = mags_base / np.sum(mags_base)
        phases_base = np.array(
            [
                mod["phiA"],
                mod["phiB"],
                2 * math.pi * mod["etaA"],
                2 * math.pi * mod["etaB"],
                S_value % (2 * math.pi),
            ]
        )
        psi = normalize(np.sqrt(mags_base) * np.exp(1j * phases_base), dtype=dtype)
        history[-1] = psi.copy()
        capture(0, meta={"S": S_value})
        for step in range(1, steps + 1):
            blend = step / steps
            phase_shift = delta * (blend - 0.5)
            phase_vec = phases_base + phase_shift * np.array([mod["etaA"], mod["etaB"], lam, eps, delta])
            gate = PHASE5(phase_vec - phases_base, dtype=dtype) @ H5(dtype=dtype)
            psi = apply(psi, gate)
            history.append(psi.copy())
            if step % stride == 0 or step == steps:
                capture(step, meta={"blend": blend, "phase_shift": phase_shift}, commit=True)
        metrics = {
            "S": S_value,
            "delta": delta,
            "lam": lam,
            "eps": eps,
            "frames_committed": committed_count,
        }
        create_adapter(
            f"quion.chshy.{seed_id}",
            data={"result": result, "metrics": metrics},
            meta={"kind": "quion-chshy"},
        )
    else:
        raise ValueError(f"unknown scenario '{scenario}'")

    series_path = _series_path(prefix)
    metrics_with_frames = {**metrics, "frames_rendered": records_logged}
    series_file = _serialise_series(scenario, seed, frames, metrics_with_frames, series_path)

    target = _video_path(prefix, "gif" if video_format == "gif" else "mp4")
    prefer_writer = "PillowWriter" if video_format == "gif" else ("FFMpegWriter" if video_format == "mp4" else "auto")
    states_for_animation = [rec.psi for rec in frames]
    saved_path = animate_series(states_for_animation, target, prefer=prefer_writer)

    summary = {
        "series": str(series_file),
        "video": str(saved_path),
        "frames_dir": str(frame_dir),
        "metrics": metrics_with_frames,
    }
    return summary
