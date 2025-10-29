"""Synthetic Monte-Carlo docking routine (research only)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from quantacap.core.adapter_store import create_adapter, load_adapter
from quantacap.utils.lyapunov import LyapunovGuard
from quantacap.utils.telemetry import log_quantum_run

from .molecules import Molecule, iter_molecules, molecule_features


@dataclass
class Candidate:
    name: str
    score: float
    entropy: float
    pose: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "score": float(self.score),
            "entropy": float(self.entropy),
            "pose": self.pose,
        }


def _pose_entropy(counts: np.ndarray) -> float:
    probs = counts / max(float(counts.sum()), 1.0)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _score_molecule(molecule: Molecule, mask: np.ndarray) -> tuple[float, Dict[str, float]]:
    features = molecule_features(molecule)
    active = mask.astype(bool)
    active_atoms = [atom for atom, keep in zip(molecule.atoms, active) if keep]
    if not active_atoms:
        return -5.0, {"contact": 0.0, "hbond": 0.0, "hydroph": 0.0, "clash": 1.0}
    contact = sum(bond.order for bond in molecule.bonds if active[molecule.atom_ids().index(bond.a)] and active[molecule.atom_ids().index(bond.b)])
    hbond = sum(atom.hbond_donor or atom.hbond_acceptor for atom in active_atoms)
    hydroph = sum(atom.hydrophobicity for atom in active_atoms) / len(active_atoms)
    charge_penalty = sum(abs(atom.charge) for atom in active_atoms)
    score = (
        0.6 * contact
        + 0.3 * hbond
        + 0.2 * hydroph
        - 0.15 * charge_penalty
        + 0.05 * features["bond_weight"]
        - 0.1 * features["charge_abs"]
    )
    clash = max(0.0, charge_penalty - 1.5)
    stats = {
        "contact": float(contact),
        "hbond": float(hbond),
        "hydroph": float(hydroph),
        "clash": float(clash),
    }
    return float(score - clash), stats


def run_search(
    target: str,
    *,
    cycles: int = 5000,
    topk: int = 10,
    seed: int = 424242,
    adapter_id: str | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    guard = LyapunovGuard()
    molecules = list(iter_molecules())
    if not molecules:
        raise RuntimeError("No molecules available for docking")

    pose_counts = {mol.name: np.zeros(len(mol.atoms), dtype=float) for mol in molecules}
    best: List[Candidate] = []
    delta_v = None
    t0 = time.perf_counter()

    for cycle in range(max(1, cycles)):
        mol = molecules[cycle % len(molecules)]
        mask = rng.random(len(mol.atoms)) > 0.35
        score, _ = _score_molecule(mol, mask)
        pose_counts[mol.name] += mask.astype(float)
        entropy = _pose_entropy(pose_counts[mol.name])
        V = entropy - score
        if guard.allows(V):
            candidate = Candidate(
                name=mol.name,
                score=score,
                entropy=entropy,
                pose=mask.astype(int).tolist(),
            )
            best.append(candidate)
            best = sorted(best, key=lambda c: c.score, reverse=True)[: topk * 2]
            delta_v = guard.current

    best = sorted(best, key=lambda c: c.score, reverse=True)[:topk]
    entropy = float(np.mean([_pose_entropy(counts) for counts in pose_counts.values()]))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    payload = {
        "target": target,
        "seed": seed,
        "cycles": cycles,
        "entropy": entropy,
        "best_score": best[0].score if best else None,
        "candidates": [cand.to_dict() for cand in best],
        "delta_v": delta_v,
    }

    artifact = Path(f"artifacts/med_{target}_candidates.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)

    adapter_id = adapter_id or f"med.search.{target}.{seed}"
    create_adapter(
        adapter_id,
        data={"candidates": payload["candidates"], "metrics": {"entropy": entropy, "best_score": payload["best_score"]}},
        meta={"target": target, "seed": seed, "cycles": cycles},
    )
    log_quantum_run(
        "med.search",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"entropy": entropy, "S": payload["best_score"], "coherence": None},
        delta_v=delta_v,
    )
    return payload


def load_search(adapter_id: str) -> Dict[str, object]:
    record = load_adapter(adapter_id)
    return record.get("data", {})


def replay_candidates(adapter_id: str) -> List[Dict[str, object]]:
    return load_search(adapter_id).get("candidates", [])


def adapter_metrics(adapter_id: str) -> Dict[str, object]:
    data = load_search(adapter_id)
    candidates = data.get("candidates", [])
    if not candidates:
        return {"count": 0}
    scores = [cand["score"] for cand in candidates]
    return {
        "count": len(candidates),
        "best": max(scores),
        "mean": sum(scores) / len(scores),
        "entropy": data.get("metrics", {}).get("entropy"),
    }

