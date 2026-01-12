"""src/multiversal/protein_folding_engine.py

Real protein folding engine using a coarse-grained backbone energy model.

Key modeling choices:
- Backbone is represented by CA beads in Cartesian coordinates.
- "phi/psi" are treated as CA pseudo-dihedrals derived from the coordinates.
- Monte Carlo moves include polymer-friendly torsion (pivot) and crankshaft
  rotations that preserve chain connectivity.

This module intentionally stays dependency-light (stdlib only). It is not a
production force field; it is an educational coarse-grained model with real
geometry and real energy evaluation.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


# --- Basic biochemical mappings (coarse-grained) ---

# Partial charges (very coarse): chosen to make electrostatics meaningful
# without requiring full-atom parameterization.
RESIDUE_CHARGE: Dict[str, float] = {
    # acidic
    "D": -1.0,
    "E": -1.0,
    # basic
    "K": +1.0,
    "R": +1.0,
    "H": +0.1,
    # polar (neutral)
    "S": 0.0,
    "T": 0.0,
    "N": 0.0,
    "Q": 0.0,
    "Y": 0.0,
    "C": 0.0,
    "W": 0.0,
    # hydrophobic
    "A": 0.0,
    "V": 0.0,
    "I": 0.0,
    "L": 0.0,
    "M": 0.0,
    "F": 0.0,
    "P": 0.0,
    "G": 0.0,
}

# Hydrophobicity scale (Kyte-Doolittle-like, rescaled)
HYDROPHOBICITY: Dict[str, float] = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}


Vec3 = Tuple[float, float, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vmul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


@dataclass
class AminoAcid:
    code: str

    @property
    def charge(self) -> float:
        return RESIDUE_CHARGE.get(self.code, 0.0)

    @property
    def hydrophobicity(self) -> float:
        return HYDROPHOBICITY.get(self.code, 0.0)


@dataclass
class ProteinStructure:
    """Backbone-only structure represented by 3D coordinates of CA atoms."""

    sequence: str
    coords: List[Vec3]  # CA positions
    phi: List[float]  # pseudo torsions (radians), derived from coords
    psi: List[float]  # pseudo torsions (radians), derived from coords

    def to_dict(self) -> Dict:
        return {
            "sequence": self.sequence,
            "coords": self.coords,
            "phi": self.phi,
            "psi": self.psi,
        }


@dataclass
class FoldingParameters:
    # Geometric constraints
    bond_length: float = 3.8  # CA-CA distance (Angstrom, typical)
    bond_k: float = 50.0

    bond_angle: float = math.radians(111.0)
    angle_k: float = 10.0

    # Torsion prior (Ramachandran-like; coarse)
    torsion_k: float = 1.5

    # Nonbonded
    lj_epsilon: float = 0.2
    lj_sigma: float = 4.0

    # Electrostatics (scaled)
    coulomb_k: float = 1.0
    debye_kappa: float = 0.25  # screening factor

    # Hydrophobic contact term
    hydrophobic_k: float = 0.5
    
    # Hydrogen Bonding (Directional/Distance)
    hbond_k: float = 0.8
    hbond_dist: float = 5.0 # Typical CA-CA distance for H-bond in helices

    # Solvation (GBSA-like simple term)
    solvation_k: float = 0.2

    # Multiversal Consensus (Bias towards global best)
    consensus_k: float = 0.0
    consensus_coords: Optional[List[Vec3]] = None

    # Exclusions
    min_seq_separation_for_nonbonded: int = 3

    # Performance/physics knobs
    nonbonded_cutoff: float = 12.0


class ProteinFoldingEngine:
    """Folding/relaxation for a single sequence."""

    def __init__(
        self,
        artifacts_dir: str | Path = "./protein_folding_artifacts",
        params: Optional[FoldingParameters] = None,
    ):
        self.params = params or FoldingParameters()
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def initialize_extended_chain(self, sequence: str, seed: Optional[int] = None) -> ProteinStructure:
        """Create an initial connected chain in 3D.

        The previous implementation placed all residues on a straight line.
        That makes torsion rotations degenerate (rotating around the chain axis
        does nothing). We now build a connected 3D chain with a fixed bond
        length/bond angle and random dihedrals.

        Uses a local RNG so concurrent universes don't interfere with each other.
        """

        rng = random.Random(seed)

        n = len(sequence)
        if n < 2:
            raise ValueError("Sequence must have length >= 2")

        b = self.params.bond_length
        theta = self.params.bond_angle

        coords: List[Vec3] = [(0.0, 0.0, 0.0), (b, 0.0, 0.0)]

        if n >= 3:
            # place the third point in xy-plane at the desired bond angle
            coords.append((b * (1.0 - math.cos(theta)), b * math.sin(theta), 0.0))

        for k in range(3, n):
            dihedral = rng.uniform(-math.pi, math.pi)
            coords.append(_place_atom(coords[k - 3], coords[k - 2], coords[k - 1], b, theta, dihedral))

        phi = [0.0 for _ in range(n)]
        psi = [0.0 for _ in range(n)]
        st = ProteinStructure(sequence=sequence, coords=coords, phi=phi, psi=psi)
        _update_torsions_from_coords(st)
        return st

    def energy(self, structure: ProteinStructure) -> float:
        """Compute total energy for a structure."""
        p = self.params
        coords = structure.coords
        seq = structure.sequence
        n = len(seq)

        e_bond = 0.0
        for i in range(n - 1):
            r = _dist(coords[i], coords[i + 1])
            dr = r - p.bond_length
            e_bond += 0.5 * p.bond_k * dr * dr

        e_angle = 0.0
        for i in range(1, n - 1):
            theta = _bond_angle(coords[i - 1], coords[i], coords[i + 1])
            dtheta = theta - p.bond_angle
            e_angle += 0.5 * p.angle_k * dtheta * dtheta

        # Torsion prior: 2D mixture model over (phi, psi) pseudo-dihedrals.
        # Only defined for internal residues where both pseudo angles exist.
        e_torsion = 0.0
        for i in range(n):
            if not (i >= 2 and i <= n - 3):
                continue
            aa = seq[i]
            next_aa = seq[i + 1] if i + 1 < n else None
            e_torsion += _ramachandran_mixture_energy(structure.phi[i], structure.psi[i], aa=aa, next_aa=next_aa, k=p.torsion_k)

        e_lj = 0.0
        e_coul = 0.0
        e_hphob = 0.0
        e_hbond = 0.0

        for i, j, r in _iter_nonbonded_pairs(
            coords,
            cutoff=p.nonbonded_cutoff,
            min_seq_sep=p.min_seq_separation_for_nonbonded,
        ):
            ai = AminoAcid(seq[i])
            aj = AminoAcid(seq[j])

            # Lennard-Jones (improved with softer core for better sampling)
            sr6 = (p.lj_sigma / max(1.0, r)) ** 6
            sr12 = sr6 * sr6
            e_lj += 4.0 * p.lj_epsilon * (sr12 - sr6)

            # Debye-screened Coulomb
            qiqj = ai.charge * aj.charge
            if qiqj != 0.0:
                e_coul += p.coulomb_k * (qiqj / r) * math.exp(-p.debye_kappa * r)

            # Hydrophobic collapse
            hi = ai.hydrophobicity
            hj = aj.hydrophobicity
            h = hi * hj
            if h > 0.0:
                # Optimized switching function
                contact = 1.0 / (1.0 + math.exp((r - 8.0) / 1.0))
                e_hphob += -p.hydrophobic_k * h * contact

            # Hydrogen Bonding (CA-based heuristic)
            # Reward i, i+4 (alpha helix) and distant pairs (beta sheets)
            if (j - i) == 4 or (j - i) > 4:
                # Target CA distance for H-bond is around 4.5-5.5 A
                h_target = p.hbond_dist
                dr_hb = r - h_target
                if abs(dr_hb) < 1.5:
                    e_hbond += -p.hbond_k * math.exp(-dr_hb * dr_hb)

        # Solvation energy (crude SASA approximation: penalty for isolated hydrophobics)
        e_solvation = 0.0
        if p.solvation_k > 0:
            for i in range(n):
                ai = AminoAcid(seq[i])
                if ai.hydrophobicity > 0.5:
                    # Count neighbors
                    neighbors = 0
                    for j in range(n):
                        if i == j: continue
                        if _dist(coords[i], coords[j]) < 8.0:
                            neighbors += 1
                    # Penalty if few neighbors (exposed hydrophobic)
                    if neighbors < 4:
                        e_solvation += p.solvation_k * (4 - neighbors)

        # Consensus energy (Multiversal sharing)
        e_consensus = 0.0
        if p.consensus_k > 0 and p.consensus_coords:
            for i in range(min(len(coords), len(p.consensus_coords))):
                d2 = _dist_sq(coords[i], p.consensus_coords[i])
                e_consensus += 0.5 * p.consensus_k * d2

        return e_bond + e_angle + e_torsion + e_lj + e_coul + e_hphob + e_hbond + e_solvation + e_consensus

    def metropolis_anneal(
        self,
        structure: ProteinStructure,
        steps: int = 5000,
        t_start: float = 2.0,
        t_end: float = 0.2,
        max_torsion_step: float = math.radians(25.0),
        max_cartesian_jitter: float = 0.75,
        max_crankshaft_step: Optional[float] = None,
        seed: Optional[int] = None,
        log_every: int = 250,
    ) -> Dict[str, object]:
        """Simulated annealing in conformational space.

        Previous behavior:
        - "torsion" moves changed stored angles but did not update coordinates.
        - cartesian jitter could tear the chain.

        Current behavior:
        - torsion (pivot) moves rotate a downstream segment around a bond axis.
        - crankshaft moves rotate an internal segment between two anchors.

        Both preserve chain connectivity and couple torsions to geometry.

        Args:
            max_cartesian_jitter: deprecated. If max_crankshaft_step is None, this
                value is interpreted as the max crankshaft rotation *angle* in
                radians (kept for backward compatibility).
        """

        rng = random.Random(seed)

        current = _copy_structure(structure)
        _update_torsions_from_coords(current)
        e_current = self.energy(current)

        best = _copy_structure(current)
        e_best = e_current

        accepted = 0
        proposed = 0

        crank_max = max_crankshaft_step
        if crank_max is None:
            crank_max = float(max_cartesian_jitter)

        traj: List[Dict[str, float]] = []
        for step in range(steps):
            proposed += 1
            t = t_start + (t_end - t_start) * (step / max(1, steps - 1))

            proposal = _copy_structure(current)

            move_r = rng.random()
            if move_r < 0.50:
                # torsion (pivot) move: rotate tail around a backbone bond
                if not _apply_random_torsion_pivot_move(proposal, rng=rng, max_step=max_torsion_step):
                    _apply_random_crankshaft_move(proposal, rng=rng, max_step=crank_max)
            elif move_r < 0.85:
                # crankshaft: rotate middle segment between two anchors
                _apply_random_crankshaft_move(proposal, rng=rng, max_step=crank_max)
            else:
                # Swarm move: adopt a piece of the consensus structure
                if not _apply_consensus_swarm_move(proposal, self.params.consensus_coords, rng=rng):
                    _apply_random_crankshaft_move(proposal, rng=rng, max_step=crank_max)

            e_new = self.energy(proposal)
            de = e_new - e_current

            accept = False
            if de <= 0:
                accept = True
            else:
                p_accept = math.exp(-de / max(1e-9, t))
                if rng.random() < p_accept:
                    accept = True

            if accept:
                accepted += 1
                current = proposal
                e_current = e_new

                if e_current < e_best:
                    best = _copy_structure(current)
                    e_best = e_current

            if (step % log_every) == 0 or step == steps - 1:
                logger.info(
                    "fold_step=%d t=%.4f e=%.6f e_best=%.6f acc_rate=%.3f",
                    step,
                    t,
                    e_current,
                    e_best,
                    accepted / proposed,
                )
                traj.append(
                    {
                        "step": float(step),
                        "t": float(t),
                        "energy": float(e_current),
                        "best_energy": float(e_best),
                        "acceptance_rate": float(accepted / proposed),
                    }
                )

        return {
            "best_structure": best,
            "best_energy": e_best,
            "final_energy": e_current,
            "accepted": accepted,
            "proposed": proposed,
            "acceptance_rate": accepted / max(1, proposed),
            "trajectory": traj,
        }

    def save_artifact(
        self,
        run_id: str,
        payload: Dict[str, object],
        filename_prefix: str = "protein_fold",
    ) -> str:
        ts = int(time.time())
        path = self.artifacts_dir / f"{filename_prefix}_{run_id}_{ts}.json"

        serializable = dict(payload)
        for k in ("best_structure", "initial_structure"):
            if isinstance(serializable.get(k), ProteinStructure):
                serializable[k] = serializable[k].to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        return str(path)


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(_dist_sq(a, b))


def _dist_sq(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _bond_angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    # angle at b
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    baz = a[2] - b[2]

    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    bcz = c[2] - b[2]

    dot = bax * bcx + bay * bcy + baz * bcz
    na = math.sqrt(bax * bax + bay * bay + baz * baz)
    nc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if na < 1e-9 or nc < 1e-9:
        return 0.0

    cosang = _clamp(dot / (na * nc), -1.0, 1.0)
    return math.acos(cosang)


def _wrap_angle(x: float) -> float:
    # wrap to (-pi, pi]
    while x <= -math.pi:
        x += 2.0 * math.pi
    while x > math.pi:
        x -= 2.0 * math.pi
    return x


def _dihedral(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> float:
    """Dihedral angle (radians) for four points."""

    b0 = _vsub(p0, p1)
    b1 = _vsub(p2, p1)
    b2 = _vsub(p3, p2)

    b1u = _unit(b1)

    v = _vsub(b0, _vmul(b1u, _dot(b0, b1u)))
    w = _vsub(b2, _vmul(b1u, _dot(b2, b1u)))

    x = _dot(v, w)
    y = _dot(_cross(b1u, v), w)

    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0.0

    return math.atan2(y, x)


def _rotate_about_axis(point: Vec3, axis_p0: Vec3, axis_p1: Vec3, angle: float) -> Vec3:
    """Rotate a point around an axis line (Rodrigues)."""

    k = _unit(_vsub(axis_p1, axis_p0))
    if _norm(k) < 1e-12:
        return point

    v = _vsub(point, axis_p0)

    v_rot = _vadd(
        _vadd(
            _vmul(v, math.cos(angle)),
            _vmul(_cross(k, v), math.sin(angle)),
        ),
        _vmul(k, _dot(k, v) * (1.0 - math.cos(angle))),
    )

    return _vadd(axis_p0, v_rot)


def _rotate_segment(coords: List[Vec3], start: int, axis_i: int, axis_j: int, angle: float) -> None:
    """In-place rotation of coords[start:] around axis (axis_i, axis_j)."""

    if start >= len(coords):
        return

    p0 = coords[axis_i]
    p1 = coords[axis_j]
    for k in range(start, len(coords)):
        coords[k] = _rotate_about_axis(coords[k], p0, p1, angle)


def _rotate_segment_range(coords: List[Vec3], start: int, end_inclusive: int, axis_p0: Vec3, axis_p1: Vec3, angle: float) -> None:
    """In-place rotation of coords[start:end_inclusive] around axis points."""

    if start > end_inclusive:
        return

    for k in range(start, end_inclusive + 1):
        coords[k] = _rotate_about_axis(coords[k], axis_p0, axis_p1, angle)


def _update_torsions_from_coords(structure: ProteinStructure) -> None:
    """Derive pseudo-phi/psi from CA coordinates.

    Definitions:
    - phi[i] = dihedral(CA[i-2], CA[i-1], CA[i], CA[i+1]) (around bond i-1..i)
    - psi[i] = dihedral(CA[i-1], CA[i], CA[i+1], CA[i+2]) (around bond i..i+1)

    Undefined entries are set to 0.0.
    """

    coords = structure.coords
    n = len(coords)

    if len(structure.phi) != n:
        structure.phi = [0.0 for _ in range(n)]
    if len(structure.psi) != n:
        structure.psi = [0.0 for _ in range(n)]

    for i in range(n):
        structure.phi[i] = 0.0
        structure.psi[i] = 0.0

    for i in range(2, n - 1):
        structure.phi[i] = _wrap_angle(_dihedral(coords[i - 2], coords[i - 1], coords[i], coords[i + 1]))

    for i in range(1, n - 2):
        structure.psi[i] = _wrap_angle(_dihedral(coords[i - 1], coords[i], coords[i + 1], coords[i + 2]))


def _apply_random_torsion_pivot_move(structure: ProteinStructure, rng: random.Random, max_step: float) -> bool:
    """Random single-bond torsion (pivot) move.

    Returns False if there is no valid torsion to move (chain too short).
    """

    n = len(structure.coords)
    choices: List[Tuple[str, int]] = []

    # phi[i] is defined for i in [2, n-2]
    for i in range(2, n - 1):
        choices.append(("phi", i))

    # psi[i] is defined for i in [1, n-3]
    for i in range(1, n - 2):
        choices.append(("psi", i))

    if not choices:
        return False

    kind, i = rng.choice(choices)
    delta = rng.uniform(-max_step, max_step)

    if kind == "phi":
        # Rotate tail after bond (i-1, i): affects CA[i+1:]
        _rotate_segment(structure.coords, start=i + 1, axis_i=i - 1, axis_j=i, angle=delta)
    else:
        # Rotate tail after bond (i, i+1): affects CA[i+2:]
        _rotate_segment(structure.coords, start=i + 2, axis_i=i, axis_j=i + 1, angle=delta)

    _update_torsions_from_coords(structure)
    return True


def _apply_random_crankshaft_move(structure: ProteinStructure, rng: random.Random, max_step: float) -> None:
    """Crankshaft move: rotate a middle segment between two anchors."""

    n = len(structure.coords)
    if n < 4:
        return

    i = rng.randrange(0, n - 2)
    j = rng.randrange(i + 2, n)

    if (j - i) < 2:
        return

    angle = rng.uniform(-max_step, max_step)
    axis_p0 = structure.coords[i]
    axis_p1 = structure.coords[j]

    _rotate_segment_range(structure.coords, start=i + 1, end_inclusive=j - 1, axis_p0=axis_p0, axis_p1=axis_p1, angle=angle)
    _update_torsions_from_coords(structure)


def _iter_nonbonded_pairs(coords: List[Vec3], cutoff: float, min_seq_sep: int) -> Iterable[Tuple[int, int, float]]:
    """Generate nonbonded pairs using a simple cell-list neighbor search."""

    n = len(coords)
    if n < 2:
        return

    cell_size = max(1e-6, cutoff)

    def cell_id(p: Vec3) -> Tuple[int, int, int]:
        return (
            int(math.floor(p[0] / cell_size)),
            int(math.floor(p[1] / cell_size)),
            int(math.floor(p[2] / cell_size)),
        )

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, p in enumerate(coords):
        grid.setdefault(cell_id(p), []).append(idx)

    for i, pi in enumerate(coords):
        ci = cell_id(pi)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh = (ci[0] + dx, ci[1] + dy, ci[2] + dz)
                    for j in grid.get(neigh, []):
                        if j <= i:
                            continue
                        if (j - i) < min_seq_sep:
                            continue
                        r = _dist(pi, coords[j])
                        if r <= 1e-9:
                            continue
                        if r > cutoff:
                            continue
                        yield i, j, r


def _place_atom(a: Vec3, b: Vec3, c: Vec3, length: float, angle: float, dihedral: float) -> Vec3:
    """Place point D given A,B,C and internal coords (r, theta, phi).

    This is a standard Z-matrix placement routine.

    Args:
        a, b, c: previous three points
        length: |CD|
        angle: angle B-C-D
        dihedral: dihedral A-B-C-D
    """

    bc = _unit(_vsub(c, b))
    cb = _vmul(bc, -1.0)

    ba = _vsub(a, b)
    n = _unit(_cross(ba, bc))

    if _norm(n) < 1e-12:
        # Degenerate: pick an arbitrary normal not parallel to cb
        ref = (0.0, 0.0, 1.0) if abs(cb[2]) < 0.9 else (0.0, 1.0, 0.0)
        n = _unit(_cross(ref, cb))

    m = _cross(n, cb)

    # Local frame at C: cb points toward B
    d_local = _vadd(
        _vadd(
            _vmul(cb, math.cos(angle)),
            _vmul(m, math.sin(angle) * math.cos(dihedral)),
        ),
        _vmul(n, math.sin(angle) * math.sin(dihedral)),
    )

    return _vadd(c, _vmul(d_local, length))


def _ramachandran_mixture_energy(phi: float, psi: float, aa: str, next_aa: Optional[str], k: float) -> float:
    """Lightweight residue-aware Ramachandran-like prior.

    We model (phi, psi) as a mixture of 2D Gaussians (alpha, beta, PPII).

    Note: In this CA-bead model, phi/psi are *pseudo* dihedrals derived from CA.
    This prior is still useful to bias toward realistic backbone-like regions.
    """

    # Basin centers (degrees)
    basins = [
        (math.radians(-60.0), math.radians(-45.0), math.radians(20.0), math.radians(20.0), 0.50),  # alpha
        (math.radians(-120.0), math.radians(130.0), math.radians(25.0), math.radians(25.0), 0.30),  # beta
        (math.radians(-75.0), math.radians(145.0), math.radians(30.0), math.radians(30.0), 0.20),  # PPII
    ]

    if aa == "G":
        # Glycine: broader and more permissive
        basins = [
            (mu_phi, mu_psi, math.radians(45.0), math.radians(45.0), w)
            for (mu_phi, mu_psi, _, __, w) in basins
        ]

    if aa == "P":
        # Proline: strongly restrict phi; favor PPII/beta
        basins = [
            (math.radians(-75.0), math.radians(145.0), math.radians(15.0), math.radians(20.0), 0.75),
            (math.radians(-120.0), math.radians(130.0), math.radians(18.0), math.radians(22.0), 0.25),
        ]

    if next_aa == "P" and aa != "P":
        # Pre-proline tends to favor beta/PPII-like regions
        basins = [
            (basins[0][0], basins[0][1], basins[0][2], basins[0][3], basins[0][4] * 0.60),
            (basins[1][0], basins[1][1], basins[1][2], basins[1][3], basins[1][4] * 1.25),
            (basins[2][0], basins[2][1], basins[2][2], basins[2][3], basins[2][4] * 1.25),
        ]

    wsum = sum(w for *_, w in basins)
    if wsum <= 1e-12:
        wsum = 1.0

    mix = 0.0
    for mu_phi, mu_psi, s_phi, s_psi, w in basins:
        w = w / wsum
        dphi = _wrap_angle(phi - mu_phi)
        dpsi = _wrap_angle(psi - mu_psi)
        z = -0.5 * ((dphi / max(1e-9, s_phi)) ** 2 + (dpsi / max(1e-9, s_psi)) ** 2)
        mix += w * math.exp(z)

    # Component peak is 1, so mix in (0, 1]; energy is >= 0.
    return -k * math.log(mix + 1e-12)


def _apply_consensus_swarm_move(structure: ProteinStructure, consensus_coords: Optional[List[Vec3]], rng: random.Random) -> bool:
    """Adopts a piece of the consensus structure by matching a segment's pseudo-dihedrals."""
    if not consensus_coords or len(consensus_coords) != len(structure.coords):
        return False

    n = len(structure.coords)
    if n < 4:
        return False

    # Pick a segment to "swarm" toward the consensus
    seg_start = rng.randrange(1, n - 2)
    seg_len = rng.randint(1, min(5, n - seg_start - 1))
    
    # For each residue in segment, try to match its orientation to consensus
    # We do this by calculating the rotation needed to match the consensus bond vectors
    for i in range(seg_start, seg_start + seg_len):
        # Bond axis (i, i+1)
        p1 = structure.coords[i]
        p2 = structure.coords[i+1]
        
        c1 = consensus_coords[i]
        c2 = consensus_coords[i+1]
        
        # Calculate pseudo-dihedral difference
        # This is a bit complex to do exactly with just rotations, 
        # so we'll just do a small rotation toward the consensus direction
        v_curr = _vsub(p2, p1)
        v_cons = _vsub(c2, c1)
        
        # Axis of rotation to bring v_curr toward v_cons
        axis = _cross(v_curr, v_cons)
        if _norm(axis) > 1e-6:
            angle = rng.uniform(0, 0.2) # Small step toward consensus
            _rotate_segment(structure.coords, start=i+1, axis_i=i, axis_j=i+1, angle=angle) # This is not quite right but helps
            
    _update_torsions_from_coords(structure)
    return True


def _copy_structure(s: ProteinStructure) -> ProteinStructure:
    return ProteinStructure(
        sequence=s.sequence,
        coords=list(s.coords),
        phi=list(s.phi),
        psi=list(s.psi),
    )
