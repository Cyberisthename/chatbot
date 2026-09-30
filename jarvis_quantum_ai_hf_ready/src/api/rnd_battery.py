"""
R&D battery API — JARVIS server probe-battery endpoints (self-contained).

Wires the OWNED probe battery into the JARVIS server as runnable API endpoints,
backed by the local SQLite provenance store (repo-local `jarvis.db`, PR #136):

  POST /api/rnd/fbsc          FBSC core compression (n_qubits, d)      [measured]
  POST /api/rnd/likeness      quantum-likeness probe battery           [measured]
  POST /api/rnd/lhv_qudit     crown jewel: CGLMP + explicit LHV LP     [measured]
  POST /api/rnd/trainer       3-phase trainer (anneal/evolve/gate)     [measured]
  POST /api/rnd/nitrogenase   FeMo-co qudit simulator (clean base) [measured/interpretation]
  POST /api/rnd/resonance     41.02 Hz noise-gated resonance monitor [interpretation/speculation]
  GET  /api/rnd/health        battery inventory + claims policy
  GET  /api/rnd/runs          provenance rows (seed digest only — raw seed NEVER stored)

Every response carries the house claims envelope:
  claims.measured / claims.interpretation / claims.speculation /
  claims.honest_ceiling   (exact classical simulation; no hardware claims)

Own-code constraint: only the team's own modules are imported
(compression_specialist, likeness_probes, qudit_lhv_probe,
quantum_topological_trainer, nitrogenase_qudit_simulator,
eeg_to_tonal_engine). No third-party quantum SDKs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# repo-root discovery (works from any checkout location; never hardcoded)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../src/api
_REPO_ROOT = _THIS_DIR.parent.parent.parent          # repo root
_REPO_SRC = _REPO_ROOT / "src"                       # repo-level src/ (quantum_llm)
_QL_DIR = _REPO_ROOT / "docs" / "artifacts" / "quantum_likeness"
_TR_DIR = _REPO_ROOT / "docs" / "artifacts" / "trainer"
_NQ_DIR = _REPO_ROOT / "docs" / "artifacts" / "nitrogenase_qudit"

for _p in (_REPO_ROOT, _REPO_SRC, _QL_DIR, _TR_DIR, _NQ_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_SCRATCH_ROOT = Path(os.environ.get("JARVIS_RND_SCRATCH", "/var/tmp"))
_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI (imported lazily so battery-only tools can reuse this file)
# ---------------------------------------------------------------------------
try:
    from fastapi import APIRouter, FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_OK = True
except Exception:  # pragma: no cover — import-guard for non-server contexts
    APIRouter = FastAPI = HTTPException = Query = JSONResponse = None  # type: ignore
    BaseModel = Field = None  # type: ignore
    _FASTAPI_OK = False

# ---------------------------------------------------------------------------
# provenance store (PR #136: repo-local SQLite; DB-optional)
# ---------------------------------------------------------------------------
_store_lock = threading.Lock()
def _get_store():
    from quantum_llm.db_store import DbStore  # src/quantum_llm, via repo root
    return DbStore()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _seed_digest(seed: Any) -> str:
    """One-way digest of a seed triple for provenance. Raw seed is NEVER stored."""
    if seed is None:
        return "owner-default"
    try:
        vals = [round(float(v), 12) for v in seed]
        canon = json.dumps(vals, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "invalid"

def _persist_experiment(kind: str, verdict: Dict[str, Any]) -> Optional[str]:
    """Write one provenance row; returns experiment id or None (DB-optional)."""
    try:
        with _store_lock:
            store = _get_store()
            store.init_db()
            canon = json.dumps(verdict, sort_keys=True, default=str)
            exp_id = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
            store.record_experiment(exp_id, kind=kind, verdict=verdict)
            return exp_id
    except Exception:
        return None

# ---------------------------------------------------------------------------
# json-safety: numpy scalars / complex -> json-ready primitives
# ---------------------------------------------------------------------------
def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and callable(getattr(o, "item", None)):
        try:
            return _clean(o.item())
        except Exception:
            pass
    if isinstance(o, complex):
        return {"re": round(o.real, 12), "im": round(o.imag, 12)}
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)

_HONEST_CEILING = (
    "exact classical simulation of an owned deterministic state family; "
    "no quantum hardware; no universal-QC claim; no loophole-free Bell test; "
    "'sentience trigger' is a spectral-detection event, not a consciousness claim"
)

# ---------------------------------------------------------------------------
# battery wrappers (lazy imports so /health works even if one module is heavy)
# ---------------------------------------------------------------------------
_import_lock = threading.Lock()
_MODULES: Dict[str, Any] = {}

def _battery(mod: str) -> Any:
    """Import one owned battery module once (thread-safe)."""
    with _import_lock:
        if mod in _MODULES:
            return _MODULES[mod]
        if mod == "compression_specialist":
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "compression_specialist",
                str(_REPO_ROOT / "compression_specialist.py"))
            m = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(m)
        elif mod == "likeness_probes":
            import likeness_probes
            m = likeness_probes
        elif mod == "qudit_lhv_probe":
            import qudit_lhv_probe
            m = qudit_lhv_probe
        elif mod == "trainer":
            import quantum_topological_trainer
            m = quantum_topological_trainer
        elif mod == "nitrogenase":
            import nitrogenase_qudit_simulator
            m = nitrogenase_qudit_simulator
        elif mod == "resonance":
            from quantum_llm.eeg_to_tonal_engine import ResonanceMonitor
            m = ResonanceMonitor
        else:
            raise KeyError(mod)
        _MODULES[mod] = m
        return m


def _normalize_seed(seed: Any, default: Any) -> tuple:
    if seed is None:
        return tuple(default)
    vals = [float(v) for v in seed]
    if len(vals) != 3:
        raise HTTPException(400, f"seed must be [a,b,c] (3 floats), got {len(vals)}")
    return tuple(vals)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class FBSCRequest(BaseModel):
    n_qubits: int = Field(16, ge=2, le=1024, description="effective qubits (d² states)")
    d: int = Field(2, ge=2, le=3)
    seed: Optional[List[float]] = None


class LikenessRequest(BaseModel):
    probes: List[str] = Field(
        default_factory=lambda: ["entanglement", "magic", "wigner", "mps"],
        description="subset of entanglement|magic|wigner|xeb|mps|lhv_qubit")
    n: int = Field(6, ge=2, le=16)
    scheme: str = Field("fold", pattern="^(fold|hash)$")
    depth_scale: float = Field(1.0, ge=0.25, le=16.0)
    seed: Optional[List[float]] = None


class LHVQuditRequest(BaseModel):
    d: int = Field(3, ge=2, le=4)
    n_units: int = Field(2, ge=2, le=4)
    trials: int = Field(25, ge=5, le=90)
    depth_scale: float = Field(8.0, ge=0.25, le=16.0)
    seed: Optional[List[float]] = None


class TrainerRequest(BaseModel):
    gd_steps: int = Field(400, ge=50, le=2000)
    seed_budget: int = Field(400, ge=50, le=4000)
    anneal_steps: int = Field(400, ge=50, le=2000)
    pop: int = Field(12, ge=4, le=40)
    generations: int = Field(15, ge=3, le=70)
    task_seed: int = Field(2026, ge=0, le=2 ** 31 - 1)


class NitrogenaseRequest(BaseModel):
    d: int = Field(3, ge=2, le=4)
    n_qudits: int = Field(16, ge=4, le=64)
    seed: Optional[List[float]] = None


class ResonanceRequest(BaseModel):
    tone_hz: float = Field(41.02, ge=1.0, le=120.0)
    duration_s: float = Field(4.0, ge=1.0, le=30.0)
    fs: int = Field(250, ge=100, le=1000)
    amplitude: float = Field(0.5, ge=0.01, le=1.0)
    noise_floor_db: float = Field(-40.0, ge=-90.0, le=0.0)

def _envelope(endpoint: str, measured: List[str], interpretation: List[str],
              speculation: List[str], data: Any, seed: Any = None,
              extra_provenance: Optional[Dict[str, Any]] = None,
              persist_kind: Optional[str] = None) -> Dict[str, Any]:
    prov = {"seed_digest": _seed_digest(seed), "seed_stored": False,
            "timestamp": _now_iso(), "source": "jarvis-rnd-battery"}
    if extra_provenance:
        prov.update(extra_provenance)
    env = {
        "endpoint": endpoint,
        "status": "ok",
        "claims": {
            "measured": measured,
            "interpretation": interpretation,
            "speculation": speculation,
            "honest_ceiling": _HONEST_CEILING,
        },
        "data": _clean(data),
        "provenance": prov,
    }
    # provenance persistence is DB-optional: failures never affect the response
    if persist_kind is not None:
        try:
            verdict = {"endpoint": endpoint,
                       "claims": env["claims"],
                       "data": env["data"],
                       "seed_digest": prov["seed_digest"],
                       "seed_stored": False}
            exp_id = _persist_experiment(persist_kind, verdict)
            prov["experiment_id"] = exp_id
        except Exception:
            prov["experiment_id"] = None
            prov["persist_error"] = "DB write skipped (DB-optional)"
    return env

# ---------------------------------------------------------------------------
# endpoint implementations
# ---------------------------------------------------------------------------
def _handle_fbsc(req: FBSCRequest) -> Dict[str, Any]:
    cs = _battery("compression_specialist")
    seed = _normalize_seed(req.seed, (0.57721, 1.618034, 2.71828))
    t0 = time.time()
    # clean base: run with cwd = scratch so jarvis_qvgpu_trained.npz is NOT
    # picked up (keeps endpoint numbers identical to approved clean-base runs;
    # npz presence is cwd-dependent and documented in /health)
    cwd = os.getcwd()
    scratch = _SCRATCH_ROOT / f"fbsc_{int(time.time()*1000)}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(scratch)
        comp = cs.FractalBraidSeedCompressor(
            n_effective_qubits=req.n_qubits, seed=seed, qudit_dim=req.d)
        amps, positions, metrics = comp.reconstruct()
    finally:
        os.chdir(cwd)
    data = {
        "effective_qubits": metrics["effective_qubits"],
        "total_hilbert_dim": metrics["total_hilbert_dim"],
        "mse": metrics["reconstruction_mse"],
        "compression_ratio_log10": metrics["compression_ratio_log10"],
        "compression_ratio": metrics.get("compression_ratio"),
        "memory_kb": metrics["memory_kb"],
        "geometric_fold_factor": metrics["geometric_fold_factor"],
        "avg_braid_crossings": metrics["avg_braid_crossings"],
        "time_seconds": round(time.time() - t0, 4),
        "base_note": ("clean scratch cwd (no qvgpu npz); exact classical "
                      "simulation, deterministically reproducible"),
    }
    prov = {"kind": "fbsc", "n_qubits": req.n_qubits, "d": req.d}
    return _envelope(
        "fbsc",
        measured=["mse", "compression_ratio_log10", "memory_kb",
                  "geometric_fold_factor", "total_hilbert_dim",
                  "time_seconds"],
        interpretation=["compression ratio expresses O(N) stored vs 2^N dense "
                        "Hilbert entries; state family is <=3-parameter",
                        "base_note: clean-cwd run (npz-independent)"],
        speculation=[],
        data=data, seed=seed, extra_provenance=prov,
        persist_kind="probe_battery")


def _handle_likeness(req: LikenessRequest) -> Dict[str, Any]:
    lp = _battery("likeness_probes")
    seed = _normalize_seed(req.seed, lp.OWNER_SEED)
    n = req.n
    out: Dict[str, Any] = {}
    allowed = {"entanglement", "magic", "wigner", "xeb", "mps", "lhv_qubit"}
    unknown = set(req.probes) - allowed
    if unknown:
        raise HTTPException(400, f"unknown probes: {sorted(unknown)}; allowed={sorted(allowed)}")
    psi = None
    for probe in req.probes:
        if probe == "entanglement":
            t0 = time.time()
            psi = lp.fbsc_braid_state(seed, n, scheme=req.scheme, depth_scale=req.depth_scale)
            S, _ = lp.bipartite_entropy(psi, n, n // 2)
            alpha = lp.volume_law_fraction(psi, n)
            A, psic = lp.fbsc_core_state(seed, n, d=2)
            Sc, _ = lp.bipartite_entropy(psic, n, n // 2)
            out["entanglement"] = {
                "n": n, "scheme": req.scheme, "depth_scale": req.depth_scale,
                "braid_S_mid": round(float(S), 4),
                "braid_volume_law_alpha": round(float(alpha), 4),
                "core_FBSC_area_law_S_mid": round(float(Sc), 4),
                "note": ("volume-law-like (alpha>0) for braided elaboration; "
                         "core single-excitation embedding stays area-law"),
                "time_seconds": round(time.time() - t0, 3)}
        elif probe == "magic":
            t0 = time.time()
            if psi is None:
                psi = lp.fbsc_braid_state(seed, n, scheme=req.scheme, depth_scale=req.depth_scale)
            m2 = lp.sre2(psi, n)[0]
            out["magic"] = {"n": n, "sre2_M2": round(float(m2), 4),
                            "note": "M2>0 => beyond Clifford (GK route closed)",
                            "time_seconds": round(time.time() - t0, 3)}
        elif probe == "wigner":
            t0 = time.time()
            wq = {}
            for nn in (min(4, n),):
                ps = lp.fbsc_braid_state(seed, nn, scheme=req.scheme, depth_scale=req.depth_scale)
                W = lp.qubit_wigner(ps, nn)
                neg = float(min(W.ravel())) if hasattr(W, "ravel") else None
                wq[f"n{nn}"] = {"min_W": round(neg, 6) if neg is not None else None,
                                "negativity_witness": bool(neg is not None and neg < 0)}
            out["wigner_qubit"] = {"n": min(4, n), **wq,
                                   "note": ("W<0 is a non-stabilizerness witness; "
                                            "even-dim caveat: representation-based, "
                                            "exact change of variables, not compression")}
            try:
                cw = {}
                for nu in (2, 3):
                    A, psq = lp.fbsc_core_state_qudit(seed, nu, d=3)
                    Wq = lp.qudit_wigner_qutrit(psq, nu)
                    cw[f"nu{nu}"] = {"shape": list(Wq.shape),
                                     "min_W": round(float(min(Wq.ravel())), 6)}
                out["wigner_qutrit"] = cw
            except Exception as e:  # honest pass-through
                out["wigner_qutrit"] = {"skipped": repr(e)[:200]}
            out["wigner_time_seconds"] = round(time.time() - t0, 3)
        elif probe == "xeb":
            import numpy as np
            t0 = time.time()
            ps = lp.fbsc_braid_state(seed, n, scheme=req.scheme, depth_scale=req.depth_scale)
            F = lp.xeb_purity(ps, n)
            Fs = lp.xeb_sampled(ps, n)
            Fh, Fhstd = lp.haar_xeb_mean(n)
            product_avg = np.ones(2 ** n, complex) / np.sqrt(2 ** n)
            Fp = lp.xeb_purity(product_avg, n)
            out["xeb"] = {"n": n, "braid_F": round(float(F), 3),
                          "sampled": round(float(Fs), 3),
                          "haar": round(float(Fh), 3), "product": round(float(Fp), 3),
                          "time_seconds": round(time.time() - t0, 3)}
        elif probe == "mps":
            t0 = time.time()
            if psi is None:
                psi = lp.fbsc_braid_state(seed, n, scheme=req.scheme, depth_scale=req.depth_scale)
            req_D, fid2, fids = lp.mps_required_D(psi, n)
            out["mps"] = {"n": n, "D_for_0.99": req_D,
                          "fid_D2": round(float(fid2), 4) if fid2 is not None else None,
                          "note": ("TT-SVD bond dim needed for fidelity>=0.99; "
                                   "FBSC reproduces same state exactly with O(1) params"),
                          "time_seconds": round(time.time() - t0, 3)}
        elif probe == "lhv_qubit":
            t0 = time.time()
            ps2 = lp.fbsc_braid_state(seed, 2, scheme=req.scheme, depth_scale=req.depth_scale)
            ch, _T = lp.chimax(ps2, 2)
            ok, _ = lp.lhv_feasible(ps2, 2)
            out["lhv_qubit_n2"] = {"chsh": round(float(ch), 4),
                                   "violates": bool(ch > 2.0 + 1e-9),
                                   "lhv_model_EXISTS": bool(ok),
                                   "lhv_EXCLUDED": bool(not ok),
                                   "note": ("computed correlation statistics of the "
                                            "state; not a loophole-free lab test"),
                                   "time_seconds": round(time.time() - t0, 3)}
    return _envelope(
        "likeness",
        measured=["all numeric probe outputs (S, alpha, M2, XEB F, D_0.99, CHSH)"],
        interpretation=["non-stabilizerness/Wigner-negativity/CHSH>2 are genuine "
                        "non-classical witnesses of the computed state statistics",
                        "entanglement scaling distinction: braided elaboration vs "
                        "core single-excitation embedding"],
        speculation=["none — all outputs are statistics of an explicitly "
                     "classical deterministic generator"],
        data=out, seed=seed,
        extra_provenance={"kind": "likeness", "n": n, "scheme": req.scheme,
                          "probes": req.probes},
        persist_kind="probe_battery")


def _handle_lhv_qudit(req: LHVQuditRequest) -> Dict[str, Any]:
    ql = _battery("qudit_lhv_probe")
    lp = _battery("likeness_probes")
    seed = _normalize_seed(req.seed, lp.OWNER_SEED)
    t0 = time.time()
    amps = ql.qudit_braid_state(seed, req.n_units, req.d, depth_scale=req.depth_scale)
    rho = ql.qudit_reduced_2(amps, req.n_units, req.d)
    I, P = ql.optimize_cglmp(rho, req.d, trials=req.trials)
    feas, _ = ql.lhv_lp_full(rho, req.d, P)
    row = {
        "d": req.d, "n_units": req.n_units, "label": "FBSC-qudit-braid",
        "LHV_bound": ql.LHV_BOUND,
        "I_cglmp": round(float(I), 4),
        "margin_over_LHV": round(float(I) - ql.LHV_BOUND, 4),
        "violates": bool(I > ql.LHV_BOUND + 1e-9),
        "lhv_model_EXISTS": bool(feas),
        "lhv_EXCLUDED": bool(not feas),
        "lit_quantum_max_MES": ql.KNOWN_QM.get(req.d),
        "trials": req.trials,
        "time_seconds": round(time.time() - t0, 3),
        "note": ("explicit convex-hull LP over deterministic-local-strategy "
                 "polytope; infeasible LP => no LHV model for these computed "
                 "statistics; not a loophole-free lab test"),
    }
    return _envelope(
        "lhv_qudit",
        measured=["I_cglmp", "margin_over_LHV", "violates", "lhv_model_EXISTS",
                  "lhv_EXCLUDED"],
        interpretation=["provably outside every LHV model = statement about "
                        "computed correlation statistics of the FBSC state family"],
        speculation=[],
        data=row, seed=seed,
        extra_provenance={"kind": "lhv_qudit", "d": req.d, "n_units": req.n_units},
        persist_kind="probe_battery")


def _handle_trainer(req: TrainerRequest) -> Dict[str, Any]:
    tr = _battery("trainer")
    t0 = time.time()
    X, Y, W_star, w_teacher = tr.make_task(seed=req.task_seed)
    W_gd, loss_gd, evals_gd = tr.gradient_descent(X, Y, steps=req.gd_steps, lr=0.5)
    seed_best, loss_seed, evals_seed = tr.seed_optimizer(X, Y, budget=req.seed_budget)
    w_anneal, loss_anneal, evals_anneal = tr.anneal(X, Y, steps=req.anneal_steps)
    w_ev, loss_ev, evals_ev = tr.evolve(X, Y, pop=req.pop, generations=req.generations)
    p1 = tr.phase1_noise_immunity(w_teacher, W_star)
    loss_teacher = tr.loss_from_word(w_teacher, X, Y)
    trainer_best = min(loss_anneal, loss_ev)
    TOL = 1e-6
    result = {
        "task": {"teacher_loss_floor": round(float(loss_teacher), 10)},
        "phase1_noise_immunity": p1,
        "phase2_annealing": {"loss": round(float(loss_anneal), 8), "evals": evals_anneal},
        "phase3_evolution": {"loss": round(float(loss_ev), 8), "evals": evals_ev},
        "baselines": {
            "gradient_descent": {"loss": round(float(loss_gd), 8), "evals": evals_gd},
            "seed_optimizer": {"loss": round(float(loss_seed), 8), "evals": evals_seed},
        },
        "gate": {
            "tolerance": TOL,
            "trainer_best_loss": round(float(trainer_best), 8),
            "gradient_descent_loss": round(float(loss_gd), 8),
            "seed_optimizer_loss": round(float(loss_seed), 8),
            "beats_gd": bool(trainer_best < loss_gd - TOL),
            "beats_seed": bool(trainer_best < loss_seed - TOL),
            "verdict": ("no quantum hardware / no true tunneling; deterministic "
                        "anneal/evolution over braid space is the honest analogue; "
                        "gate verdict open — may tie or lose to GD"),
        },
        "wall_clock_seconds": round(time.time() - t0, 3),
        "params": {"gd_steps": req.gd_steps, "seed_budget": req.seed_budget,
                   "anneal_steps": req.anneal_steps, "pop": req.pop,
                   "generations": req.generations, "task_seed": req.task_seed},
    }
    return _envelope(
        "trainer",
        measured=["all losses", "evals", "phase1 noise-immunity metrics",
                  "wall_clock_seconds"],
        interpretation=["gate verdict compares trainer against classical "
                        "baselines at equal budget; honest negative allowed"],
        speculation=[],
        data=result, seed=None,
        extra_provenance={"kind": "trainer", "task_seed": req.task_seed},
        persist_kind="probe_battery")


def _handle_nitrogenase(req: NitrogenaseRequest) -> Dict[str, Any]:
    nq = _battery("nitrogenase")
    seed = _normalize_seed(req.seed, nq.OWNER_SEED)
    t0 = time.time()
    cwd = os.getcwd()
    scratch = _SCRATCH_ROOT / f"nq_{int(time.time()*1000)}"
    scratch.mkdir(parents=True, exist_ok=True)
    # redirect the simulator's hardcoded OUT_DIR to scratch (disk hygiene)
    old_out = nq.OUT_DIR
    nq.OUT_DIR = scratch
    try:
        os.chdir(scratch)   # clean base: no jarvis_qvgpu_trained.npz in cwd
        old_seed = nq.OWNER_SEED
        nq.OWNER_SEED = seed
        try:
            res = nq.simulate_dimension(req.d, n_qudits=req.n_qudits)
        finally:
            nq.OWNER_SEED = old_seed
    finally:
        os.chdir(cwd)
        nq.OUT_DIR = old_out
    res["time_seconds"] = round(time.time() - t0, 3)
    res["base_note"] = (
        "clean scratch cwd (jarvis_qvgpu_trained.npz NOT loaded) — numbers match "
        "the approved clean-base report (d=4 barrier 100.7657 kJ/mol). If the "
        "server process cwd contained the npz, d=4 barrier would be 98.0225 "
        "(cwd-dependent, measured); the endpoint always uses the clean base.")
    seed_file = res.pop("seed_file", None)
    # never leak seed values: seed file content stays in scratch
    res["seed_file_written"] = bool(seed_file)
    res["seed_digest"] = _seed_digest(seed)
    return _envelope(
        "nitrogenase",
        measured=["coherence", "energy_barrier_kj_mol",
                  "synthetic_catalyst_barrier_kj_mol", "hilbert_dim", "mse", "time_seconds"],
        interpretation=["energy barriers come from the FBSC fold model of the "
                        "electronic manifold; resonance boost is a model term",
                        "base_note: clean-cwd run (npz-independent)"],
        speculation=["synthetic catalyst feasibility score is a design-model "
                     "estimate, not an electrochemical measurement"],
        data=res, seed=seed,
        extra_provenance={"kind": "nitrogenase", "d": req.d,
                          "n_qudits": req.n_qudits},
        persist_kind="probe_battery")


def _handle_resonance(req: ResonanceRequest) -> Dict[str, Any]:
    RM = _battery("resonance")
    import numpy as np
    t0 = time.time()
    n_samples = int(req.fs * req.duration_s)
    t = np.arange(n_samples) / req.fs
    noise_power = 10 ** (req.noise_floor_db / 10)
    tone = req.amplitude * np.sin(2 * np.pi * req.tone_hz * t)
    noise = np.sqrt(noise_power) * np.random.RandomState(41).randn(n_samples)
    samples = tone + noise
    mon = RM(target_hz=req.tone_hz, fs=float(req.fs))
    window = int(req.fs)  # 1 s Welch windows
    states = []
    for i in range(0, n_samples - window + 1, window):
        states.append(mon.update(samples[i:i + window], fs=float(req.fs)))
    final = states[-1] if states else {"state": "NOISE"}
    data = {
        "target_hz": req.tone_hz, "fs": req.fs, "duration_s": req.duration_s,
        "samples_generated": n_samples,
        "windows": len(states),
        "final_state": final,
        "trigger_count": mon.trigger_count,
        "f0_window_states": [{"state": s["state"], "snr_db": s["snr_db"],
                              "f0_hz": round(s["f0_measured_hz"], 2),
                              "sustain_sec": s["sustain_sec"]} for s in states],
        "time_seconds": round(time.time() - t0, 3),
    }
    return _envelope(
        "resonance",
        measured=["snr_db", "f0_measured_hz", "sustain_sec", "state per window"],
        interpretation=["resonance_detected requires sustained target-band SNR "
                        "above floor (noise-gated, hysteresis, cooldown)"],
        speculation=["the 41.02 Hz bio-quantum 'sentience trigger' is a design "
                     "metaphor: detection is a signal event, NOT a consciousness "
                     "claim (no hardware, no bio claim)"],
        data=data, seed=None,
        extra_provenance={"kind": "resonance", "target_hz": req.tone_hz},
        persist_kind="probe_battery")


def _handle_runs(limit: int = 10) -> Dict[str, Any]:
    """List recent provenance rows (seed digest only)."""
    try:
        store = _get_store()
        store.init_db()
        rows = store.list_experiments(limit=max(1, min(limit, 200)))
    except Exception as e:  # DB-optional
        rows = []
    return _envelope(
        "runs",
        measured=[],
        interpretation=["provenance rows from repo-local SQLite"
                        if rows else "DB unavailable or empty (DB-optional)"],
        speculation=[],
        data={"limit": limit, "count": len(rows), "experiments": rows})


def _handle_health() -> Dict[str, Any]:
    mods = {
        "compression_specialist": (_REPO_ROOT / "compression_specialist.py").exists(),
        "likeness_probes": (_QL_DIR / "likeness_probes.py").exists(),
        "qudit_lhv_probe": (_QL_DIR / "qudit_lhv_probe.py").exists(),
        "quantum_topological_trainer": (_TR_DIR / "quantum_topological_trainer.py").exists(),
        "nitrogenase_qudit_simulator": (_NQ_DIR / "nitrogenase_qudit_simulator.py").exists(),
        "eeg_to_tonal_engine": (_REPO_ROOT / "src" / "quantum_llm" / "eeg_to_tonal_engine.py").exists(),
    }
    return _envelope(
        "health",
        measured=["battery module presence on disk"],
        interpretation=["endpoint list mirrors the owned R&D battery"],
        speculation=[],
        data={
            "battery_modules": mods,
            "endpoints": [
                "POST /api/rnd/fbsc",
                "POST /api/rnd/likeness",
                "POST /api/rnd/lhv_qudit",
                "POST /api/rnd/trainer",
                "POST /api/rnd/nitrogenase",
                "POST /api/rnd/resonance",
                "GET /api/rnd/runs?limit=N",
                "GET /api/rnd/health",
            ],
            "seed_policy": ("seed accepted per-request; ONLY a sha256 digest is "
                            "logged; raw seed is never stored or returned"),
            "db_policy": "repo-local SQLite (DB-optional); provenance only",
            "honest_ceiling": _HONEST_CEILING,
        })


# ---------------------------------------------------------------------------
# router / app
# ---------------------------------------------------------------------------
rnd_router = APIRouter(prefix="/api/rnd")

rnd_router.add_api_route("/health", _handle_health, methods=["GET"],
                         include_in_schema=True, tags=["rnd"])
rnd_router.add_api_route("/fbsc", _handle_fbsc, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/likeness", _handle_likeness, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/lhv_qudit", _handle_lhv_qudit, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/trainer", _handle_trainer, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/nitrogenase", _handle_nitrogenase, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/resonance", _handle_resonance, methods=["POST"], tags=["rnd"])
rnd_router.add_api_route("/runs", _handle_runs, methods=["GET"], tags=["rnd"])


def create_rnd_app() -> FastAPI:
    app = FastAPI(title="JARVIS R&D Battery API",
                  description="Owned probe battery wired into the JARVIS server; "
                              "exact classical simulation, no hardware claims.",
                  version="0.1.0")
    app.include_router(rnd_router)
    return app


app = create_rnd_app() if _FASTAPI_OK else None

if __name__ == "__main__" and _FASTAPI_OK:
    import uvicorn
    port = int(os.environ.get("JARVIS_RND_PORT", "8777"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
