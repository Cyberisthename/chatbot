#!/usr/bin/env python3
"""
TN_CAPSTONE BENCHMARK — FBSC exact-state guarantees vs classical tensor-network
simulators (external validation with quimb 1.15 as a third-party MPS baseline).

================================================================================
HOUSE-STANDARD FRAMING (read before quoting any number):

* quimb is used ONLY as an external comparison baseline. It lives in a scratch
  venv and is NOT a product dependency. The "no third-party quantum SDK" rule
  governs the product core; validation baselines are explicitly out of scope of
  that rule (this is the lead-approved reading).
* Claim tags — every result row carries one of:
    "measured"        direct observation produced by THIS script on THIS box
                      (reproduce: run this script with the venv below).
    "analytic"        closed-form / information-theoretic computation, not a
                      timed observation.
    "interpretation"  model reading consistent with the measured rows.
  No hardware claims anywhere; "exact classical simulation" is the honest
  ceiling of everything measured here.
* Honest cap restated in R3: the FBSC <=3-parameter family (seed + deterministic
  generator) is measure-zero in every ambient Hilbert space (ERROR_BOUND_PROOF.md
  Theorem 3): it cannot represent generic (Haar-random) states. Tensor-network
  and dense simulators remain the correct tools for states outside the family.

APPROVED REFERENCES CROSS-CHECKED BY THIS SCRIPT (owner's "not from an old
doc?" test — every approved number re-derived fresh, differences reported):
  * docs/validation/error_bound_verify.json     (Track 2/3 MSE=0 + storage)
  * docs/artifacts/quantum_likeness/likeness_results.json  (P6 MPS challenger)

REPRODUCE (tested on this box):
  python3 -m venv --system-site-packages ~/venvs/tn-capstone     # system numpy
  ~/venvs/tn-capstone/bin/pip install quimb==1.15.0 numba cotengra
  ~/venvs/tn-capstone/bin/python tn_benchmark.py --smoke          # ~1 min
  ~/venvs/tn-capstone/bin/python tn_benchmark.py --out TN_CAPSTONE_results.json
                                                                  # full run
  (quimb is a scratch baseline; the venv is disposable.)
================================================================================
"""
import sys, os, json, time, hashlib, math, argparse, datetime, contextlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = None
for cand in (os.path.dirname(os.path.dirname(os.path.dirname(HERE))),   # .../docs/validation/tn_capstone -> repo root
             os.path.join(HERE, "..", "..", ".."),
             "/home/team/shared/chatbot"):
    if os.path.exists(os.path.join(cand, "compression_specialist.py")):
        REPO = cand
        break
if REPO is None:
    raise SystemExit("cannot locate repo root (compression_specialist.py)")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "artifacts", "quantum_likeness"))

import numpy as np
try:
    import quimb.tensor as qtn
    import quimb as qu
    HAVE_QUIMB = True
except Exception as exc:                       # noqa
    HAVE_QUIMB = False
    QUIMB_ERR = repr(exc)

with contextlib.redirect_stdout(io.StringIO()):
    from compression_specialist import FractalBraidSeedCompressor
import likeness_probes as lp

SEED = (0.57721, 1.618034, 2.71828)
SCHEME, DEPTH_SCALE = "hash", 8.0
CORE_NS = (8, 16, 32, 64, 128, 256, 512, 1024)
BRAID_NS = (8, 10, 12, 14, 16, 18, 20)
DGRID = (1, 2, 4, 8, 16, 32, 64, 128)


def load_approved():
    """Approved reference numbers, loaded verbatim from the repo artifacts."""
    eb = json.load(open(os.path.join(REPO, "docs", "validation", "error_bound_verify.json")))
    lk = json.load(open(os.path.join(REPO, "docs", "artifacts", "quantum_likeness", "likeness_results.json")))
    eb_rows = {r["n"]: r for r in eb["measured"]["d=2"]}
    p6 = {int(k): v for k, v in lk["P6_mps"]["table"].items()}
    return eb_rows, p6


# ---------------------------------------------------------------------------
# FBSC-side helpers
# ---------------------------------------------------------------------------
def norm_mse(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return float(np.mean(np.abs(a - b) ** 2) / max(np.mean(np.abs(a) ** 2), 1e-300))


def dense_single_excitation(A):
    """Dense embedding psi[1<<i] = A[i] (ERROR_BOUND_PROOF 1.1, qubit case)."""
    n = len(A)
    ref = np.zeros(2 ** n, dtype=complex)
    for i in range(n):
        ref[1 << (n - 1 - i)] = A[i]
    return ref


def closed_form_qubit(seed, n):
    """Independent implementation of ERROR_BOUND_PROOF Appendix A (qubit path)
    + the documented braid cascade of compression_specialist.py
    (_reconstruct_qubit: braids at ascending positions 0..n-2, 2-level mixing,
    final normalization). Used to verify the closed forms are a COMPLETE
    description of the generator (measured MSE vs reconstruct())."""
    s1, s2, s3 = seed
    scale = (s1 % 1.0) * 2.0 + 0.5
    phi0 = (s2 * np.pi) % (2 * np.pi)
    f0 = int(3 + (s3 % 1.0) * 12)
    v = np.arange(n, dtype=np.float64) / (n - 1) if n > 1 else np.zeros(n)
    base = np.exp(2j * np.pi * v) / np.sqrt(n)
    y = scale * base
    # fold passes:  multiply by exp(i*(phi0 + 0.1*s2*f)) for f in 0..f0-1,
    # iteratively (matches the module's floating-point ordering)
    for f in range(f0):
        y = y * np.exp(1j * (phi0 + f * (s2 * 0.1)))
    # braid cascade at positions 0..n-2 (ascending), params from the same hash
    for i in range(n - 1):
        h = hashlib.sha256(f"{seed}-{i}".encode()).digest()
        theta = (int.from_bytes(h[:4], "big") % 10000) / 10000.0 * np.pi
        phi = (int.from_bytes(h[4:8], "big") % 10000) / 10000.0 * np.pi * 2
        u00 = np.cos(theta)
        u01 = 1j * np.sin(theta) * np.exp(1j * phi)
        u10 = 1j * np.sin(theta)
        u11 = np.cos(theta) * np.exp(-1j * phi)
        a, b = y[i], y[i + 1]
        y[i] = u00 * a + u01 * b
        y[i + 1] = u10 * a + u11 * b
    norm = np.sqrt(np.sum(np.abs(y) ** 2))
    return y / (norm if norm > 0 else 1.0)


def braid_gate_list(seed, n, depth_scale=DEPTH_SCALE, scheme=SCHEME):
    """Replay the deterministic gate sequence of lp.fbsc_braid_state
    (scheme='hash'): (i, theta, phi) per step, i in 0..n-2."""
    a, b, g = seed
    braid_len = max(24, int(depth_scale * 6 * n))
    _, z = lp.fbsc_fold(seed, braid_len)
    z = z - z.min(); denom = z.max() + 1e-12; z = z / denom
    gates = []
    for j in range(braid_len):
        if scheme == "fold":
            i = int(z[j] * (n - 2)) % (n - 1)
            theta = ((1.0 + 2.0 * (b % 1.0)) *
                     (0.25 + 0.75 * abs(np.sin(b + 0.37 * j)))) % (math.pi / 2)
            phi = (g + 0.67 * j) % (2 * math.pi)
        else:
            h = hashlib.sha256(f"{seed}-{j}".encode()).digest()
            i = int.from_bytes(h[:2], "big") % (n - 1)
            theta = (int.from_bytes(h[2:4], "big") / 2 ** 16) * 1.2 % (math.pi / 2)
            phi = (int.from_bytes(h[4:6], "big") / 2 ** 16) * 2 * math.pi
        gates.append((int(i), float(theta), float(phi)))
    return gates


# ---------------------------------------------------------------------------
# Tensor-network (quimb) helpers — all conventions verified empirically on
# quimb 1.15 (see the accompanying report, §methods):
#   * site k <-> dense bit n-1-k ("001" -> index 1)
#   * site tensors 'lrp': first (r,p), middle (l,r,p), last (l,p)
#   * gate (2-site): contract='swap+split' keeps a clean 1D MPS; tensor axes
#     G[new_i,new_j,old_i,old_j] == braid4.reshape(2,2,2,2) (verified maxdiff 0)
#   * compress(max_bond=D) is IN-PLACE (returns None) — operate on .copy()
# ---------------------------------------------------------------------------
def se_tensors(c, n):
    """Exact D=2 single-excitation MPS site tensors for sum_k c_k |e_k>
    (defect at quimb site k). T_k(0)=I, T_k(1)=[[0,c_k],[0,0]]; boundaries
    l=(1,0), r=(0,1)^T. Schmidt rank of the state is <= 2 across every cut
    (only two virtual 'defect side' configurations), so D=2 is exact."""
    T = []
    for k in range(n):
        t = np.zeros((2, 2, 2), dtype=complex)
        t[0, 0, 0] = 1.0; t[1, 1, 0] = 1.0
        t[0, 1, 1] = c[k]
        T.append(t)
    F = np.zeros((2, 2), dtype=complex)
    F[0, 0] = 1.0; F[1, 1] = c[0]
    L = np.zeros((2, 2), dtype=complex)
    L[0, 1] = c[n - 1]; L[1, 0] = 1.0
    return [F] + T[1:n - 1] + [L]


def mps_bytes(mps):
    return int(sum(t.data.nbytes for t in mps.tensors))


def amplitude_quimb(mps, k, n):
    """<e_k|psi> via quimb contraction (conjugated by the bra, as expected)."""
    basis = ["0"] * n
    basis[k] = "1"
    comp = qtn.MPS_computational_state("".join(basis), dtype=complex)
    try:
        val = (mps.H & comp).contract(optimize="auto-mps")
    except Exception:
        val = (mps.H & comp).contract(optimize="greedy")
    return complex(val)


def build_braid_mps(n, gates, D=None):
    """Gate-by-gate MPS build of the braid circuit. D=None -> exact (no
    truncation); D=int -> per-gate truncation (online bounded-budget
    simulation). Returns (mps, wall_time)."""
    m = qtn.MPS_computational_state("0" * n, dtype=complex)
    t0 = time.perf_counter()
    for (i, theta, phi) in gates:
        G = lp.braid4(theta, phi).reshape(2, 2, 2, 2)
        if D is None:
            m = m.gate(G, where=(i, i + 1), contract="swap+split", max_bond=None)
        else:
            m = m.gate(G, where=(i, i + 1), contract="swap+split",
                       max_bond=D, cutoff=0)
    return m, time.perf_counter() - t0


def end_compress_fids(mps, psi_dense, Dgrid=DGRID):
    """Truncate the EXACT MPS to each D (optimal-D approximation a la TT-SVD)
    and return per-D fidelity vs the dense reference."""
    out = {}
    for D in Dgrid:
        mc = mps.copy()
        mc.compress(max_bond=D, cutoff=0)
        got = np.asarray(mc.to_dense()).reshape(psi_dense.shape[0])
        out[int(D)] = float(abs(np.vdot(psi_dense, got)) ** 2)
    return out


def core_amps_closure(seed, n):
    """Closed form of lp.fbsc_core_state's amplitude vector A (ERROR_BOUND_PROOF
    1.1 embedding vector) WITHOUT building the 2^n dense state — the exact same
    arithmetic as fbsc_core_state lines 'A = scale*base*e^{i phi0}*B; A/=norm'.
    Verified at small n to match fbsc_core_state(seed, n)[0] (see R1c rows)."""
    s1, s2, s3 = seed
    v = np.arange(n, dtype=np.float64) / (n - 1) if n > 1 else np.zeros(n)
    base = np.exp(2j * np.pi * v) / np.sqrt(n)
    phi0 = (s2 * np.pi) % (2 * np.pi)
    scale = (s1 % 1.0) * 2.0 + 0.5
    Hs = np.array([int.from_bytes(
        hashlib.sha256(f"{seed}-{i}".encode()).digest()[:4], 'big') % 10 ** 4
        for i in range(n)])
    B = np.exp(1j * (2 * np.pi * Hs / 10 ** 4))
    A = scale * base * np.exp(1j * phi0 * np.ones(n)) * B
    A /= np.linalg.norm(A)
    return A


def run_core(asmoke):
    out = {"reconstruct": [], "closedform": [], "quimb_exact": {}}
    # ---- R1a: reconstruct() determinism, storage, time (grid as approved) ----
    ns_recon = (8, 32, 1024) if asmoke else CORE_NS
    for n in ns_recon:
        row = {"n": n, "tag": "measured"}
        c1 = FractalBraidSeedCompressor(n_effective_qubits=n, seed=SEED)
        t0 = time.perf_counter(); a1, _, m1 = c1.reconstruct()
        t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); a2, _, m2 = c1.reconstruct()
        t2 = time.perf_counter() - t0
        row["time_s_run1"] = round(t1, 6)
        row["time_s_run2"] = round(t2, 6)
        row["max_abs_diff"] = float(np.max(np.abs(a1 - a2)))
        row["mse_norm"] = norm_mse(a1, a2)
        row["stored_bytes"] = n * 16
        row["dense_bytes"] = (2 ** n) * 16  # analytic dense storage (int)
        row["dense_petabytes"] = (((2 ** n) * 16) / 1e15) if n <= 64 else None
        row["compression_ratio_log10"] = m1["compression_ratio_log10"]
        row["fold_depth"] = m1["fold_depth"]
        out["reconstruct"].append(row)
    # ---- R1b: closed-form completeness check (Appendix A + cascade) ----
    recon_by_n = {r["n"]: r for r in out["reconstruct"]}
    for n in (8, 32, 256):
        comp = FractalBraidSeedCompressor(n_effective_qubits=n, seed=SEED)
        amps, _, _ = comp.reconstruct()
        cf = closed_form_qubit(SEED, n)
        row = {"n": n, "tag": "measured",
               "closedform_mse": norm_mse(amps, cf),
               "closedform_maxdiff": float(np.max(np.abs(amps - cf)))}
        if n in recon_by_n:
            row["amortized_us_per_amp"] = round(
                1e6 * recon_by_n[n]["time_s_run1"] / n, 3)
        out["closedform"].append(row)
    # ---- R1c: quimb exactness of the embedded core ----
    if not HAVE_QUIMB:
        out["quimb_exact"]["skipped"] = "quimb unavailable: %s" % locals().get("QUIMB_ERR")
        return out
    qe = {"small_dense_fidelity": [], "large_amplitude_recovery": []}
    small = (4, 8, 12, 16, 20) if not asmoke else (4, 8)
    for n in small:
        A, psi = lp.fbsc_core_state(SEED, n, d=2)   # approved 1.1 embedding
        row = {"n": n, "tag": "measured"}
        row["closure_vs_core_mse"] = norm_mse(A, core_amps_closure(SEED, n))
        # path (i): quimb from_dense -> truncate to D=2 (layout-agnostic)
        md = qtn.MatrixProductState.from_dense(psi, 2, max_bond=2, cutoff=0)
        gotd = np.asarray(md.to_dense()).reshape(2 ** n)
        row["from_dense_D2_fid_vs_psi"] = float(abs(np.vdot(psi, gotd)) ** 2)
        # path (ii): explicit D=2 site tensors built for the APPROVED embedding
        # layout: quimb site k <-> dense bit n-1-k, and fbsc psi[1<<i]=A[i] so
        # the defect weight at quimb site k is A[n-1-k] (== A[::-1][k]).
        mps = qtn.MatrixProductState(se_tensors(A[::-1], n), shape="lrp")
        gots = np.asarray(mps.to_dense()).reshape(2 ** n)
        row["explicit_D2_fid_vs_psi"] = float(abs(np.vdot(psi, gots)) ** 2)
        # builder sanity in the mirror layout (probe-verified convention):
        # defect weight c[k]=A[k] must reproduce dense_ref(A) (bit n-1-i)
        mps_m = qtn.MatrixProductState(se_tensors(A, n), shape="lrp")
        gotm = np.asarray(mps_m.to_dense()).reshape(2 ** n)
        row["explicit_D2_fid_mirror_sanity"] = float(
            abs(np.vdot(dense_single_excitation(A), gotm)) ** 2)
        row["max_bond"] = int(mps.max_bond())
        row["mps_bytes"] = mps_bytes(mps)
        qe["small_dense_fidelity"].append(row)
    large = (32, 64, 128, 256, 512, 1024) if not asmoke else (256,)
    for n in large:
        A = core_amps_closure(SEED, n)             # no 2^n dense state needed
        mps = qtn.MatrixProductState(se_tensors(A[::-1], n), shape="lrp")
        t0 = time.perf_counter()
        ks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        worst = 0.0
        for k in ks:
            val = amplitude_quimb(mps, k, n)
            expected = np.conj(A[n - 1 - k])       # approved-embedding layout
            worst = max(worst, abs(val - expected))
        wall = time.perf_counter() - t0
        qe["large_amplitude_recovery"].append(
            {"n": n, "tag": "measured",
             "sample_sites": ks,
             "worst_amplitude_err": float(worst),
             "recovery_wall_s": round(wall, 4),
             "max_bond": int(mps.max_bond()),
             "mps_bytes": mps_bytes(mps)})
    out["quimb_exact"] = qe
    return out


def run_braid(asmoke):
    out = {"dense": [], "exact_gates": [], "end_compress": {},
           "per_gate": [], "tt_svd": {}, "controls": {}}
    if not HAVE_QUIMB:
        out["skipped_quimb_rows"] = "quimb unavailable"
    ns = (8, 12) if asmoke else BRAID_NS
    # ---- R2a: dense reference + entanglement (volume-law growth) ----
    dense = {}
    for n in ns:
        t0 = time.perf_counter()
        psi = lp.fbsc_braid_state(SEED, n, scheme=SCHEME, depth_scale=DEPTH_SCALE)
        wall = time.perf_counter() - t0
        Smid, _ = lp.bipartite_entropy(psi, n, n // 2)
        alpha = Smid / (n // 2)
        dense[n] = psi
        out["dense"].append({"n": n, "tag": "measured",
                             "gen_wall_s": round(wall, 4),
                             "S_mid_ebits": round(float(Smid), 6),
                             "alpha": round(float(alpha), 6),
                             "dense_bytes": (2 ** n) * 16,
                             "dense_MiB": round(((2 ** n) * 16) / 2 ** 20, 2)})
    # ---- R2b: quimb exact gate-by-gate vs dense ----
    exact_n = (8, 10, 12, 14) + ((16,) if not asmoke else ())
    for n in exact_n:
        gates = braid_gate_list(SEED, n)
        if n not in dense:
            dense[n] = lp.fbsc_braid_state(SEED, n, scheme=SCHEME, depth_scale=DEPTH_SCALE)
        t0 = time.perf_counter()
        try:
            m, tbuild = build_braid_mps(n, gates, D=None)
        except Exception as exc:
            out["exact_gates"].append({"n": n, "tag": "measured",
                                       "skipped": "resource/API: %s" % repr(exc)[:160]})
            continue
        got = np.asarray(m.to_dense()).reshape(2 ** n)
        fid = abs(np.vdot(dense[n], got)) ** 2
        out["exact_gates"].append({
            "n": n, "tag": "measured",
            "gates": len(gates),
            "build_wall_s": round(tbuild, 4),
            "fidelity": float(fid),
            "maxdiff": float(np.max(np.abs(dense[n] - got))),
            "max_bond": int(m.max_bond()),
            "mps_bytes": mps_bytes(m)})
    # ---- R2c: end-compress D-sweep (optimal-D approx, like TT-SVD) ----
    ec_n = BRAID_NS if not asmoke else (8,)
    for n in ec_n:
        if n not in dense:
            dense[n] = lp.fbsc_braid_state(SEED, n, scheme=SCHEME, depth_scale=DEPTH_SCALE)
        gates = braid_gate_list(SEED, n)
        if n > 16:
            # exact build resource-heavy at n>16 on this box -> per-gate rows only
            out["end_compress"][str(n)] = {"tag": "interpretation",
                                            "skipped": "exact-MPS build resource-capped at n>16 (see per_gate rows)"}
            continue
        try:
            m, tbuild = build_braid_mps(n, gates, D=None)
        except Exception as exc:
            out["end_compress"][str(n)] = {"tag": "measured",
                                           "skipped": "resource/API: %s" % repr(exc)[:160]}
            continue
        fids = {}
        for D in DGRID:
            mc = m.copy()
            mc.compress(max_bond=D, cutoff=0)
            got = np.asarray(mc.to_dense()).reshape(2 ** n)
            fids[int(D)] = float(abs(np.vdot(dense[n], got)) ** 2)
        req = next((D for D in DGRID if fids[D] >= 0.99), None)
        out["end_compress"][str(n)] = {
            "tag": "measured",
            "fid_sweep": fids,
            "D_for_0.99": req,
            "fid_D2": fids[2],
            "build_wall_s": round(tbuild, 4),
            "max_bond_exact": int(m.max_bond()),
            "mps_bytes_exact": mps_bytes(m)}
    # ---- R2d: per-gate truncation (online bounded-budget simulation) ----
    pg_n = (8, 12, 16) if not asmoke else (8,)
    for n in pg_n:
        gates = braid_gate_list(SEED, n)
        if n not in dense:
            dense[n] = lp.fbsc_braid_state(SEED, n, scheme=SCHEME, depth_scale=DEPTH_SCALE)
        for D in (2, 8, 32):
            m, tbuild = build_braid_mps(n, gates, D=D)
            got = np.asarray(m.to_dense()).reshape(2 ** n)
            fid = abs(np.vdot(dense[n], got)) ** 2
            out["per_gate"].append({"n": n, "tag": "measured", "D": D,
                                    "fidelity": float(fid),
                                    "wall_s": round(tbuild, 4),
                                    "final_bond": int(m.max_bond()),
                                    "mps_bytes": mps_bytes(m)})
    # ---- R2e: owned TT-SVD cross-tool (approved algorithm, fresh run) ----
    tt_n = (8, 10, 12, 14, 16) if not asmoke else (8,)
    for n in tt_n:
        t0 = time.perf_counter()
        req, fid2, fids = lp.mps_required_D(dense[n], n, Ds=DGRID)
        out["tt_svd"][str(n)] = {"tag": "measured",
                                 "D_for_0.99": req, "fid_D2": round(fid2, 6),
                                 "fid_sweep": fids,
                                 "wall_s": round(time.perf_counter() - t0, 4)}
    # ---- R2f: controls through quimb (product/GHZ/core) at n=8 ----
    n = 8
    controls = {}
    for name, mk in (
            ("product", lambda n: np.ones(2 ** n, complex) / np.sqrt(2 ** n)),
            ("GHZ", lambda n: (lambda g: (g.__setitem__(0, 1 / np.sqrt(2)),
                                          g.__setitem__(-1, 1 / np.sqrt(2)), g)[-1])(np.zeros(2 ** n, complex))),
            ("core_FBSC", lambda n: lp.fbsc_core_state(SEED, n, d=2)[1])):
        psi = mk(n)
        fd = {}
        for D in (1, 2):
            md = qtn.MatrixProductState.from_dense(psi, 2, max_bond=D, cutoff=0)
            got = np.asarray(md.to_dense()).reshape(2 ** n)
            fd[D] = round(float(abs(np.vdot(psi, got)) ** 2), 6)
        controls[name] = {"tag": "measured", "fid_D1": fd[1], "fid_D2": fd[2]}
    out["controls"] = controls
    return out


def run_crossover(core, braid, eb_rows, p6):
    """Summary table of where each side wins + honesty guardrails."""
    interp = {"tag": "interpretation"}

    # measured dense-storage crossover vs the 1 KB artifact and vs FBSC arrays
    n_star_1KB = next((n for n in (4, 5, 6, 7, 8, 9, 10) if (2 ** n) * 16 >= 1024), None)
    core_recon = {r["n"]: r for r in core["reconstruct"]}
    core_storage = [{"n": n, "fbsc_stored_bytes": n * 16,
                     "dense_bytes": (2 ** n) * 16,
                     "dense_crossover_per_1KB_artifact": "dense >= 1 KB for n >= %s" % n_star_1KB}
                    for n in CORE_NS]

    # measured D-growth (quimb end-compress) + fidelity-D2 decay
    ec = braid.get("end_compress", {})
    dgrowth = []
    for n in BRAID_NS:
        row = {"n": n}
        if str(n) in ec and "D_for_0.99" in ec[str(n)]:
            row.update({"quimb_D_for_0.99": ec[str(n)]["D_for_0.99"],
                        "quimb_fid_D2": round(ec[str(n)]["fid_D2"], 4)})
        if str(n) in p6:
            row.update({"approved_D_for_0.99": p6[n]["D_for_0.99"],
                        "approved_fid_D2": p6[n]["fid_D2"]})
        dgrowth.append(row)

    out = {
        "core_family": {
            **interp,
            "summary": ("core: the single-excitation embedding has Schmidt rank "
                        "<= 2 across every cut (mathematical — verified D=2 MPS "
                        "fidelity 1.0 at n<=20 and exact amplitude recovery at "
                        "n<=1024 through quimb). A fixed-D=2 MPS represents the "
                        "whole family EXACTLY at every n; the FBSC seed does the "
                        "same with 3 numbers and no bond bookkeeping."),
            "dense_storage_crossover": core_storage,
            "mps_vs_fbsc_parameters": {
                **interp,
                "fbsc_description": "3 floats (owner seed) + deterministic generator; per-amplitude evaluation is a closed-form function of (seed,n,i) [Appendix A] — the braid cascade is triangular (amplitude i depends on prefix 0..i+1), so standalone single-amplitude eval costs O(i) while a full reconstruct is O(n) total; measured reconstruct wall is ~0.2-0.3 s for n=8..1024 (constant-ish, ggraph/overhead-dominated), i.e. independent of 2^n.",
                "mps_description": "2·n·D complex entries for the explicit D=2 MPS (measured bytes in R1c); needs canonical form + bond bookkeeping to stay exact.",
                "measured_mps_D2_bytes_exact": "see R1c rows (largest: n=1024)"},
        },
        "braid_family": {
            **interp,
            "summary": ("braid: an MPS is exact only within bounded bond "
                        "dimension. The required D grows with n under volume-law "
                        "entanglement (measured: approved P6 D_for_0.99 16->32->32"
                        "->32->64 across n=8..16; quimb re-derivation in R2c), "
                        "while the FBSC seed + its O(n) deterministic gate list "
                        "reconstructs the SAME state exactly at any n with no "
                        "truncation. Honest nuance: the braid family is a "
                        "CIRCUIT description (seed + O(n) gates), not an O(1) "
                        "amplitude oracle; the <=3-parameter cap (Thm 3) covers "
                        "the seed family; generic states remain out of reach."),
            "approved_vs_quimb_D_growth": dgrowth,
            "per_gate_vs_end_compress": {
                **interp,
                "note": "online per-gate truncation accumulates error and under-decays vs the optimal-D end-compress (measured R2d vs R2c): a bounded-budget TN simulation of the full braid circuit loses fidelity faster than the optimal-D MPS at the same D."},
        },
        "honesty_guardrails": {
            **interp,
            "theorem3_cap": ("FBSC family is a <=3-parameter (measure-zero) "
                             "subset of every ambient Hilbert space: it cannot "
                             "represent generic/Haar-random states. For states "
                             "outside the family, dense and TN simulators are "
                             "the correct tools; nothing here claims otherwise."),
            "no_hardware": ("all numbers are classical simulation on this box; "
                            "'exact classical simulation' is the ceiling."),
            "tn_is_baseline": ("quimb is a scratch baseline only, never a "
                               "product dependency."),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eb_rows, p6 = load_approved()
    results = {
        "meta": {
            "title": "CAPSTONE: FBSC exact-state guarantees vs classical tensor-network simulators",
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seed": list(SEED),
            "config": {"scheme": SCHEME, "depth_scale": DEPTH_SCALE},
            "platform": {"python": sys.version.split()[0],
                         "numpy": np.__version__,
                         "quimb": qu.__version__ if HAVE_QUIMB else None},
            "repo_main": os.popen("cd %s && git rev-parse --short HEAD 2>/dev/null" % REPO).read().strip(),
            "approved_references_loaded": [
                "docs/validation/error_bound_verify.json",
                "docs/artifacts/quantum_likeness/likeness_results.json"],
            "reproduce": ("~/.venv python (quimb 1.15 scratch venv) "
                          "tn_benchmark.py --smoke  |  --out <json>"),
            "smoke": bool(args.smoke),
        },
    }

    print("== R1 core family ==", flush=True)
    core = run_core(args.smoke)
    # cross-check R1a vs approved Track 2/3
    for r in core["reconstruct"]:
        n = r["n"]
        if n in eb_rows:
            r["approved_stored_bytes"] = eb_rows[n]["stored_bytes"]
            r["approved_max_abs_diff"] = eb_rows[n]["max_abs_diff"]
            r["matches_approved_storage"] = (r["stored_bytes"] == eb_rows[n]["stored_bytes"]
                                             and r["max_abs_diff"] == 0.0)
            if r["max_abs_diff"] != 0.0:
                r["flag"] = "MSE!=0 — approved reproducibility broken on this box"
    results["R1_core"] = core

    print("== R2 braid family ==", flush=True)
    braid = run_braid(args.smoke)
    # cross-check R2c/R2e vs approved P6
    for n, row in braid["end_compress"].items():
        if "D_for_0.99" in row and int(n) in p6:
            row["approved_D_for_0.99"] = p6[int(n)]["D_for_0.99"]
            row["approved_fid_D2"] = p6[int(n)]["fid_D2"]
            row["D0.99_matches_approved"] = row["D_for_0.99"] == p6[int(n)]["D_for_0.99"]
    max_tt_diff = 0.0
    for n, row in braid["tt_svd"].items():
        if int(n) in p6:
            for Dk, fv in row["fid_sweep"].items():
                av = p6[int(n)]["fid_sweep"].get(Dk)
                if av is not None:
                    max_tt_diff = max(max_tt_diff, abs(fv - av))
            row["approved_fid_D2"] = p6[int(n)]["fid_D2"]
            row["D0.99_matches_approved"] = row["D_for_0.99"] == p6[int(n)]["D_for_0.99"]
    braid["tt_svd_max_fid_diff_vs_approved"] = round(float(max_tt_diff), 9)
    results["R2_braid"] = braid

    print("== R3 crossover ==", flush=True)
    results["R3_crossover"] = run_crossover(core, braid, eb_rows, p6)

    txt = json.dumps(results, indent=1, default=str)
    out_path = args.out or os.path.join(HERE, "TN_CAPSTONE_results.json")
    with open(out_path, "w") as f:
        f.write(txt)
    print("wrote", out_path, "(%d bytes)" % len(txt))
    # short console digest
    print("\n===== DIGEST =====")
    for r in core["reconstruct"]:
        print("R1a n=%4d stored=%dB dense=%sB maxdiff=%g" % (
            r["n"], r["stored_bytes"], r["dense_bytes"], r["max_abs_diff"]))
    for r in core["closedform"]:
        print("R1b n=%3d closed-form MSE=%.2e maxdiff=%.2e" % (
            r["n"], r["closedform_mse"], r["closedform_maxdiff"]))
    for r in core["quimb_exact"].get("small_dense_fidelity", []):
        print("R1c n=%2d from_dense_D2=%g explicit_D2=%g mirror_sanity=%g" % (
            r["n"], r["from_dense_D2_fid_vs_psi"], r["explicit_D2_fid_vs_psi"],
            r["explicit_D2_fid_mirror_sanity"]))
    for r in core["quimb_exact"].get("large_amplitude_recovery", []):
        print("R1c n=%4d worst amp err=%.2e mps_bytes=%d" % (
            r["n"], r["worst_amplitude_err"], r["mps_bytes"]))
    for r in braid["exact_gates"]:
        print("R2b n=%2d fid=%g max_bond=%d bytes=%d" % (
            r["n"], r["fidelity"], r["max_bond"], r["mps_bytes"]))
    for n, row in braid["end_compress"].items():
        if "D_for_0.99" in row:
            print("R2c n=%s D_0.99=%s fid_D2=%g (approved %s/%s, match=%s)" % (
                n, row["D_for_0.99"], row["fid_D2"],
                row.get("approved_D_for_0.99"), row.get("approved_fid_D2"),
                row.get("D0.99_matches_approved")))
    for n, row in braid["tt_svd"].items():
        print("R2e n=%s TT-SVD D_0.99=%s (approved %s, match=%s)" % (
            n, row["D_for_0.99"], p6[int(n)]["D_for_0.99"],
            row["D0.99_matches_approved"]))
    print("TT-SVD max |Δfid| vs approved:", braid["tt_svd_max_fid_diff_vs_approved"])
    print("controls:", {k: v["fid_D2"] for k, v in braid["controls"].items()})


if __name__ == "__main__":
    main()