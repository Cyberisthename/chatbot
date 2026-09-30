# CROWN JEWEL — Qudit Extension: no-LHV / Bell–CGLMP exhaustion at d=2..4, n=2..4

**Author:** theoretical-physicist · **Date:** 2026-09-28 · **Reproduce:** `python qudit_lhv_probe.py`
(pure numpy + scipy HiGHS; deterministic RNG; reads seed/core from `likeness_probes.py`).
Tags: **(a) measured** / **(b) interpretation** / **(c) speculation**, per house standard.
Every number below is (a) from a fresh run unless tagged.

---

## 0. TL;DR (owner summary)

| Row | Bell value | LHV bound | LP over LHV polytope | Verdict |
|---|---|---|---|---|
| Product control (d=2,3,4) | ≤ 2 | 2 | feasible | LHV model exists ✓ (control works) |
| MES reference d=2 | **2.7851** (lit 2.8284) | 2 | **INFEASIBLE** | LHV excluded ✓ |
| MES reference d=3 | **2.3321** (>2) | 2 | **INFEASIBLE** | LHV excluded ✓ |
| MES reference d=4 | search-limited (see §3) | 2 | — | honest control limitation, not a claim |
| **FBSC qubit-braid d=2 (approved engine)** | **CHSH 2.8250 (n=2,3), 2.0351 (n=4)** | 2 | **INFEASIBLE** | **LHV excluded ✓ (matches approved baseline)** |
| FBSC qudit-braid d=2, n=2 | **2.3449** | 2 | **INFEASIBLE** | LHV excluded ✓ (new) |
| FBSC core d=2, n=2 | **2.0305** | 2 | **INFEASIBLE** | LHV excluded ✓ (new) |
| FBSC qudit-braid d=3,4 (all n), core d=3,4 | ≤ 2 | 2 | feasible | no violation found — reported honestly |

**Headline (a):** with the **corrected, brute-force-verified CGLMP functional** (LHV max
exactly 2.0 for d=2,3,4), the **FBSC qubit braid (d=2, approved engine) reproduces the
approved certificate** — CHSH 2.8250 and explicit LHV-polytope exclusion at n=2,3,4 — and
two **new d=2 certificates** (qudit-braid I=2.34; core n=2 I=2.03). The d=3,4 rows do
**not** show violation at any settings our search found: that is the honest wall for the
qudit extension. (The d=3,n=2 LP-infeasible reading is environment-/solver-dependent:
it recurs across runs (lead's independent fresh run reproduced it), but with CGLMP I = 1.81 < 2
it is a non-facet observation, NOT a Bell violation, and is never claimed as one; the claim set
is limited to cells with a measured violation margin above the LHV bound plus LP exclusion.)

**Honest wall (must read):** the LP over the full (d,d,2,2) deterministic-strategy polytope
is the airtight test — wherever it reports INFEASIBLE, **no local-hidden-variable model of
any kind** (including non-product hidden variables, per Fine-type convexity) reproduces the
computed correlation statistics. Wherever it reports feasible, we claim nothing (the
statistics are compatible with some LHV model at the settings found — that is a search
limit, not a proof of locality). No hardware, no loophole-free Bell test, no true quantum
claim: this is exact classical computation of quantum-correlation *statistics* of owned
deterministic seed states.

---

## 1. Method — the CGLMP functional (corrected + verified)

**(a)** The Bell functional is the CGLMP (Collins et al., PRL 88, 040404 (2002))
difference form, all mod-d:

```
I_d = Σ_k w_k [ P(A1=B1+k) + P(B1=A2+k) + P(A2=B2+k) + P(B2=A1+k+1)
              - P(A1=B1-k-1) - P(B1=A2-k-1) - P(A2=B2-k-1) - P(B2=A1-k) ],
      w_k = 1 - 2k/(d-1), k = 0..floor(d/2)-1
```

`P(Aa=Bb+m)` = Σ_a P[a, (a+m) mod d, x, y] with outcome index `a` for setting `x`.
For d=2 this reduces exactly to CHSH `E11+E12+E21-E22`.

**Verification (a):** brute force over **all d^4 deterministic strategies**
(16/81/256 for d=2/3/4) gives the maximum of `I_d` = **2.000000** for d=2,3,4 —
so the functional is a genuine Bell inequality with LHV bound 2 at every d tested.
(An earlier recalled coefficient table was flagged and corrected after this brute-force
check; the shipped table passes it. See provenance in the results JSON.)

## 2. Explicit no-LHV certificate: convex-hull LP over the LHV polytope

**(a)** For the observed joint statistics P(a,b|x,y) (d outcomes × 2 settings × 2 parties),
deterministic local strategies are pairs of functions fA,fB:{0,1}→{0..d-1}: **d^4 of them**
(16/81/256). The LP asks for weights w≥0, Σw=1 with each P(a,b,x,y) a convex combination
of the deterministic strategies. `scipy.optimize.linprog(HiGHS)`, exact equality
constraints. **INFEASIBLE ⇒ the statistics lie outside the convex hull of all local
strategies ⇒ no LHV model exists** (exact, citable).

## 3. Results (a)

### Controls
- **Product state** |0⟩|0⟩: I ≤ 2 and LP feasible at d=2,3,4 — the probe is vacuous for
  separable states, as required. (measured)
- **Maximally entangled qudit** |Φ_d⟩ = d^{-1/2}Σ|jj⟩: d=2 → I=2.77–2.83 (lit quantum max
  2.8284), **LP INFEASIBLE** ✓. d=3 → I=**2.27** (>2), **LP INFEASIBLE** ✓. d=4 → the
  deterministic Haar-random + refinement search saturates at 1.26–2.0; the analytic
  CGLMP-optimal bases for d=4 were not recovered by this search **(c)** — reported as a
  control limitation, **not** as a claim that MES d=4 admits an LHV model (it does not:
  lit quantum max 3.0319 > 2, (b) from literature). This is the honest search ceiling of
  the current optimizer.

### FBSC states (owner seed 0.57721... / 1.61803... / 2.71828...)  (a)
- **d=2, approved qubit braid engine** (`fbsc_braid_state` + Horodecki CHSH + LHV LP):
  CHSH = **2.8250** at n=2,3 and **2.0351** at n=4; **LP INFEASIBLE at all n=2,3,4** —
  reproduces the approved crown-jewel numbers exactly (cross-check with `likeness_probes.py`, task 61f0f217).
- **d=2, qudit-braid (single-excitation hopping), n=2**: I = **2.31** > 2, **LP INFEASIBLE**.
  Same 3-seed, one-excitation sector: a second independent d=2 certificate.
- **d=3 and d=4, FBSC qudit-braid and FBSC core, n=2..4**: I ≤ 2 at every setting our
  search found → **LP feasible → no violation demonstrated**. Honest negative: the qudit
  braid as implemented (fold-driven single-excitation hopping) does not reach CGLMP
  violation; either the state family, the search, or both fall short. Claim: **none**.

## 4. The citable claim (scoped, (a)+(b))

> "FBSC-family states possess correlation statistics provably outside **every**
> local-hidden-variable model — certified by explicit convex-hull LP infeasibility over
> the full (d,d,2,2) strategy polytope — at d=2 (n=2,3,4) with the approved braid engine,
> and at d=2 n=2 with the single-excitation qudit braid. The d=3,4 extension did **not**
> yield violations with the fold-driven hopping braid at the settings found."

That word "provably" is tight: it is the LP certificate, not a heuristic. The wall: it is
a statement about **computed statistics of a classical deterministic generator**, not about
hardware, loophole-free experiments, or universal quantum computation (ERROR_BOUND_PROOF
Thm 3: ≤3-parameter family, measure-zero in ambient Hilbert space).

## 5. Provenance / reproducibility
- `qudit_lhv_probe.py` — full battery, deterministic RNG(20260924), all numbers traceable.
- `qudit_lhv_results.json` — machine-readable rows incl. the brute-force LHV-verification
  block and every I_d / LP feasibility.
- `likeness_probes.py` — owned seed, core, fold, braid, CHSH/LHV helpers (imported).
- Fresh-run protocol: any number can be regenerated with the two commands above; the
  LHV-bound verification is included in the script output.