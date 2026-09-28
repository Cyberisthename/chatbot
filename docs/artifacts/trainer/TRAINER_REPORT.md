# TRAINER_REPORT.md — quantum_topological_trainer (3-phase)

**Author:** agent-theoretical-physicist · 2026-09-20 (regenerated 2026-09-24)
**Artifacts:** `quantum_topological_trainer.py` (reproducible, pure numpy), `trainer_results.json`
**Tags:** **(a)** measured · **(b)** interpretation · **(c)** speculation. Reproduce with
`python3 quantum_topological_trainer.py`.

---

## 0. Bottom line (owner summary)

1. **(a)** All three phases are implemented and run on a **real** task — non-convex tanh
   regression — with measured losses, not theater. Wall-clock ~3.5 s, fully deterministic.
2. **(a) Phase 1 (noise immunity) PROVEN:** a braid-encoded weight matrix is **exactly** immune
   to isotopy noise (weight change **0.0** under 20 inserted σᵢσᵢ⁻¹ kinks, by the theorem
   σᵢσᵢ⁻¹ = identity), while a float weight matrix moves under any noise (change **1.6×10⁻³**).
   Crossing flips *do* change the braid weights (change 0.5) but are **detected** by the Burau
   invariant (distance 0.995).
3. **(a) Phase 2 (annealing) did NOT escape local minima** — it got stuck at loss **0.064**, worse
   than gradient descent. Braid-space optimization has **no smooth gradient**, so annealing is
   *harder*, not easier, than GD.
4. **(a) Phase 3 (evolution) found the exact optimum** (loss **0.0**), tying gradient descent.
5. **(a) GATE (honest):** the trainer **does NOT beat gradient descent** (tie at ~0 loss) and
   **does beat the plain seed optimizer** (0.0 vs 0.054). **No superiority claim over gradient
   descent is made.**
6. **(b) The deep finding:** there is a real **trade-off** — the braid encoding buys *noise
   immunity* (Phase 1) at the cost of *trainability* (Phases 2–3 lose their smooth landscape).
   Topological protection and gradient-free optimization are in tension, not synergy.

---

## 1. Task (real, differentiable-ish)

Non-convex tanh regression: `d=6`, `n=64`, `Ŷ = tanh(X·W)`, teacher `W*` = the Burau matrix of a
random braid word (so the braid space *contains* the optimum — a target, not an advantage).
Loss = MSE(Ŷ, Y). The teacher loss floor is 0 by construction.

| encoding | weight matrix W | search method |
|---|---|---|
| float (baseline a) | free ℝ⁶ˣ⁶ | analytic gradient descent |
| seed (baseline b) | Burau matrix of `word_from_seed(seed)` | Nelder–Mead over the 3-seed |
| **braid (trainer)** | Burau matrix of a braid word | annealing (P2) + evolution (P3) |

---

## 2. Phase 1 — braid-encoded weights (noise immunity)

**(a) Measured:**

| noise | braid weight change | float weight change |
|---|---|---|
| isotopy (20 σᵢσᵢ⁻¹ kinks) | **0.0** (exact) | — |
| Gaussian σ=1e-3 | — | 1.61×10⁻³ |
| crossing flips (30%) | 0.5 (invariant distance **0.995**) | — |

**(b)** The braid encoding gives a weight matrix whose *topological class* is exactly immune to
isotopy noise — σᵢσᵢ⁻¹ = identity *in the representation*, so the weights literally do not move
under deformation. A float matrix has no such protected structure: any perturbation moves it.
**(c)** The protection is specifically against **isotopy** (deformation), not against
**crossing flips** (which are the realistic "bit-flip" analogue) — those change the weights but
are *detected* by the invariant. This is the same mechanism as the anyon experiment
(`docs/artifacts/anyon/`): detect, don't correct.

---

## 3. Phase 2 — annealing loss landscape

**(a)** Simulated annealing over braid-space moves (crossing flip, generator change, kink
insert/delete), 2000 evals, exponential temperature schedule 2.0 → 10⁻³.

**Result: final loss 0.064318 — did NOT converge.** **(b)** The loss is a *discontinuous*
function of the braid word (one crossing flip jumps the Burau matrix, hence the weights, hence
the loss), so there is no smooth gradient for the Metropolis walk to follow; and kink moves are
*neutral* (they never change the weights), wasting a share of the proposals. The "annealing
escapes local minima that lock gradient descent" claim is **NOT supported** on this task — here
it is the *gradient* method that wins.

---

## 4. Phase 3 — evolutionary mutation loop

**(a)** Population 30, 70 generations (~2100 evals), mutation = crossing flips / generator
changes, truncation selection on loss.

**Result: final loss 0.0 — the population found the exact teacher braid word.** **(b)** Discrete
population search does not need a gradient; with enough diversity it can stumble onto the exact
optimum even on a discontinuous landscape. This is why evolution succeeds where annealing
struggles: selection pressure is global, not local.

---

## 5. GATE — vs the two baselines (equal-ish ~2000-eval budget)

| method | loss | evals | result |
|---|---|---|---|
| gradient descent (a) | **0.000000** | 2000 | baseline |
| seed optimizer (b) | 0.053649 | 170 | baseline |
| annealing (Phase 2) | 0.064318 | 2000 | stuck |
| **evolution (Phase 3)** | **0.000000** | 2100 | ties GD |

**(a) Verdict:** the trainer **ties gradient descent** (both 0.0) and **beats the seed
optimizer** (0.0 vs 0.054). It does **not** beat GD by any meaningful margin — so **no
superiority claim over gradient descent is made**, and the seed-optimizer win is against a
*baseline*, not against GD. The gate is **partially passed** (beats seed; ties GD).

---

## 6. Honest science section

* **No quantum hardware, no true quantum tunneling.** Annealing/evolution here are the
  *deterministic classical analogues*; "quantum tunneling" is not claimed.
* **Measured wall-clock epochs.** ~3.5 s total; evals counted; nothing "instant".
* **What is proven (a):** (1) braid-encoded weights are *exactly* isotopy-immune (by theorem),
  and the invariant *detects* crossing flips; (2) evolution finds the optimum on this task,
  tying GD; (3) the trainer beats the plain seed optimizer.
* **What is NOT proven (a/b):** (1) annealing does **not** escape local minima better than GD —
  the "quantum-tunneling advantage" is absent; (2) the trainer does **not** beat GD; (3) the
  noise immunity is to *isotopy*, a narrower class than arbitrary weight noise.
* **The real trade-off (b):** topological protection and gradient-based trainability are in
  tension. The braid encoding freezes the weight class against deformation, but the resulting
  loss landscape is discrete and discontinuous — exactly the regime where gradient descent
  works and annealing does not. A fair summary: **the topological trainer is a noise-robust
  *encoding*, not a faster *optimizer*.**
* **Reproducibility (a):** `python3 quantum_topological_trainer.py` regenerates every number
  (seed 2026); pure numpy/scipy; no proprietary code.

---

## 7. Next steps

1. **A task with many local minima** (e.g., a deeper MLP with saddle points) — the regime where
   annealing might actually beat GD, if it ever does.
2. **Weight the annealing move set** away from neutral kink moves (they waste evals) toward
   fitness-changing flips.
3. **Jones-polynomial weight invariant** instead of Burau, for a richer topological fingerprint.
4. **Hybrid** — use GD to *pre-train* float weights, then encode/round to the nearest braid word
   (post-hoc protection), rather than optimizing *in* braid space.

---

### Reproduce

```
cd docs/artifacts/trainer
python3 quantum_topological_trainer.py   # prints table, writes trainer_results.json
```
Dependencies: Python 3, `numpy`, `scipy`. No hardware.
