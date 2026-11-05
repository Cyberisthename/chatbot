# ibm_fez run-001 — the first real QPU data, recorded at its true confidence

**Status: `OWNER-REPORTED`. Nothing in this directory is `[measured]`.**

The owner ran two blocks on IBM `ibm_fez`, qubits 44 and 45. The numbers came to us as prose
in a written brief, with no job IDs, no job JSON, no shot counts, no measurement settings and no
calibration record. That is enough to record the results and to do real analysis on them. It is
not enough to call them measured, and the difference is not pedantry — the two numbers that
matter most here (shots and settings) are exactly the two a reviewer will ask for first.

- Machine-readable record: [`provenance.json`](provenance.json) — every field, with `null` where
  the owner did not supply it and a `promotion_rule` listing what closes the gap.
- Every derived number below: [`analysis.py`](analysis.py) → [`analysis_output.json`](analysis_output.json).
  Standard library only. No numpy, no qiskit, no vendor SDK, nothing that could become a
  dependency on the owned-core boundary.

**Promotion rule.** These results become `[measured]` when the job IDs, the job JSON for both
blocks, the shot counts, the four CHSH settings and the covering calibration record are attached.
Not one of those is here. Until then the tag stays `OWNER-REPORTED` and no file outside this
directory may cite these numbers in any other form.

| # | Result | Status | One line |
|---|---|---|---|
| R1 | CHSH = **2.61 ± 0.02**, qubits 44/45, n = 2 | `[reported]` | standard Bell physics on a prepared Bell state; not new physics, not loophole-free, and at n = 2 not attributable to our core |
| R2 | frequency deviation **1.85 → 1.50 kHz** over a 20-min no-recalibration wait | `[reported]` | **the more valuable result**: the drift is autonomous, so a calibration timestamp does not bound a device's validity |
| R3 | nitrogen chemistry check | `OWED` | no clear result; nothing to cite, recorded so the gap is visible |
| — | spend: 39 s CHSH, 55 s total, over a 30 s cap | `[reported]` | the cap is enforced by estimate, not by the runtime; §5 |

---

## 1. R1 — CHSH 2.61 ± 0.02: say plainly what it is

### 1.1 The plain statement, first

**This is standard Bell physics on a state we prepared on purpose. It is not new physics.** A
CHSH value above 2 says a two-qubit state is entangled and Bell-nonlocal at the measured
settings. That was established in 1964–1982 and has been measured thousands of times on
superconducting hardware. Nothing about the number is a discovery, and nothing about it is
specific to this project.

**It is not loophole-free, and on this hardware it cannot be.** Qubits 44 and 45 are adjacent
transmons on one chip, hundreds of micrometres apart, read out by the same control electronics
within the same job. There is no spacelike separation, so the locality loophole is wide open;
detectors are not high-efficiency and the fair-sampling assumption is not defended, so the
detection loophole is open too; and the measurement settings were fixed in software before
submission, so there is no freedom-of-choice argument either. A correct name for this experiment
is a **CHSH *value* on a prepared state**, not a CHSH *test*. The repo's working agreement already
says "no hardware claims — exact classical simulation is the ceiling"; this result does not move
that ceiling and does not test it.

**At n = 2 it is possibly not distinguishable from a generic Bell state — and in fact it is not.**
This is the sharpest limitation and it deserves the arithmetic:

- The E2 circuit in `docs/QPU_NITROGENASE_EXPERIMENTS.md` is `H(q0)`, `CNOT(q0,q1)`. That prepares
  the textbook `|Φ⁺⟩` Bell state. **The state on the device was not the FBSC state.** So this
  measurement cannot carry information about the FBSC generator, at n = 2 or anywhere else.
- For any two-qubit state, the maximum CHSH value depends only on the two largest singular values
  `t₁, t₂` of the 3×3 correlation matrix: `S = 2√(t₁² + t₂²)`. A 9-entry matrix is compressed to
  one number. Two-qubit pure states modulo local unitaries have exactly **one** parameter, and
  `S` determines it. So `S` identifies a Bell state's entanglement and nothing about its origin.
  **Any source producing a Bell state at 92.3 % visibility yields 2.61.**
- The degeneracy is worse than "one parameter". A pure rotation of the measurement frame by
  **22.66°** reproduces `S = 2.61` with *no* visibility loss whatsoever (from
  `2√2·cos δ = 2.61`). `S` alone cannot separate "the state decohered" from "the settings were
  off by 23°". Those are different failures with different fixes, and one number cannot tell
  them apart. §4.2 proposes the 9-setting measurement that can.

The honest one-sentence claim R1 earns: *the settings our LHV-exclusion search returns are
physically realisable, and on this device they yield S = 2.61 ± 0.02 — a measured Tsirelson
deficit of 0.218, or 7.7 %.* That upgrades presentation honesty. It does not upgrade any claim
about the core.

### 1.2 The arithmetic

| quantity | value | how |
|---|---|---|
| S measured | 2.61 ± 0.02 | `[reported]` |
| classical (LHV) bound | 2.0 | exact |
| Tsirelson bound 2√2 | 2.828427 | exact |
| our design value | 2.8250 | `likeness_results.json` → `P5_no_lhv.n2.chsh` |
| violation significance | **30.5 σ** above 2 | `(2.61−2)/0.02` |
| deficit from Tsirelson | **0.2184** (7.72 %) | `2√2 − 2.61` |
| deficit from our design | **0.2150** = **10.75 σ** | `(2.825 − 2.61)/0.02` |

The last row is the substantive one. **The hardware result is not statistically consistent with
our own design value** — it is 10.75 σ low at the stated error. So the gap is not shot noise: it
is the channel, which is exactly what E2 was designed to expose. That part worked.

### 1.3 It does not clear our own pre-registration

`docs/QPU_NITROGENASE_EXPERIMENTS.md` §3 (E2) pre-registered a two-leg success: `S ≥ 2.60`
**and** `|S − S_model(channel)| ≤ 0.08`, with `2.00 < S < 2.60` pre-declared "inconclusive, not a
success".

- **Leg 1: marginal pass.** 2.61 clears 2.60 by **0.01 — half the stated error bar.** The 95 %
  lower bound is 2.571, which sits *inside* the band the document pre-declared inconclusive.
- **Leg 2: unevaluable.** `S_model` requires E1's *measured* channel parameters. **E1 never ran.**
  So the comparison that would tell us whether our noise model is right cannot be made.
- **No failure threshold was triggered** (that needs `S ≤ 2.00`).

**Verdict: not a pre-registered success.** The honest record is "leg 1 marginal, leg 2
unevaluable", and a motivated reading would be wrong to upgrade it. This is the second time the
pre-registration has saved us from a story — the first was `docs/validation/BLIND_PREDICTION_2.md`.

One caveat that makes the ±0.02 *generous*: the owner did not say whether it is statistical-only.
Every σ in this document therefore assumes statistical-only, which makes each stated significance
an **upper bound**. With readout and setting systematics folded in, the real error is larger and
the "30.5 σ" is looser than it looks.

### 1.4 A real inconsistency: 98 % fidelity and S = 2.61 do not fit together

The owner reports the state was "~98 % faithful to design". Take that at face value, treat the
state as a Werner state (visibility `v`), and the numbers refuse to agree:

| step | value | formula |
|---|---|---|
| CHSH-implied visibility | **v = 0.9228** | `S / 2√2` |
| CHSH-implied state fidelity | **F = 0.9421** | `(1+3v)/4` — so **7.7 % white noise** |
| if F really were 0.98 | v = 0.9733 → **S would be 2.753** | `(4F−1)/3`, then `2√2·v` |
| observed gap | **0.143 = 7.15 σ** | |
| correlator attenuation needed | 0.9481 | `2.61 / 2.753` |
| **implied per-qubit readout bit-flip error** | **1.32 %** | `(1−√0.9481)/2` |

A ~1.3 % per-qubit readout error is entirely typical for a fixed-frequency transmon and would
account for the whole residual, cleanly: readout bit-flips attenuate a correlator by `(1−2e)²`,
so 1.3 % per qubit costs `2√2 → 2.61` on its own. That is a **coherent and checkable**
explanation, and it is the reading I would bet on.

But it is `[interpretation]`, not a result, because it rests on two things that are not
established: that "98 % faithful" is a Werner-equivalent *state* fidelity (it could be a CNOT
process fidelity, or one minus trace distance — different quantities with different Werner
mappings), and that the residual is symmetric readout error rather than a gate error or a
setting error. **The measurement that settles it costs four circuits** (§4.2). Until then, the
inconsistency stands as an open defect in the reported result set: *as reported*, R1's two numbers
are 7 σ apart under the simplest noise model.

### 1.5 And a finding from our own artifact, not from the hardware

`likeness_results.json` `P5_no_lhv` reports `n2.chsh = 2.825`, `n3.chsh = 2.825` and
`n3.mermin3 = 1.9315`. The Mermin-3 combination has **LHV bound 2 and GHZ value 4**
(`mermin_max_3q` docstring, `likeness_probes.py`). **1.9315 < 2 — our n = 3 state does not violate
the Mermin inequality.** And the n = 3 CHSH value is identical to the n = 2 value because CHSH is
a bipartite quantity evaluated on a bipartition.

So: **the n = 3 row of our quantum-likeness table is not independent evidence of multipartite
nonlocality.** The only multipartite witness we computed fails, and the n = 3 CHSH row is the n = 2
physics wearing a different label. That does not weaken the published `[measured]` statement
(which is about CHSH values and LP exclusion, both true), but it does mean the table reads as
three rows of evidence where it is one row of evidence measured twice. Flagged for the lead; it
belongs in the truth sweep, not in this PR's scope.

---

## 2. R2 — the 1.85 → 1.50 kHz drift is the more valuable finding

The lead is right that this is the more valuable result, and the reason is methodological rather
than physical. **The owner ran a control.** They waited 20 minutes, confirmed IBM did not
recalibrate, re-ran, and found the qubit frequency had moved anyway. That single control retires
the most common excuse for a failed replication — "the backend got recalibrated between your
runs" — and it converts a nuisance observation into a *measured property of the device*: the
drift is **autonomous**.

**Reading assumed, flag for confirmation.** I have read "1.85 → 1.50 kHz" as the magnitude of the
qubit's frequency *deviation from its calibrated value*. The alternative reading — a drift *rate*
in kHz per unspecified time unit — is not analysable, because a rate without a denominator is not
a quantity. Everything below depends on this reading; if it is wrong, R2 must be restated before
it is cited anywhere. Signs were not reported, which costs us one bit and matters (§2.1).

### 2.1 The rate is a bracket, not a number

Two magnitude points 1200 s apart cannot distinguish *drifting back toward the calibrated value*
from *drifting through it*:

| reading | rate | per hour | relative/hour at 5 GHz |
|---|---|---|---|
| toward the calibrated value (slowest consistent) | **0.292 Hz/s** | 1050 Hz/h | 2.1 × 10⁻⁷ |
| through it (fastest consistent) | **2.792 Hz/s** | 10050 Hz/h | 2.0 × 10⁻⁶ |

Both are reported everywhere below. The spread is a factor of 9.6 and it is entirely a
consequence of the missing signs — one extra reported bit (`1.85 → 1.50` *or* `+1.85 → −1.50`)
would collapse it.

### 2.2 What the drift does NOT hurt — and this is most of the list

`[analytic]`, all of it re-runnable in `analysis.py`.

**Gate angles: nothing.** A drive detuned by `δ` from the qubit gives `Ω′ = √(Ω² + δ²)`, so the
angle error per gate is `≈ (δ/Ω)²/2`. At `δ/2π = 1.5–1.85 kHz` on a transmon drive
`Ω/2π = 5–20 MHz`:

| drive rate | angle error per gate at 1.50 kHz | at 1.85 kHz |
|---|---|---|
| 5 MHz | 4.5 × 10⁻⁸ | 6.8 × 10⁻⁸ |
| 10 MHz | 1.1 × 10⁻⁸ | 1.7 × 10⁻⁸ |
| 20 MHz | 2.8 × 10⁻⁹ | 4.3 × 10⁻⁹ |

A part in 10⁸. You could stack ten million gates and not see it. **Gate-only circuits are immune
to this drift mode**, and E3 and E5 are gate-only circuits.

**T1: nothing.** T1 is a Z-basis population measurement. A frequency offset rotates the frame; it
does not change `|1⟩` population. T1 is first-order insensitive.

**T2 Hahn echo: nothing, by design.** Refocusing a *static* detuning is precisely what an echo
does — `+φ` in the first half, π pulse, `−φ` in the second half, net zero. The only part that
survives is the part that *changes during the sequence*, and that residual is
`φ_res = (π/2)·r·τ²` (derived and verified numerically to 2 × 10⁻⁵ relative in `analysis.py`):

| delay τ | rate 0.292 Hz/s | rate 2.792 Hz/s |
|---|---|---|
| 150 µs | 1.0 × 10⁻⁸ rad | 9.9 × 10⁻⁸ rad |
| 300 µs | 4.1 × 10⁻⁸ rad | 3.9 × 10⁻⁷ rad |

At worst 4 × 10⁻⁷ rad — six orders below any measurable phase. **The drift cannot bias E1's T2
echo.** This is worth stating loudly because the intuitive worry ("the frequency is moving, so
the coherence measurement must be compromised") is wrong, and the reason it is wrong is the
reason echo exists.

**Readout: nothing.** Readout is dispersive, set by the qubit–cavity detuning `χ`. A 2 kHz shift
on a 1–2 GHz detuning changes `χ` by ~10⁻⁶ relative. Invisible.

### 2.3 What it DOES hurt: any phase-sensitive measurement, immediately

For a free-evolution delay `τ`, the accumulated phase is `φ = 2π δf τ`. At the *observed* offset:

| delay τ | phase at 1.50 kHz | at 1.85 kHz | offset that costs 1° |
|---|---|---|---|
| 10 µs | 5.4° | 6.7° | 278 Hz |
| 100 µs | **54.0°** | 66.6° | 27.8 Hz |
| 150 µs | 81.0° | 99.9° | 18.5 Hz |
| 300 µs | 162.0° | 199.8° | 9.3 Hz |

This is the real content of the finding. **At 100 µs, the device's existing 1.5 kHz offset already
wrecks a Ramsey-type measurement by 54°** — not from drift during the run, from the offset that is
already there when the run starts. The tolerance for a 1° error at that delay is **27.8 Hz**, which
is 54× smaller than the offset itself.

Consequence, and it is a hard design constraint rather than a caution: **any phase-sensitive
sequence must be phase-calibrated per job, not per backend calibration.** A Ramsey fringe measured
at 09:00 is worthless at 09:20. None of E1–E6 is currently phase-sensitive in this way — T1, echo
T2, CHSH (basis set by exact `RZ`), and mirror survival are all population measurements — so this
constraint does not bite today. It bites the moment anyone proposes a phase-estimation experiment,
which is where most of the roadmap's interesting hardware ideas live.

### 2.4 E1's useful half-life

The lead asked for this specifically, so it needs a definition rather than a vibe.

> **Definition.** E1's *useful half-life* is the wall-clock time after which the calibration E1
> produces no longer supports the claim it was measured for — i.e. after which a parameter it
> reports has moved by more than the tolerance of its most demanding consumer.

There are two answers, and **which one governs is not knowable from two points.** That is the
honest headline.

**(a) If the drift is only a frequency offset: 3–112 days.** The binding tolerance is the 1 %
gate-angle criterion, which needs `δ = √(2·0.01)·Ω`:

| drive rate | offset for 1 % angle error | half-life at 0.292 Hz/s | at 2.792 Hz/s |
|---|---|---|---|
| 5 MHz | 707 kHz | 28.1 d | 2.9 d |
| 10 MHz | 1.41 MHz | 56.1 d | 5.9 d |
| 20 MHz | 2.83 MHz | 112.2 d | 11.7 d |

So in the benign branch, a calibration is good for **about a week at worst, a couple of months at
best**, and the 20-minute drift we measured is 4 orders of magnitude below the tolerance. Nothing
about E1 expires in a block.

**(b) If the drift is a symptom of a mechanism that also moves T1: ~1–2 hours.** This is the
branch that matters, and it is `[speculation]` with a stated test. Kilohertz-scale frequency
wander on a fixed-frequency transmon is the classic signature of a **two-level system (TLS) or
1/f flux noise** wandering near the qubit. TLS drift does not only move the frequency — it moves
`T1`, often by 10–30 % on hour timescales, and `T1` is the parameter E1 exists to measure. If that
is the mechanism, then the tolerance that binds is the pre-registered CI width on T1 (< 20 % of
its value), and the half-life is **~1–2 hours**.

**The decision this forces.** Both branches are much longer than the 300 s block, so **E1's output
is valid for E3 and E5 within the block under either branch** — the "E1 first because E3/E5 consume
it" ordering is safe, and the drift finding does not disturb it. The 1–2 hour branch only matters
for *reuse across sessions*, and it is settled by a **2-second amendment** (§4.1). Until it is
settled, no E1 calibration may be reused in a later session without a T1 re-check, and the honest
provenance statement is "valid for the block it was measured in".

**(c) The provenance consequence, which holds under both branches.** The published calibration
timestamp does **not** bound the device's validity. At 1050–10050 Hz/h, the frequency field in a
published calibration is stale within an hour of publication, and IBM's update cadence is hours.
So a `calibration_timestamp_utc` field is **necessary but insufficient**, and the prereg template
must grow `block_start_utc`, `block_end_utc` and an in-block probe. This is a direct correction to
`docs/qpu/nitrogenase_prereg_TEMPLATE.json` and it is the deliverable the drift earned.

### 2.5 What it means for E3

**E3 is safe, by five orders of magnitude — and that makes E3 more falsifiable, not less.** During
E3's ~2.9 s block the drift moves the offset by 8.1 Hz; over a 20 µs circuit that is a phase of
1.0 × 10⁻³ rad, i.e. a fidelity change of **1.0 × 10⁻⁶**, against a pre-registered P2 effect of
**0.123**. The drift is **8.4 × 10⁻⁶ of the effect**. (20 µs is conservative — depth-10 layers of
single- and two-qubit gates on 32 qubits run in ~6 µs — and the effect scales linearly with circuit
duration, so even a 1 ms circuit gives 5 × 10⁻⁵, still 3 orders below.)

Two consequences:

1. **A null result on E3 would be a real refutation, not noise.** Because the drift floor is 10⁻⁶
   and the predicted effect is 10⁻¹, E3 has ~5 orders of headroom. If P2's 0.123 drop does not
   appear, no one gets to blame the drift. That is a much stronger experiment than it was before
   we knew this number.
2. **E3's k = 0 and k = 40 points must be in the same job.** Not because drift would fake a
   k-dependence (it would not — 10⁻⁶ against 10⁻¹), but because E3's *other* named vacuity risk is
   real: if a transpiler cancels the kink pair, the compiled depths are identical across k and the
   run is void. Splitting the sweep across jobs adds a second way to be vacuous for no gain.

### 2.6 What it means for E5

**In-block: safe, same arithmetic.** E5's block is ~3.6 s, drift moves the offset by 10.1 Hz, the
fidelity change is 1.6 × 10⁻⁶ — against a pre-registered model band of ±0.05 on points that span
0.004 to 0.376. Four orders of headroom. The depth axis of `VALIDATION_BENCHMARK.md` §3.1 can be
measured with this drift present and the measurement will not be corrupted by it.

**But E5 has a real vulnerability the drift exposes, and it is not the drift itself.** E5's
pre-registered success criterion is "≥ 80 % of (n, D) points inside the 95 % band of our model
**re-evaluated with E1's measured channel parameters**". If the TLS branch is true, E1's T1 measured
at the *start* of the block may not be the T1 during E5 at the *end*. Then the model band and the
measured points are evaluated at different channels, and a systematic offset between them would be
misread as "our model is wrong" — a false refutation of our own benchmark. That is a failure mode
where the experiment damages a true claim, which is worse than the reverse.

**Fix, and it is cheap:** the T1-repeat amendment in §4.1 runs E1's T1 arm again at the end of the
block, so the channel is measured at both ends and E5 can state which channel its points were
compared against. 2 seconds of QPU time against a false refutation is not a hard trade.

**One more thing E5 inherits from R1:** E5's model needs the two-qubit error, and R1's §1.4
inconsistency means we do not currently know the readout error (1.3 % is implied, not measured).
E5's band is a function of exactly that number. So the SPAM circuits in §4.2 are a prerequisite for
E5 as well as for E2 — one 4-circuit measurement serves two experiments.

---

## 3. R3 — the nitrogen chemistry check: owed, and correctly framed

No clear result, still owed. Recorded in `provenance.json` so the gap is visible rather than absent.
There is no number here, so there is nothing to cite and nothing to defend.

The framing rule from `docs/QPU_NITROGENASE_EXPERIMENTS.md` and the prereg template survives
untouched and applies however this turns out: **hardware calibrates or validates our simulator;
hardware never computes the chemistry.** No sentence of the form "the QPU produced X kJ/mol" is
permitted. The permanent ceiling — FeMo-co's 54-orbital active space at ~10¹¹ T gates, and the
d = 4/n = 16 state's 4¹⁶ amplitudes with no known state-prep circuit — is not a budget problem and
this run does not touch it.

---

## 4. Amendments this run earns

All `[design]`. None is pre-registered; each needs a freeze before it can be used.

### 4.1 A1 — E1 gains a closing T1 repeat (+50 000 shots, ≈ 2 s)

Repeat the T1 arm at the end of the block: 4 qubits × 5 delays × 2500 shots. Also record the qubit
frequency at both ends (read the backend property, or one Ramsey fringe).

- **Success:** `|ΔT1|/T1 < 10 %` → the block's channel is time-invariant, and E1's half-life is
  the benign branch (§2.4a). E5's model band is then valid for the whole block.
- **Failure:** `|ΔT1|/T1 > 30 %` → the block's channel is time-varying; E1 must report T1 at both
  ends and E5 must state which one it used. The block is not void, but the "one channel" claim is.
- **Why it is worth 2 s:** it is the only way to tell §2.4(a) from §2.4(b), and §2.4(b) is the
  branch that would silently corrupt E5.

### 4.2 A2 — E2 measures the 3×3 correlation matrix, not one CHSH number (+9 circuits)

Replace the 4 CHSH settings with the 9 Pauli-pair settings (`XX, XY, XZ, YX, … ZZ`), from which
`S_max = 2√(t₁²+t₂²)` and the frame rotation are both recoverable. Add 4 SPAM circuits
(prepare `|0⟩` and `|1⟩`, measure in Z; prepare `|+⟩`, measure in X) to measure readout error
directly.

- **Why:** §1.4's 7 σ inconsistency between "98 % fidelity" and "S = 2.61" is *unresolvable* with
  4 correlators, because §1.2's degeneracy means a 22.66° frame rotation and a 7.7 % visibility
  loss produce the same `S`. The full matrix separates them, and the SPAM circuits measure the
  1.3 % rather than inferring it.
- **Cost:** 13 circuits instead of 4. Under IBM's published metering that is ~13 × 5000 shots of
  marginal cost — small next to the per-job fixed cost (§5). It also makes E5's model band
  trustworthy, since the band depends on the readout error.

### 4.3 A3 — persist the CHSH angles in the artifact (a defect, not a preference)

E2's design says the settings "are the LP-optimal angles already in our artifact — not re-tuned
after seeing hardware data". **Our artifact does not contain the angles.** `likeness_probes.py`
computes them in `chimax_settings()` from the SVD of the correlation matrix and
`likeness_results.json` stores only the resulting `chsh` value. So today we cannot verify that the
hardware run used the pre-registered settings, which is the single thing E2 exists to establish.

The fix is small — write `a1, a2, b1, b2` into `P5_no_lhv.n2` — but it touches an approved
artifact's script, so it is proposed here rather than smuggled into this PR. Flagged to the lead.

### 4.4 A4 — provenance fields the prereg template is missing

Add to `docs/qpu/nitrogenase_prereg_TEMPLATE.json`'s `device` block: `run_date_utc`,
`block_start_utc`, `block_end_utc`, `job_ids`, `shots_per_circuit`, `chsh_settings`, `rep_delay`,
and `calibration_record_sha256`. Rationale: R1 arrived without any of these, and §2.4(c) shows that
`calibration_timestamp_utc` alone cannot bound validity.

---

## 5. Spend — and a reconciliation that partly rescues the budget

`[reported]`: 39 s for the CHSH block, 16 s for the recalibration-control block (derived:
55 − 39), **55 s total** against a **30 s cap** that was exceeded by 9 s because "a shot setting
failed and the circuits ran at full shots".

### 5.1 The cap did not hold, and the reason is structural

The 30 s cap was enforced by an **estimate at submission**, not by the runtime. A failed shot
setting silently escalated the job to full shots and the job was allowed to finish. **A cap that
can be exceeded by a configuration error is not a cap.** The fix is to enforce it on the quantity
that is known before submission and is not negotiable mid-flight: the shot count in the PUB list.
Under IBM's published formula `est = 2 + 0.00035 × executions`
(`docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §5.1), a 30 s ceiling means `executions ≤ 80 000` — checkable
locally, in advance, in one line.

### 5.2 The 39 s is *below* IBM's published estimate for E2 — and that is genuinely interesting

E2 as pre-registered is 4 circuits × 50 000 shots = 200 000 executions. IBM's published formula
gives `2 + 0.00035 × 200 000 = 72 s`. **Observed: 39 s.** Two readings fit, and one data point
cannot separate them:

| reading | implied | consequence |
|---|---|---|
| **A** default metering (350 µs/execution), fewer shots | `N = (39−2)/0.00035 ≈ 105 700` executions | the run used ~53 % of the planned shots — which would mean σ(S) ≈ 0.004, not 0.02, and the stated ±0.02 is doing other work |
| **B** full 200 000 shots, reduced `rep_delay` | `(39−2)/200 000 = 185 µs/execution`, i.e. `rep_delay ≈ 85 µs` | the runbook's 350 µs/execution is conservative, and the six-experiment block re-costs at **283 s** instead of **534 s** — inside 300 s |

Reading B would be good news: it means the whole six-experiment block is affordable after all, and
the runbook's §5.1 conclusion ("534 s = 89 % of the allowance, the shot budget is now the thing to
defend") is too pessimistic by ~2×. Reading A would mean the shots were cut and the error bar is
misstated. **The job JSON separates them in one line — it reports both the shot count and
`rep_delay`.** That is now the most valuable single artifact to attach, and it is why
`provenance.json` puts it at the top of the promotion list.

I am not booking a plan on either reading. One point is not a model.

### 5.3 What is not in doubt

E2 was planned at **1.2 s** and cost **39 s — 32.5× the plan.** Even if reading B is right, the
plan's 1.2 s was wrong by a factor of 30, because it priced marginal cost and ignored the per-job
fixed overhead. The runbook already knows this (§3.3: "as 100 jobs that is 200 s of pure overhead")
and prescribes batching. **This run is the first empirical confirmation of that warning, and it
came from an unbatched 4-circuit job.** The rule that follows is not "we can't afford E2"; it is
**"never submit a small experiment as its own job"** — batch everything into one job per block, and
the fixed cost is paid once.

---

## 6. What would change our mind

| observation | what we would have to change |
|---|---|
| job JSON shows < 150 000 shots for R1 | the ±0.02 is not a statistical error bar; every σ in §1 is void and §5.2 reading A is correct |
| SPAM circuits show readout error ≫ 2 % per qubit | §1.4's readout explanation fails; the 7 σ gap needs a gate or setting error instead, and E2's "settings are realisable" claim weakens |
| the 9-setting matrix shows a frame rotation > 5° | the settings we submitted were not the settings we computed — A3 becomes urgent and E2's result says nothing about our LP search |
| the T1 repeat moves > 30 % | §2.4(b) governs: E1's half-life is ~1–2 h and E5's model band must be evaluated at a stated end-of-block channel |
| the drift is a *rate* in kHz per unit time, not an offset | R2 must be restated and §2 re-derived; nothing in §2 survives unchanged |
| E3's compiled depths are identical across k | E3 is void — report as loudly as a hit, per `docs/validation/BLIND_PREDICTION_2.md` |
| anyone proposes citing R1 as evidence about the FBSC core | refuse: §1.1, the state was a Bell state |

---

## 7. Reproduce

```bash
cd docs/qpu/results/ibm_fez_run-001
python3 analysis.py                 # prints, and writes analysis_output.json
```

Inputs are the owner's reported numbers at the top of `analysis.py` (`REPORTED`). Every other
number in this document is an output of that file or an exact identity (`2√2`, `(1+3v)/4`,
`(1−2e)²`, `(π/2)rτ²`) quoted with its formula. Standard library only.

The one thing `analysis.py` cannot reproduce is the data — because there is no data file. That is
the whole point of the status tag.

---

## 8. Provenance of the claims in this document

| claim | source | tag |
|---|---|---|
| S = 2.61 ± 0.02, ibm_fez q44/45, n = 2, F ≈ 98 % | owner brief | `[reported]` |
| 1.85 → 1.50 kHz, 20 min, no recalibration | owner brief | `[reported]` |
| 39 s CHSH / 55 s total / 30 s cap | owner brief | `[reported]` |
| design 2.8250, and its "0.003 off Tsirelson" | `docs/artifacts/quantum_likeness/likeness_results.json` → `P5_no_lhv.n2` (PR #146) | `[measured, classical optimisation]` |
| 30.5 σ above 2; 0.2184 below Tsirelson; 10.75 σ below design | `analysis.py` | `[analytic]` |
| Werner v = 0.9228, F = 0.9421; 22.66° frame degeneracy | `analysis.py` | `[analytic]` |
| 98 % ⇒ S = 2.753; implied 1.32 %/qubit readout | `analysis.py` | `[interpretation]` — conditional on two unestablished premises (§1.4) |
| Mermin-3 = 1.9315 < LHV bound 2 | `likeness_results.json`; bound from `mermin_max_3q` docstring | `[measured, of our own state]` |
| drift rate bracket [0.292, 2.792] Hz/s | `analysis.py` | `[analytic]`, given the §2 reading |
| gate angle error 10⁻⁹–10⁻⁸; half-life 3–112 d | `analysis.py` | `[analytic]` |
| echo residual ≤ 4 × 10⁻⁷ rad | `analysis.py`, verified numerically | `[analytic]` |
| E3/E5 drift floor ≈ 10⁻⁶ vs effect 0.123 | `analysis.py` | `[analytic]` |
| TLS/1-f as the drift mechanism, ⇒ 1–2 h half-life | this document §2.4(b) | `[speculation]` — test A1 |
| IBM metering `2 + 0.00035·executions` | `docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §5.1 | `[published]` |
| block re-cost 283 s vs 534 s | `analysis.py` | `[analytic]`, one data point |
| A1–A4 amendments | this document §4 | `[design]` — none pre-registered |

**Not claimed anywhere in this document:** new physics; a loophole-free Bell test; quantum
advantage; topological protection or anyons; any chemistry in kJ/mol; any device T1/T2 (E1 never
ran); that the FBSC core produced the state that was measured; that these numbers are `[measured]`.
