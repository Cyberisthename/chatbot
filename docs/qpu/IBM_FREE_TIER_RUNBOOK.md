# IBM Quantum free tier — runbook for the six nitrogenase experiments

**Status:** design document. **No QPU job has been submitted.** Nothing here is a measurement.
**Owner decision recorded (2026-11-05):** we use IBM Quantum's free Open Plan (Qiskit Runtime),
not a paid plan, for the first hardware block.
**Maps onto:** `docs/QPU_NITROGENASE_EXPERIMENTS.md` (PR #166) — the six experiments E1–E6, their
shot budgets and their pre-registered thresholds are **unchanged by this document**. This file
changes only *how they are paid for, scheduled and costed on IBM's free tier*, and it corrects one
number in that document that does not survive contact with IBM's published metering (§5).
**Terms verified:** on IBM's own pages, **2026-11-05**. IBM changes plans; re-verify §1–§4 at
booking and update this file's date stamp. Every external claim in §1–§4 carries its source URL in
§11; every number we computed carries a tag.

Tag discipline (unchanged from the repo standard): `[published]` = on a vendor page, quoted;
`[analytic]` = arithmetic from stated inputs; `[assumption]` = a value we chose and must read back
at booking; `[design]` = untested; `[interpretation]` = judgement; `[measured]` = we ran it (there
is none in this file).

---

## 0. The one-paragraph version

The free Open Plan gives **10 minutes of QPU time per 28-day rolling window** (`1 RU = 1 minute`),
plus a documented one-time promotion of **180 minutes over 12 months** for users who have already
used 20 minutes in a 12-month period. Metering counts **QPU lock time only** — not transpile, not
queue wait — and IBM's own quick estimate is **`2 + 0.00035 × executions` seconds**, where an
execution is one circuit × one shot. Our six experiments are **1.52×10⁶ executions**: at IBM's
default settings that is **≈ 534 s = 89 % of one monthly allowance**, not the 24 s the experiment
document assumes. With `rep_delay` tuned down and circuits kept short, the same plan costs
**≈ 192 s = 32 % of a month** and fits with a 2× reserve. So: the plan fits on the free tier, but
only in the tuned configuration, and the honest headline is that **the free tier is now the binding
constraint on shot count, where the 300-second budget never was**. Two of the six experiments need
a hard decision: **E4 (d ≥ 3 qudits) is unobtainable on IBM** because pulse-level control was
deprecated and removed — that is availability, not price — and **E1's T1/T2 delays are irreducible
QPU time** that no tuning removes. Everything Qiskit lives under `hardware/` and is never imported
by the FBSC path (§9).

---

## 1. What the Open Plan actually is (verified 2026-11-05)

| term | value | tag |
|---|---|---|
| allowance | "the RU equivalent of up to **10 minutes per 28-day rolling window**" | `[published]` |
| unit | "For standard QPU workloads, **1 RU = 1 minute** of QPU compute time" | `[published]` |
| window | 28-day **rolling**; Instances page shows the last 28 days | `[published]` |
| region | "Open Plan instances **can only be created in the us-east region**" | `[published]` |
| sessions | "**Open Plan users cannot submit session jobs**" | `[published]` |
| batch mode | not excluded for Open Plan (only sessions are) | `[published]` (by omission) |
| dry-run jobs | "available on the Flex, Premium, and On-Prem plans" — **not on Open** | `[published]` |
| per-job ceiling | service maximum execution time **3 hours** (QPU time, not wall clock) | `[published]` |
| one-time promotion | opt in after using 20 min in any 12-month period → **180 min for 12 months**, any pace, no rollover limit, then back to 10 min/28 d | `[published]` |
| cost | free; no payment instrument | `[published]` |

**Consequences we must design around, not discover on the day:**

1. **No dry run on Open Plan.** The one tool IBM provides for "what would this job be charged"
   is unavailable to us. Our pre-flight estimate is therefore *our own* `estimate_duration`
   arithmetic (§4.3), and it must be written into the pre-registration before submission.
2. **No sessions.** Every job goes through the normal fair-share queue separately. That is exactly
   the "three channels measured in three jobs" hazard the experiment document warns about — on the
   free tier we cannot buy our way out of it with a session, so we buy it out with **batch mode**
   (§3.3) and accept that the batch may be interleaved with other users' jobs and calibrations.
3. **us-east only.** All our devices are in one region; there is no EU fallback if the fleet is
   busy.
4. **The 180-minute promotion is the difference between "one block" and "drift statistics".**
   At 32 % of a monthly window per block, the standard allowance buys **3 blocks per 28 days** at
   best (and only 1 at default settings). The promotion buys **≈ 56 blocks** — which is what §8 of
   the experiment document calls "1 hour: repeated blocks across days → drift statistics".
   It is a *one-time* offer with a pre-requisite (20 min of prior use), so it cannot be the plan of
   record; it is an upside.

---

## 2. Which backends the free plan gives

**The authoritative list is account-specific and must be read at booking, not quoted from here.**
IBM exposes the fleet per instance; the public docs describe processor *families*, and the
`/computers` page and the REST API both require an authenticated session (`GET /api/v1/backends`
returns `401` unauthenticated — verified 2026-11-05).

What *is* publicly documented as of 2026-11-05:

| family | revisions | notes | tag |
|---|---|---|---|
| **Nighthawk** | r2 (Sep 2026), r1 (Dec 2025) | r2: 120 programmable qubits / 458 physical, grid topology, fast reset "significantly increases the maximum number of circuits per second" | `[published]` |
| **Heron** | r3 (Jul 2025), r2 (Jul 2024), r1 (Dec 2023) | 156 qubits, heavy-hex | `[published]` |
| **ibm_kingston** | Heron r2 | "now available to all Open Plan users", ~340k CLOPS, median two-qubit error **2.03×10⁻³** | `[published]` (IBM blog, 16 Mar 2026) |

Two operational facts that matter more than the names:

- **The free-plan fleet is a subset, and it changes.** `ibm_kingston` was paid-plan-only until
  March 2026 and is now open — i.e. the free fleet *grows*, and any name we hard-code today can be
  retired tomorrow. Record the device in the artifact (§10), never in the code.
- **`ibm_kingston`'s published median two-qubit error 2.03×10⁻³ is a *median over the chip*, not a
  per-pair number.** E1 measures four *adjacent pairs* precisely because the median is not the pair
  you will use. Do not substitute the published number for the measurement; the whole point of E1
  is that the published number is not our number.

**Booking-time commands** (run under `hardware/`, never from the FBSC path):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum_platform")   # token from env, never a file
backends = service.backends(operational=True, simulator=False, min_num_qubits=100)
for b in backends:
    print(b.name, b.num_qubits, b.backend_version, b.status().status_msg)
    print("  default_rep_delay:", b.default_rep_delay, " range:", b.rep_delay_range)
```

Record `name`, `backend_version`, `default_rep_delay`, `rep_delay_range`, `num_qubits` and the UTC
date into the pre-registration JSON **before** submitting anything. Those five fields are what make
a hardware number reproducible; a device without them is not a measurement, it is an anecdote.

---

## 3. Qiskit Runtime primitives on the free tier

### 3.1 What we use

**Primitives only.** `SamplerV2` for everything in E1–E6 except the energy estimation in E6, and
`EstimatorV2` for E6's expectation values. `QiskitRuntimeService` for account/instance handling.
Nothing else from the SDK is needed, and nothing else should be imported — every extra import is
another line the originality audit has to defend (§9).

Note the version split IBM documents: server-side primitives are the pre-`0.50.0` form; from
`qiskit-ibm-runtime` **0.50.0** the client-side primitives live at
`qiskit_ibm_runtime.executor_estimator` / `executor_sampler`. **Pin the version in
`hardware/requirements.txt` and record the exact version in the artifact**, because the import path
itself changed inside the last year. `[published]`

### 3.2 The knobs that cost money

Only four options change the QPU bill. Set all four explicitly — never inherit a default:

| option | effect on cost | our setting |
|---|---|---|
| `execution.rep_delay` | **the dominant term.** Default is `backend.default_rep_delay`, "250 microseconds on most IBM Quantum backends" | set to the low end of `backend.rep_delay_range` (§5.2) |
| `execution.init_qubits` | adds a reset per shot | `True` for E1 readout, else default |
| error suppression / mitigation | "extra instructions might be inserted… increase the total number of executions" | E6: readout mitigation only (calibration-matrix inversion, done **classically** on returned counts — no PEA/PEC) |
| `max_execution_time` | a *ceiling*, not a cost | set to our estimate × 1.5 per job, so a runaway is cancelled (§8) |

**Do not use PEA/PEC or `NoiseLearner` on the free tier.** They run extra noise-learning circuits
whose cost we would pay out of the 10 minutes, and the docs themselves warn a learned noise model
"becomes stale" — on a queue that can be hours long, it will be stale before it is used. `[published]`

### 3.3 Execution mode: **batch**, not job-per-circuit, not session

- **Session: impossible** (Open Plan excluded).
- **One job per circuit: ruinous.** The meter charges "approximately **2 s per sub-job**" of
  overhead. E1's RB component alone is 4 pairs × 25 depths = **100 circuits**; as 100 jobs that is
  **200 s of pure overhead** — a third of the month, for nothing. `[analytic]`
- **Batch, or one primitive call with many PUBs: correct.** Package each experiment as **one job**
  whose PUB list is every circuit at its shot count, and put the five jobs in **one batch**. Then
  the 2 s overhead is paid ~5 times, not ~200.

```python
from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler
with Batch(backend=backend) as batch:
    sampler = Sampler(mode=batch)
    sampler.options.execution.rep_delay = REP_DELAY
    sampler.options.max_execution_time  = EST_S + 60
    job = sampler.run(pubs)          # pubs = [(circuit, None, shots), ...] for the whole experiment
```

**Caveat we accept, written down in advance:** a batch does not get exclusive access, jobs "might run
in parallel with other users' jobs", and "QPU calibration jobs could run between the batched jobs".
So the "one measured channel" ideal is *approximated*, not achieved, on the free tier. E1 must
therefore report its **per-component timestamps** and we must state in the artifact that the channel
was measured across one batch, not one session. That is a weaker claim than the experiment document
assumes, and it is the correct one. `[published]` + `[interpretation]`

---

## 4. How the 10 minutes is metered

### 4.1 What is charged

> "Usage represents consumption of IBM Quantum Compute Service and is determined by the amount of
> time a **QPU is locked to execute** workloads." — *Workload usage*, IBM docs `[published]`

- **Single job:** time the QPU is locked to execute that job.
- **Batch:** "the cumulative time the QPU is locked to execute all jobs in the batch".
- **Session:** wall-clock while the session is active — *irrelevant to us, sessions are blocked.*
- **Not charged:** transpilation (local), queue wait, classical post-processing, dashboard time.

### 4.2 Failed and cancelled jobs

| cause | charged? | tag |
|---|---|---|
| system error | **zero** | `[published]` |
| user error, or user-cancelled | "any consumption that has occurred up to that point, **including the overhead incurred to prepare the QPU**" | `[published]` |

So a bad circuit that dies at the transpiler costs nothing; a bad circuit that runs and is cancelled
costs the ~2 s QPU prep plus whatever ran. **A pre-flight validation that fails locally is free; a
pre-flight validation that fails on hardware is not.** This is the argument for §4.4.

### 4.3 The estimate formula (IBM's own)

```
usage ≈ <per sub-job overhead ≈ 2 s> + (rep_delay + circuit_length) × executions
executions = Σ over circuits of shots          # circuits AFTER PUB broadcasting
quick form (no mitigation, default rep_delay):  2 + 0.00035 × executions      [published]
```

`0.00035 s` per execution = **350 µs = 250 µs default `rep_delay` + ~100 µs typical circuit**
(IBM: "A typical circuit length is likely between 50-100 microseconds"). `[published]` + `[analytic]`

The local estimator IBM documents, which we must run and store in the pre-registration:

```python
pm = generate_preset_pass_manager(target=backend.target, optimization_level=0, scheduling_method="alap")
scheduled = pm.run(isa_circuits)
init_d = backend.target["reset"][(0,)].duration
rep = sampler.options.execution.rep_delay or backend.default_rep_delay
circ = sum(c.estimate_duration(backend.target) for c in scheduled)
circ += len(scheduled) * (init_d + rep)          # per IBM's example
est  = 2 + circ * shots
```

**Note `optimization_level=0` in IBM's own estimator example** — which is also exactly what E3
requires for its own reasons (§6.3). One setting, two obligations.

### 4.4 Read the truth back after every job

```python
print(job.metrics()["usage"]["quantum_seconds"])   # actual QPU seconds charged  [published]
print(job.usage())                                 # qiskit-ibm-runtime >= 0.30
```

Every artifact we publish carries `estimated_usage_s` **and** `measured_usage_s`. A runbook that
only predicts its cost is a runbook that never learns. The estimate/actual ratio becomes the
calibration constant for the next block.

---

## 5. The budget — and the number that does not survive

### 5.1 The correction

`docs/QPU_NITROGENASE_EXPERIMENTS.md` §10 lists this provenance line:

> "per-shot 5 µs reset+readout; 5 µs/shot ⇒ 3–6×10⁷ shots in 300 s — `[assumption]`/`[analytic]`"

**That assumption is wrong on IBM hardware by a factor of ~70.** IBM's published metering is
**350 µs per execution at default settings** (250 µs `rep_delay` + ~100 µs circuit), not 5 µs.
Re-costing the same 1.52×10⁶ shots with IBM's formula:

```
2 + 0.00035 × 1 520 000 = 534 s          [analytic, from published formula]
```

**534 s = 89 % of one 28-day allowance**, and it leaves 66 s of reserve — enough to re-run E2 once,
and nothing else. The experiment document's conclusion ("seconds are not the scarce resource") is
**true for a 300-second budget and false for a 10-minute-per-month budget.** On the free tier the
scarce resource is *shots*, and the shot budget is now the thing to defend in review.

We are **not** editing the experiment document's thresholds or designs in this PR (they are approved
and its numbers stay; the `5 µs/shot` line is an explicit `[assumption]`, not a claim). It is
flagged here so the lead can decide whether that provenance row gets a pointer to this file. **This
is the correction that matters and it is stated first, per the working agreement.**

> **Update, 2026-11-05 — the pointer exists.** `docs/QPU_NITROGENASE_EXPERIMENTS.md` §10 now marks
> the `5 µs/shot` row **superseded on IBM hardware** and links here, quoting this section's 534 s
> and §5.3's re-cost table. Nothing in that document's §3 was re-opened, and no QPU job has run.

### 5.2 What we can do about it

`rep_delay` is user-settable, "Choices: Value in the range supplied by `backend.rep_delay_range`".
Lowering it "decreases the total QPU execution time, but at the expense of increased state
preparation error rate". `[published]`

That trade is **acceptable for E2/E3/E5/E6** (short circuits, statistical estimators, and E3
*needs* the kinks to survive anyway) and **is not a trade at all for E1's T1/T2 arms**, where the
delay τ *is* the experiment: those seconds are irreducible. We do not know the floor of
`rep_delay_range` without an account, so §5.3 gives the bound at `rep_delay = 0` — the optimistic
limit — and labels the tuned column accordingly.

### 5.3 Per-experiment budget (E4 excluded, see §7)

Model: `cost = 2 + (rep_delay + L + τ̄) × shots`; `L` = mean circuit length including readout;
`τ̄` = mean inserted delay (E1 only). `[analytic]`

| exp. | circuits × shots | executions | `L` (µs) `[assumption]` | τ̄ (µs) | **default** (350 µs/exec) | **tuned** (`rep_delay→0`) |
|---|---|---|---|---|---|---|
| E1 T1 | 40 × 2 500 | 100 000 | 100 | 93 | 37.0 s | 21.3 s |
| E1 T2 | 48 × 2 500 | 120 000 | 100 | 46 | 44.0 s | 19.5 s |
| E1 readout | 2 × 20 000 | 40 000 | 100 | 0 | 16.0 s | 6.0 s |
| E1 RB | 100 × 2 000 | 200 000 | 106 | 0 | 72.0 s | 23.2 s |
| **E1 total** | | **460 000** | | | **169.0 s** | **70.0 s** |
| E2 CHSH | 4 × 50 000 | 200 000 | 100 | 0 | 72.0 s | 22.0 s |
| E3 kink | 12 × 20 000 | 240 000 | 140 | 0 | 86.0 s | 35.6 s |
| E5 depth | 30 × 10 000 | 300 000 | 120 | 0 | 107.0 s | 38.0 s |
| E6 floor | 20 × 10 000 | 200 000 | 120 | 0 | 72.0 s | 26.0 s |
| **grand total** | | **1 400 000** | | | **506 s = 84.3 %** | **192 s = 31.9 %** |

Of the 10-minute (600 s) window. Against the 180-minute promotion: **4.7 %** and **1.8 %**.

**Sensitivity (the number that decides whether the tuned column is real):** at `L = 200 µs` the
tuned total becomes **311 s (52 % of the month)**. So the tuned plan is only comfortable while mean
circuit length stays ≲ 150 µs. E5 is the risk — 32-qubit, depth-10 mirror circuits transpiled onto
heavy-hex will be deeper than the raw layer count, and routing SWAPs are not free. **Measure `L`
with `estimate_duration` before believing this table.** If `L` comes back at 250 µs for E5, either
drop n = 32 (the depth-1-collapse point the experiment cares about) or halve its shots — and record
which, in the pre-registration, before seeing data.

### 5.4 The resulting rules of engagement

1. **Default settings do not fit.** 84 % of a month with one block per month is not a plan, it is a
   one-shot. Submit tuned or don't submit.
2. **Budget in executions, not seconds.** `executions_available = (allowance_s − 2·jobs) / per_execution_cost`.
   At 600 s and 350 µs: **1.71×10⁶ executions/month** — our plan needs 1.40×10⁶, i.e. it fits with
   18 % headroom, and *that* is the honest statement of "it fits at defaults". At 100 µs tuned:
   **6.0×10⁶ executions**, 4.3× our plan.
3. **Reserve policy: 40 % of the window.** Submit only if `estimate ≤ 0.6 × remaining allowance`.
   At tuned 192 s that holds (192/600 = 32 %); at default 506 s it does not — which is another way
   of saying rule 1.
4. **One block per window, three per 28 days maximum.** Multiple blocks inside one window are the
   only route to drift statistics without the promotion, and three is the ceiling.
5. **Disclose the cost in every artifact** (the experiment document's §7 requires it): "this block
   consumed N s of QPU time, M % of an IBM Open Plan 28-day allowance (10 min)". With the measured
   value from `job.metrics()`, not the estimate.

---

## 6. Per-experiment runbook

Shared to all: `optimization_level=0`; `rep_delay` from §5.2; one job per experiment; one batch;
`max_execution_time = 1.5 × estimate`; pre-registration frozen and pushed **before** submission;
`job.metrics()["usage"]["quantum_seconds"]` recorded per job. Thresholds below are **quoted from
`docs/QPU_NITROGENASE_EXPERIMENTS.md` and unchanged** — this runbook does not re-open them.

### 6.1 E1 — device-channel calibration (run first; everything else consumes it)

| | |
|---|---|
| primitives | 4 jobs (T1, T2, readout, RB) or 1 job with 190 PUBs; batch |
| executions | 460 000 |
| cost | **169 s default / 70 s tuned** |
| irreducible | the T1/T2 delays themselves: **≈ 15 s at any `rep_delay`** (93 µs × 100 000 + 46 µs × 120 000) `[analytic]` |
| trap | the τ grid is QPU time. 10 T1 points to 300 µs and 12 T2 points to 150 µs are 2500 shots each *through* the delay. Trimming to 8 points each saves ≈ 10 s and costs fit quality — decide before the run, not after. |
| success / failure | as written in the experiment document §3 (χ²/dof, physical ranges, T2 ≥ 5 µs or abort the block) |
| produces | `T1`, `T2`, `readout_error`, `e2` per pair, each with bootstrap CI — the independent variable for E3 and E5 |

**Free-tier specific:** E1 must **not** be run as the "calibrate → run → analyse" single session of
the experiment document. It is a batch. Therefore: record wall-clock start/end of each sub-job, and
if the batch spans more than a few hours of queue, **say so in the artifact** and treat drift as a
known systematic. `[interpretation]`

### 6.2 E2 — CHSH with our LP settings (cheapest, most quotable)

| | |
|---|---|
| primitives | 1 job, 4 PUBs (a₀b₀, a₀b₁, a₁b₀, a₁b₁), 50 000 shots each |
| executions | 200 000 |
| cost | **72 s default / 22 s tuned** |
| trap | the settings are copied from `docs/artifacts/quantum_likeness/` and **may not be re-tuned after seeing data**. Put the four angles in the pre-registration JSON as literals, hash them, and check the hash in the artifact. |
| measurement-basis detail | `RZ(−2a)` on q0 and `RZ(−2b)` on q1 are **virtual** rotations — they cost gate time ≈ 0 and no `rep_delay` change. This is the one place where the free tier is generous. `[published]` (RZ is virtual on IBM backends) |
| success / failure | S ≥ 2.60 **and** \|S − S_model\| ≤ 0.08; fail at S ≤ 2.00 or deviation > 0.15; 2.00 < S < 2.60 is **pre-declared inconclusive** |

### 6.3 E3 — does the exact-zero braid invariance survive measurement

| | |
|---|---|
| primitives | 1 job, 12 PUBs (6 k × 2 arms), 20 000 shots each |
| executions | 240 000 |
| cost | **86 s default / 36 s tuned** |
| **the trap that kills the experiment silently** | a transpiler at default optimisation **cancels σσ⁻¹** and the compiled circuit becomes bit-identical at every k. `optimization_level=0` is mandatory, and the **compiled depth per k must be printed, asserted strictly increasing, and stored in the artifact**. If the depths are flat, the run is void — report it as void, do not report "kinks change nothing". |
| cost interaction | `optimization_level=0` also means *more* gates after transpilation → longer circuits → higher `L` → higher cost. This experiment is where §5.3's `L = 140 µs` assumption is most likely to be wrong. Estimate before submitting. |
| predictions | P1 `e(40) − e(0) ≤ 0.05`; P2 mirror fidelity falls ≥ 0.15 from k=0 to k=40 (our own analytic says 0.123 — a *smaller* drop contradicts our model); P3 control arm error ≥ 10× isotopy at every k ≥ 2 |

### 6.4 E5 — measured depth decay replaces the analytic table

| | |
|---|---|
| primitives | 1 job, 30 PUBs (5 n × 6 D), 10 000 shots each |
| executions | 300 000 |
| cost | **107 s default / 38 s tuned** — **and the most likely to overrun** |
| trap | n = 32 at D = 10 on heavy-hex needs routing; the transpiled depth is not the layer count. Compute `L` per (n, D) with `estimate_duration` and use the **mean** in the estimate. If the estimate exceeds 120 s, cut n = 32 or halve shots, and pre-register the cut. |
| success / failure | ≥ 80 % of points inside the 95 % band of our model at E1's measured channel; **fail if the model is systematically optimistic at more than half the points** — then `VALIDATION_BENCHMARK.md`'s domination claim must be weakened in the published table |
| claim NOT earned | any speedup, any advantage, any fault tolerance. Device decay vs an exact classical program is not an advantage. |

### 6.5 E6 — the chemical-accuracy floor (the negative experiment, on purpose)

| | |
|---|---|
| primitives | `EstimatorV2`, 1 job, 20 PUBs (20 pre-computed parameter sets), 10 000 shots each |
| executions | 200 000 |
| cost | **72 s default / 26 s tuned** |
| mitigation | readout only, and **classically**: build the calibration matrix from E1's readout arm and invert it on the returned counts. Do **not** enable PEA/PEC — the extra noise-learning circuits are charged to us. `[published]` |
| trap | no per-point re-optimisation on hardware. The 20 parameter sets come from a classical optimisation, frozen in the pre-registration. Optimising on the QPU would multiply the cost by the iteration count and is not in the budget. |
| expectation | ε ∈ [5, 50] kJ/mol after mitigation `[speculation]`; **ε ≤ 4.184 kJ/mol is a pre-declared HIT** and gets published at the top, not buried |
| claim NOT earned | anything about FeMo-co. 4 qubits → 54 orbitals is an extrapolation, stated as one. |

---

## 7. The d ≥ 3 qudit reality: **E4 is unobtainable, not unaffordable**

The experiment document flags E4 with a ⚠ and says qudit access "may require pulse-level control that
current cloud plans do not expose `[published/uncertain — verify at booking]`". **Verified. It is
worse than uncertain — it is documented as removed:**

> "With the introduction of fractional gates, **pulse-level control on all IBM Quantum® processors
> has been deprecated and will be removed in early 2025**. Additionally, the `qiskit.pulse` module
> has been deprecated as of the Qiskit SDK v1.3.0 and **will be removed in Qiskit SDK v2.0.0**."
> — *Migrate from Qiskit Pulse to fractional gates*, IBM docs `[published]`, verified 2026-11-05

E4's circuit, as designed, requires exactly what was removed:

1. `SU(3)`/`SU(4)` pulses to prepare `(|0⟩+|2⟩)/√2`, `(|0⟩+|1⟩+|2⟩)/√3` and the d=4 analogues.
2. `|2⟩ → |1⟩` shelving plus a second readout pulse to read 3–4 levels.

On IBM's cloud today there is **no pulse access**, the sampler returns **bitstrings**, and the
"fractional gates" that replaced pulse control are **qubit** rotations `RX(θ)` and `RZZ(θ)` on
Heron — a partial-angle gate is not a qutrit gate. `[published]`

**Therefore:**

- E4 as specified **cannot be run on the IBM free tier, and — as far as the public documentation
  shows — not on any IBM cloud plan**, because the capability was removed fleet-wide, not gated by
  price. This is a stronger and cleaner statement than "unaffordable", and it is the one to publish.
- Its 44 s (default) / 14 s (tuned) does not enter §5.3's total, and E4 must not appear in any
  table as "skipped for budget reasons".

**Three honest options, in order of our preference:**

| option | what it earns | what it costs |
|---|---|---|
| **(a) Drop E4, say why** — publish "d ≥ 3 on a single transmon is unavailable through the only vendor whose free tier we can use; the d=4 row's dimension remains unverified on hardware" | nothing, but it removes a promise | a sentence in the artifact and in `docs/QPU_NITROGENASE_EXPERIMENTS.md`'s ⚠ row |
| **(b) E4-q, a qubit-encoded d=4 analogue** — prepare the d=4 state on **2 transmons** (a ququart's 4 levels mapped to 2 qubits), measure assignment fidelity | strictly weaker: "the d=4 *state* is preparable on a 2-qubit register" — **not** "one transmon supports 4 levels" | ≈ 120 000 executions, 44 s / 14 s; and the claim must be worded to make the weaker reading impossible |
| **(c) Wait for a vendor with a documented qutrit API** | the original claim | a new vendor = a new originality/legal decision, i.e. back to the owner |

**Recommendation:** (a) now, (b) only if the lead wants a hardware number in that row at all. Do not
let (b) be reported as (E4) — the two claims are different and the difference is exactly the kind
of slide this repo's standard forbids.

`[interpretation]` on the recommendation; `[published]` on the capability.

---

## 8. The block, end to end, on the free tier

**One batch, five jobs, submitted once.** Not because the queue is short — because drift between
jobs is a bigger threat than statistics, and because we only have one month's allowance.

| # | stage | tuned s | cumulative | default s | gate |
|---|---|---|---|---|---|
| 0 | **pre-flight** (free, local): transpile at `optimization_level=0`, `estimate_duration` per circuit, frozen pre-registration hash, `depths strictly increasing` assertion for E3 | 0 | 0 | 0 | abort here, it costs nothing |
| 1 | **E1** channel | 70 | 70 | 169 | T2 < 5 µs → **abort the batch**, publish the abort |
| 2 | **E2** CHSH | 22 | 92 | 241 | independent |
| 3 | **E3** kink | 36 | 128 | 327 | needs E1 |
| 4 | **E5** depth | 38 | 166 | 434 | needs E1 |
| 5 | **E6** floor | 26 | 192 | 506 | independent |
| 6 | **reserve** | 408 | 600 | 94 | one re-run of E2/E6 only |

**Abort gates, decided now so they are cheap later:**

- `estimate > 0.6 × remaining allowance` → **do not submit**; cut shots (halve E5 first, it is the
  most expensive per unit of claim) and re-freeze the pre-registration.
- E1 fails its physical ranges → the whole block is void. Report "the device could not support the
  measurement", publish the numbers that failed, and **do not spend the reserve** chasing it.
- E3 compiled depths flat → void E3 only; the other four stand.
- Any job cancelled by us → we are charged up to that point; note it in the artifact.
- Job hits `max_execution_time` → it is cancelled and charged; this is why the ceiling is 1.5× the
  estimate rather than "unlimited".

**After the batch (free, do it immediately — the numbers are per-window):**

```python
for j in batch_jobs:
    print(j.job_id(), j.metrics()["usage"]["quantum_seconds"])
```

Sum them, divide by 600, and write the percentage into the artifact. If the sum exceeds 60 % of the
window, the *next* block must be planned with fewer shots — that is the whole point of measuring it.

---

## 9. The `hardware/` sandbox rule (non-negotiable)

The owner's decision to use IBM collides with two standing constraints — the PR #149 governance
audit ("no third-party quantum SDKs in the owned core") and the `legacy/` boundary ("zero
third-party imports outside `legacy/`"). Both stay true. The resolution is that Qiskit lives in a
**third top-level sandbox** that nothing in the owned path can reach.

### 9.1 Layout

```
hardware/
  README.md                 # "calibration of a third-party device; not part of the owned stack"
  requirements.txt          # qiskit==<pinned>, qiskit-ibm-runtime==<pinned>   (only place these appear)
  check_boundary.sh         # the grep that must return empty (CI gate)
  preflight.py              # local estimate, no account needed
  run_block.py              # the batch submission
  collect.py                # metrics -> JSON
```

### 9.2 Rules

1. **Nothing outside `hardware/` imports `qiskit`, `qiskit_ibm_runtime` or `qiskit_aer`.** Not
   `src/`, not `scripts/`, not `docs/`, not the JARVIS server, not the lab site.
2. **`hardware/` is never imported by the FBSC path**, directly or transitively. It is a leaf.
3. **No result crosses back as code.** Hardware produces **JSON** (`docs/artifacts/qpu/<UTC-date>/`);
   the owned core may read that JSON the way it reads any other data file, and nothing more.
4. **The token is an environment variable** (`QISKIT_IBM_TOKEN`) read at runtime. Never in a file,
   never in a commit, never in `.env`, never in a notebook output.
5. **No vendor code is vendored.** MIT covers our files; it does not cover IBM's SDK terms, and it
   does not cover "J.A.R.V.I.S." (separate, unresolved owner decision).
6. **`legacy/` keeps its rule unchanged.** `hardware/` is a second, *named* exception, not a
   loosening of the first — the boundary README must say so in those words.

### 9.3 The evidence (this is what a reviewer will run)

```bash
# must print nothing
grep -rn --include='*.py' -E 'qiskit|qiskit_ibm_runtime|qiskit_aer' . \
  | grep -v '^./hardware/' | grep -v '^./legacy/' | grep -v '^./node_modules/'

# the owned core's own imports, unchanged
grep -rn '^import \|^from ' src/ | grep -vE 'numpy|math|json|os|sys|time|typing|dataclasses|pathlib|hashlib|collections|itertools|random|re|abc|functools|logging|sqlite3|datetime|subprocess|threading|queue|asyncio|fastapi|pydantic|uvicorn|starlette'
```

`hardware/check_boundary.sh` is the first command, with a non-zero exit when it is non-empty, and it
runs in CI. **A boundary that is only a paragraph in a README is a boundary that will be crossed.**

### 9.4 What the audit must then say (suggested wording)

> The owned core (`src/`, `scripts/`) imports no third-party quantum SDK and no LLM SDK; this is
> enforced by `hardware/check_boundary.sh` in CI. Device calibration code lives in `hardware/`, is
> a leaf, is not imported by the FBSC path, and is disclosed here because it uses `qiskit` /
> `qiskit-ibm-runtime` under IBM's terms. No vendor code is vendored and no result is imported as
> code. The owned-core claim is unchanged; the hardware numbers are calibration of a third-party
> device, not part of the owned stack.

Add that as **A5** in `docs/artifacts/legal/GOVERNANCE_GATE.md` with status *conditional pass —
boundary enforced*, and note that **A4** ("no hardware claims") now needs one sentence of nuance:
we may report *device calibration* and *measured device behaviour*, never *our core on a QPU*.

---

## 10. Pre-registration deltas (add these fields, keep the rest)

`docs/qpu/nitrogenase_prereg_TEMPLATE.json` freezes thresholds. The free tier adds fields that must
be frozen *too*, because they are the ones a reader needs to judge cost and reproducibility:

| field | why |
|---|---|
| `backend_name`, `backend_version`, `num_qubits` | an unnamed device is an unreproducible claim |
| `instance_plan` (`"open"`), `region` (`"us-east"`) | the terms under which the number was produced |
| `qiskit_version`, `qiskit_ibm_runtime_version` | the import path changed at 0.50.0 |
| `rep_delay_used`, `default_rep_delay`, `rep_delay_range` | the single largest cost lever, and a physics lever |
| `optimization_level` (must be 0) | E3's validity depends on it |
| `estimated_usage_s` per job, `estimated_total_s` | our pre-flight, frozen before submission |
| `allowance_at_submit_s`, `allowance_remaining_s` | what the block was actually competing for |
| `batch_id`, `job_ids` | the only way to find the data again |
| `mitigation` (exact method + where it was applied) | "readout matrix inversion, classical" is a different claim from "PEA" |

`measured_usage_s` is filled **after** the run, in the artifact, never in the frozen file.

---

## 11. Provenance

All URLs accessed **2026-11-05**. IBM reorganises its docs; if a link 404s, the claim needs
re-verifying, not re-asserting.

| claim | source | tag |
|---|---|---|
| 10 min / 28-day rolling window; 1 RU = 1 min; us-east only; dry-run not on Open; 180-min promotion (20 min prerequisite) | `quantum.cloud.ibm.com/docs/en/guides/plans-overview` | `[published]` |
| `ibm_kingston` (Heron r2) opened to Open Plan; 340k CLOPS; median 2Q error 2.03e-3; 20 min → 180 min promotion terms | `ibm.com/quantum/blog/open-plan-updates` (16 Mar 2026) | `[published]` |
| usage = QPU lock time; per-mode definitions; failed/cancelled rules; `2 + 0.00035 × executions`; 2 s/sub-job; default `rep_delay` 250 µs; typical circuit 50–100 µs; `estimate_duration` example | `.../guides/estimate-job-run-time` | `[published]` |
| session jobs barred on Open Plan; batch interleaving and calibration-jobs-between-batches | `.../guides/execution-modes` | `[published]` |
| fair-share ordering; 28-day rolling window for fairness; FIFO within an instance | `.../guides/fair-share-scheduler` | `[published]` |
| 3-hour service maximum execution time; `max_execution_time` is QPU time; `job.metrics()['usage']['quantum_seconds']` | `.../guides/max-execution-time` | `[published]` |
| `rep_delay` range/default semantics; `max_execution_time` ∈ [1, 10800]; options table | `.../guides/sampler-options` | `[published]` |
| pulse-level control deprecated, removed early 2025; `qiskit.pulse` removed in Qiskit 2.0; fractional gates `RX(θ)`, `RZZ(θ)` as the replacement | `.../guides/pulse` ("Migrate from Qiskit Pulse to fractional gates") | `[published]` |
| Nighthawk r1/r2, Heron r1–r3, qubit counts | `.../guides/processor-types` | `[published]` |
| client-side primitives at `qiskit_ibm_runtime.executor_estimator` from 0.50.0 | `.../guides/max-execution-time` code sample | `[published]` |
| 1.52×10⁶ shots for the six experiments; per-experiment shot budgets; all thresholds | `docs/QPU_NITROGENASE_EXPERIMENTS.md` §3 | `[design]` (unchanged) |
| 534 s at default settings for 1.52×10⁶ executions | `2 + 0.00035 × 1 520 000` | `[analytic]` |
| 506 s / 192 s totals; 84.3 % / 31.9 % of a window; 1.71×10⁶ / 6.0×10⁶ execution ceilings; 311 s at `L=200 µs` | §5.3 model, `cost = 2 + (rep_delay + L + τ̄)·shots` | `[analytic]` on `[assumption]` `L` |
| `L` = 100/106/120/140 µs per experiment; `τ̄` = 93/46 µs for T1/T2 | our choice, anchored on IBM's "typical circuit length 50–100 µs" and its 350 µs quick formula; **must be replaced by `estimate_duration` at booking** | `[assumption]` |
| E1's ≈ 15 s of irreducible delay time | geometric delay grids from §3 of the experiment document: mean τ 93 µs (T1) / 46 µs (T2) × 100 000 / 120 000 shots | `[analytic]` |
| 3 blocks / 28 days at tuned settings; ≈ 56 under the promotion | 600/192; 10800/192 | `[analytic]` |
| dropping E4 rather than E4-q | judgement | `[interpretation]` |

---

## 12. What this document does not claim

- **No QPU job has been submitted.** Every second in §5 is arithmetic from IBM's published formula.
  No number here is `[measured]`.
- **The backend list in §2 is not the Open Plan's backend list.** It is the documented fleet; the
  per-instance list requires an account and must be read at booking.
- **`L` is an assumption, and it is the assumption the whole budget rests on.** If measured circuit
  lengths come back at 2× our estimate, the tuned plan is 52 % of a month and rule 3 of §5.4 starts
  refusing submissions. That would be a finding, not a failure.
- **The free tier does not make the science bigger.** It makes it *narrower*: fewer shots, no
  sessions, one block a month. The ceiling from §0 of the experiment document is untouched —
  **exact classical simulation, calibrated against a measured device**, with structure ≠ meaning and
  no hardware claim about the FBSC core.
- **E4 is dead on IBM.** Saying so is the deliverable, not a disappointment.
