# `docs/qpu/results/` — hardware results, and the rules for what counts as one

This directory holds results that came from real quantum hardware. Everything else under
`docs/qpu/` is design, pre-registration or costing. The separation exists so that a reader can
tell, at a glance, what has been run from what has been planned.

## The status ladder

Every result carries exactly one of these tags. They are not interchangeable and they are not
decorative — the whole point of this directory is that the tag is honest.

| tag | means | what it licenses |
|---|---|---|
| `OWNER-REPORTED` | numbers supplied by the owner in prose; **no job artifact attached** | recording and analysis. **Not citable** as a measurement, inside or outside the repo |
| `MEASURED` | job IDs + job JSON + shots + settings + covering calibration record all attached and committed | citable, with device, qubits, run date and calibration timestamp stated in the same sentence |
| `OWED` | an experiment that was supposed to return a result and did not | nothing; recorded so the gap is visible |
| `VOID` | ran, but the run cannot support its claim (identical compiled depths, aborted block, etc.) | report it as loudly as a hit — precedent: `docs/validation/BLIND_PREDICTION_2.md` |

**Promotion from `OWNER-REPORTED` to `MEASURED` requires all of:**

1. the job IDs for every block,
2. the job JSON for each, committed or hashed in the result directory,
3. shot counts and measurement settings **read out of the job JSON**, not remembered,
4. the backend calibration record covering the run window.

Anything less and the tag stays. A remembered shot count is not a provenance field.

## Index

| results id | device | date | tag | what it contains |
|---|---|---|---|---|
| [`ibm_fez_run-001`](ibm_fez_run-001/RESULTS.md) | IBM `ibm_fez`, qubits 44–45 | unknown | `OWNER-REPORTED` | CHSH = 2.61 ± 0.02; the autonomous 1.85 → 1.50 kHz drift; spend; the chemistry check, owed |

## Why this directory exists at all

The repo's working agreement says a claim a skeptic can disprove with one grep is a liability, and
that findings which weaken an approved claim are the deliverable. Hardware results are the highest-
risk class of claim we can make, for three reasons: they cannot be re-run by a reviewer, they carry
an implicit authority ("it was measured") that our simulations deliberately do not, and they invite
exactly the over-reading we have spent the project refusing — new physics, quantum advantage,
"the QPU computed it".

So the rules for a result in this directory are stricter than anywhere else in the repo:

1. **A result with no artifact is `OWNER-REPORTED`, whatever its quality.** R1's numbers are
   internally interesting and we analyse them properly — under the correct tag.
2. **State the ceiling in the result itself**, not in a document a reader has to find. Every
   `RESULTS.md` here has a "what this is not" section before its first number.
3. **Derived numbers must be re-runnable.** Each directory carries an `analysis.py` (standard
   library only, so it can never become a vendor-SDK dependency) that recomputes every non-reported
   figure from the reported inputs.
4. **Null is a value.** A provenance file with `"shots": null` and a `promotion_blocked_by` list is
   more useful than one that omits the field and lets a reader assume.

## Framing rule (unchanged, applies to every result here)

From `docs/QPU_NITROGENASE_EXPERIMENTS.md` and `docs/qpu/nitrogenase_prereg_TEMPLATE.json`:
**hardware calibrates or validates our simulator; hardware never computes the chemistry.** No
sentence of the form "the QPU produced X kJ/mol" is permitted. No hardware result in this directory
moves the permanent ceiling — FeMo-co's 54-orbital active space at ~10¹¹ T gates, and the d = 4,
n = 16 state's 4¹⁶ amplitudes with no known state-prep circuit. The ceiling is physical, not
budgetary, and no result here changes it.
