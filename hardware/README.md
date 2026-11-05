# `hardware/` — QPU device calibration sandbox

**Status: boundary only. Nothing here has run. No QPU job has been submitted by this repository.**
Every number in `docs/qpu/` is arithmetic from IBM's published metering formula — none of it is
`[measured]` on a device.

## What this directory is

This is **calibration of a third-party device; it is not part of the owned stack.**

The owner decided to use IBM Quantum's free tier, which means Qiskit — a third-party quantum SDK.
The JARVIS owned core (the Fractal-Braid Seed Compressor, the braid/topological machinery, the
quantum transformer, the TCL engine) is original code built on numpy + the Python standard library,
and it stays that way. This directory exists so that the hardware work can happen **without**
touching that claim.

## The two rules that make the claim literally true

1. **`hardware/` is never imported by the FBSC path.** Not by `src/`, not by `scripts/`, not by the
   R&D battery server, not by the lab site. It is a **leaf**: nothing imports it, it imports nothing
   of ours that isn't a pure-stdlib helper. Enforced by `hardware/check_boundary.sh`.
2. **Nothing outside `hardware/` imports `qiskit`, `qiskit_ibm_runtime` or `qiskit_aer`.** Enforced
   by the same script.

`hardware/` is a **second, named exception** to the third-party-dependency rule — **not a loosening
of the first**. The `legacy/` boundary keeps its rule unchanged: zero third-party imports outside
`legacy/` still holds, and `hardware/` does not relax it. Two sandboxes, two names, one rule each.

## What may cross the boundary

**JSON, and only JSON.** Hardware runs produce JSON under `docs/artifacts/qpu/<UTC-date>/`; the
owned core may read that JSON the way it reads any other data file. **No result crosses back as
code**, no vendor object is returned to the owned path, and no vendor package becomes a dependency
of `src/`.

## What this buys us, and what it does not

| | |
|---|---|
| **Earned** | *device calibration* and *measured device behaviour* — T1/T2, readout error, CHSH measured on hardware, measured depth decay. |
| **NOT earned** | any claim that **our core ran on a QPU**. The FBSC state is an exact classical simulation. Device decay versus an exact classical program is not an advantage. |
| **NOT earned** | any d ≥ 3 qudit result on IBM. IBM deprecated pulse-level control (removed early 2025; `qiskit.pulse` removed in Qiskit 2.0), so qudit access is **unobtainable there, not unaffordable**. We did not run it and must never word the audit as if we had. |

## Rules of the sandbox

1. **No vendor code is vendored.** MIT covers our files; it does not cover IBM's SDK terms, and it
   does not cover the name "J.A.R.V.I.S." (separate, unresolved owner decision).
2. **The token is an environment variable** (`QISKIT_IBM_TOKEN`), read at runtime. Never in a file,
   never in a commit, never in `.env`, never in a notebook output.
3. **No raw seeds here.** The 3-seed key never enters this directory; hardware work consumes frozen
   pre-registered parameter sets, not the seed.
4. **Pin versions at booking, not now.** `requirements.txt` is deliberately unpinned — no account has
   been used, so any version written today would be invented. Pin both packages at booking and
   record them in the pre-registration (`qiskit_version`, `qiskit_ibm_runtime_version`), per
   `docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §10.

## The evidence (this is what a reviewer runs)

```bash
bash hardware/check_boundary.sh    # exits non-zero if the boundary is crossed; runs in CI
```

It must print `BOUNDARY OK`. The underlying grep, if you want to run it by hand:

```bash
# must print nothing
grep -rn --include='*.py' -E 'qiskit|qiskit_ibm_runtime|qiskit_aer' . \
  | grep -v '^./hardware/' | grep -v '^./legacy/' | grep -v '^./node_modules/'
```

**A boundary that is only a paragraph in a README is a boundary that will be crossed.** That is why
the grep is a script and the script is in CI.

## References

- `docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §9 — the sandbox rule, layout, and the wording the audit uses.
- `docs/artifacts/compliance/ORIGINALITY_AUDIT.md` §4.6 — the disclosure this directory exists to keep true.
- `docs/artifacts/legal/GOVERNANCE_GATE.md` A8 — the ownership criterion this boundary enforces.
- `docs/QPU_NITROGENASE_EXPERIMENTS.md` §7 — the legal boundary, stated before the hardware was chosen.
