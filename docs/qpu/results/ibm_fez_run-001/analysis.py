#!/usr/bin/env python3
"""
Derived analysis for the owner-reported ibm_fez hardware run (run-001).

Every number in RESULTS.md that is NOT a direct owner report is computed here, so a
reviewer can re-run this file and check us rather than trust us.

    python3 analysis.py            # prints a report and writes analysis_output.json

INPUTS are the owner's reported numbers only (see provenance.json). Nothing here reads
a vendor API, a job JSON, or a device. Standard library only -- no numpy, no qiskit,
so this runs anywhere and cannot accidentally become a dependency on a vendor SDK
(the PR #149 originality boundary).

TAGS used in the output, matching the repo's working agreement:
  [reported]      owner-reported number, no job artifact yet
  [analytic]      closed-form arithmetic from reported inputs
  [interpretation] a reading of the numbers, not itself a measurement
  [speculation]   a hypothesis with a stated test
"""

import json
import math

# ---------------------------------------------------------------------------
# 1. Reported inputs. Do not tune these to make the story nicer.
# ---------------------------------------------------------------------------
REPORTED = {
    "backend": "ibm_fez",
    "qubits": [44, 45],
    "chsh_measured": 2.61,
    "chsh_error": 0.02,          # owner's stated uncertainty; definition not supplied
    "chsh_design": 2.8250,       # docs/artifacts/quantum_likeness/likeness_results.json P5_no_lhv.n2.chsh
    "state_fidelity_vs_design": 0.98,  # owner's number; METRIC NOT SUPPLIED
    "freq_drift_first_khz": 1.85,
    "freq_drift_second_khz": 1.50,
    "wait_between_blocks_s": 20 * 60,
    "qpu_seconds_chsh": 39.0,
    "qpu_seconds_total": 55.0,
    "qpu_second_cap": 30.0,
}

TSIRELSON = 2.0 * math.sqrt(2.0)   # 2.8284271...
CLASSICAL = 2.0

# Device-class parameters that are NOT measured for ibm_fez here. Each is a range,
# and the analysis reports the whole range so no conclusion hangs on one guess.
DRIVE_RATE_MHZ = (5.0, 10.0, 20.0)      # Omega/2pi for a transmon pi-pulse (30-100 ns)
GATE_ANGLE_TOL = 0.01                   # 1% gate-angle error = "calibration is stale"


def _f(x, nd=6):
    return round(float(x), nd)


def chsh_section():
    S, dS = REPORTED["chsh_measured"], REPORTED["chsh_error"]
    Sd = REPORTED["chsh_design"]
    out = {
        "S_measured": S,
        "S_error_stated": dS,
        "classical_bound": CLASSICAL,
        "tsirelson": _f(TSIRELSON),
        "design": Sd,
        "violates_classical_by_sigma": _f((S - CLASSICAL) / dS, 2),
        "deficit_from_tsirelson": _f(TSIRELSON - S),
        "deficit_from_tsirelson_fraction": _f((TSIRELSON - S) / TSIRELSON, 4),
        "deficit_from_design": _f(Sd - S),
        "deficit_from_design_in_sigma": _f((Sd - S) / dS, 2),
        "tsirelson_minus_design": _f(TSIRELSON - Sd),
    }
    # Pre-registered E2 thresholds, docs/QPU_NITROGENASE_EXPERIMENTS.md section 3 E2.
    out["prereg"] = {
        "success_leg1_floor": 2.60,
        "leg1_clears_floor_by": _f(S - 2.60),
        "leg1_clears_floor_in_sigma": _f((S - 2.60) / dS, 2),
        "leg1_95pct_low": _f(S - 1.96 * dS),
        "leg1_95pct_low_inside_inconclusive_band": bool(S - 1.96 * dS < 2.60),
        "success_leg2_tolerance": 0.08,
        "leg2_evaluable": False,
        "leg2_reason": "requires S_model at E1's MEASURED channel; E1 never ran",
        "failure_threshold": 2.00,
        "verdict": "leg 1 marginal pass, leg 2 UNEVALUABLE -> not a pre-registered success",
    }

    # Werner (depolarised Bell) visibility that would produce this S.
    v = S / TSIRELSON
    F_werner = (1.0 + 3.0 * v) / 4.0
    out["werner"] = {
        "visibility_v": _f(v, 4),
        "implied_fidelity_F_eq_1p3v_over_4": _f(F_werner, 4),
        "white_noise_fraction": _f(1.0 - v, 4),
        "note": "CHSH of a Werner state is 2*sqrt(2)*v (Horodecki); F=(1+3v)/4",
    }

    # The owner's 98% figure vs the CHSH-implied 94.2%: what closes the gap?
    F_rep = REPORTED["state_fidelity_vs_design"]
    v_rep = (4.0 * F_rep - 1.0) / 3.0
    S_from_F = TSIRELSON * v_rep
    atten = S / S_from_F
    e_readout = (1.0 - math.sqrt(atten)) / 2.0
    out["fidelity_vs_chsh_consistency"] = {
        "F_reported": F_rep,
        "v_implied_by_F_if_Werner": _f(v_rep, 4),
        "S_implied_by_that_F": _f(S_from_F, 4),
        "gap_S": _f(S_from_F - S, 4),
        "gap_in_sigma": _f((S_from_F - S) / dS, 2),
        "required_correlator_attenuation": _f(atten, 4),
        "implied_per_qubit_readout_bitflip_error": _f(e_readout, 4),
        "consistent_with_typical_readout": bool(0.005 <= e_readout <= 0.030),
        "caveat": "conditional on (a) F=0.98 being a Werner-equivalent state fidelity and "
                  "(b) the whole residual being symmetric readout bit-flip. Neither is established.",
    }

    # Degeneracy: a pure frame rotation reproduces the same S with no visibility loss at all.
    cos_d = S / TSIRELSON
    out["frame_rotation_degeneracy"] = {
        "rotation_rad_that_gives_same_S": _f(math.acos(cos_d), 4),
        "rotation_deg_that_gives_same_S": _f(math.degrees(math.acos(cos_d)), 2),
        "note": "S is one number. Visibility loss and measurement-frame rotation are "
                "indistinguishable from S alone. The 3x3 correlation matrix separates them.",
    }

    # How much of the 2-qubit state does S actually pin down?
    out["identifiability_n2"] = {
        "S_depends_on": "two largest singular values t1,t2 of the 3x3 correlation matrix T: S=2*sqrt(t1^2+t2^2)",
        "pure_2q_state_parameters_up_to_local_unitaries": 1,
        "correlation_matrix_entries": 9,
        "statement": "At n=2 CHSH certifies entanglement and Bell nonlocality of WHATEVER state "
                     "was prepared. The E2 circuit (H, CNOT) prepares the textbook |Phi+> Bell "
                     "state, not the FBSC state. Any source with equal visibility gives 2.61.",
    }
    return out


def drift_section():
    f1, f2 = REPORTED["freq_drift_first_khz"], REPORTED["freq_drift_second_khz"]
    T = REPORTED["wait_between_blocks_s"]
    delta = abs(f1 - f2) * 1e3            # Hz
    span = (f1 + f2) * 1e3                # Hz
    # Two points cannot tell "drifting back toward calibration" from "drifting through it".
    r_lo = delta / T                      # slowest consistent rate
    r_hi = span / T                       # fastest consistent rate
    out = {
        "reported_drift_khz": [f1, f2],
        "wait_s": T,
        "rate_bracket_hz_per_s": [_f(r_lo, 4), _f(r_hi, 4)],
        "rate_bracket_note": "lower bound = drifting toward the calibrated value; upper bound = "
                             "drifting through it. TWO POINTS CANNOT DISTINGUISH THESE.",
        "rate_bracket_hz_per_hour": [_f(r_lo * 3600, 2), _f(r_hi * 3600, 2)],
        "relative_drift_per_hour_at_5GHz": [_f(r_lo * 3600 / 5e9, 12), _f(r_hi * 3600 / 5e9, 12)],
    }

    # (a) Does the drift hurt gate ANGLE?  Omega' = sqrt(Omega^2+delta^2) -> angle error
    #     per gate ~ (delta/Omega)^2 / 2.  At kHz on a 5-20 MHz drive this is nothing.
    gate = {}
    for MHz in DRIVE_RATE_MHZ:
        for label, dHz in (("1.50kHz", f2 * 1e3), ("1.85kHz", f1 * 1e3)):
            rel = (dHz / (MHz * 1e6)) ** 2 / 2.0
            gate[f"Omega_2pi={MHz}MHz@{label}"] = {"angle_error_per_gate": f"{rel:.3e}"}
    out["gate_angle_error"] = gate
    # time for the drift to REACH a 1% angle error, i.e. delta = sqrt(2*tol)*Omega
    half_life = {}
    for MHz in DRIVE_RATE_MHZ:
        d_tol = math.sqrt(2.0 * GATE_ANGLE_TOL) * MHz * 1e6
        half_life[f"Omega_2pi={MHz}MHz"] = {
            "delta_for_1pct_angle_error_Hz": _f(d_tol, 1),
            "days_at_slow_rate": _f(d_tol / r_lo / 86400, 2),
            "days_at_fast_rate": _f(d_tol / r_hi / 86400, 2),
        }
    out["gate_angle_half_life"] = half_life

    # (b) Does it hurt PHASE?  A free-evolution delay tau accumulates phi = 2*pi*delta_f*tau.
    phase = {}
    for tau_us in (10.0, 100.0, 150.0, 300.0):
        tau = tau_us * 1e-6
        phase[f"tau={tau_us}us"] = {
            "phase_rad_at_1.50kHz": _f(2 * math.pi * f2 * 1e3 * tau, 6),
            "phase_deg_at_1.50kHz": _f(math.degrees(2 * math.pi * f2 * 1e3 * tau), 3),
            "phase_deg_at_1.85kHz": _f(math.degrees(2 * math.pi * f1 * 1e3 * tau), 3),
            "offset_for_1deg_error_Hz": _f(math.radians(1.0) / (2 * math.pi * tau), 3),
        }
    out["phase_accumulation"] = phase
    out["phase_accumulation_note"] = (
        "T1 is a Z-basis population measurement and is INSENSITIVE to this phase. "
        "T2 Hahn echo REFOCUSES a static offset to first order -- which is why echo is used. "
        "So the observed kHz offset does not by itself invalidate E1.")

    # (c) Echo residual for drift that CHANGES during the sequence: phi_res = (pi/2)*r*tau^2.
    #     Verified numerically below rather than trusted.
    echo = {}
    for tau_us in (150.0, 300.0):
        tau = tau_us * 1e-6
        row = {}
        for r in (r_lo, r_hi):
            analytic = (math.pi / 2.0) * r * tau ** 2
            # numerical: s(t)=+1 for t<tau/2, -1 after; delta_f(t)=delta0+r*t
            N = 200001
            s = 0.0
            for i in range(N):
                t = tau * i / (N - 1)
                s += (1.0 if t < tau / 2 else -1.0) * (r * t) * (tau / (N - 1))
            numeric = 2 * math.pi * abs(s)
            echo[f"tau={tau_us}us,r={r:.3f}Hz/s"] = {
                "analytic_rad": f"{analytic:.3e}",
                "numeric_rad": f"{numeric:.3e}",
                "relative_difference": f"{abs(numeric - analytic) / max(analytic, 1e-300):.2e}",
                "agree": abs(numeric - analytic) <= 1e-2 * max(analytic, 1e-30),
            }
        echo[f"tau={tau_us}us_note"] = (
            "echo refocuses a STATIC offset completely; only the changing part survives, "
            "and it is <=1e-7 rad -- at least 6 orders below any measurable phase")
    out["echo_residual"] = echo

    # (d) Does the drift swamp E3's effect size?
    e3_drop = 0.123          # our own analytic prediction, QPU doc E3 P2 (0.967 -> 0.844)
    for block_s in (2.9, 3.6):
        d_delta = r_hi * block_s
        phi = 2 * math.pi * d_delta * 20e-6      # 20 us circuit
        out[f"e3e5_drift_vs_effect_block_{block_s}s"] = {
            "drift_during_block_Hz": _f(d_delta, 3),
            "phase_over_20us_circuit_rad": f"{phi:.3e}",
            "implied_fidelity_change": f"{phi**2:.3e}",
            "predicted_kink_drop_E3": e3_drop,
            "drift_as_fraction_of_effect": f"{phi**2/e3_drop:.3e}",
            "verdict": "drift is ~1e-5 of the pre-registered effect: E3/E5 are safe from THIS mode",
        }
    return out


def metering_section():
    """Reconcile the reported 39 s against IBM's published metering formula.

    docs/qpu/IBM_FREE_TIER_RUNBOOK.md section 5.1: usage_est = 2 + 0.00035 * executions.
    """
    S_used = REPORTED["qpu_seconds_chsh"]
    planned_shots = 4 * 50000                      # E2 as pre-registered
    N_from_time = (S_used - 2.0) / 0.00035
    est_at_planned = 2.0 + 0.00035 * planned_shots
    block_shots = 1_520_000                        # six-experiment block, runbook 5.1
    out = {
        "formula": "est = 2 + 0.00035 * executions  [published, runbook 5.1]",
        "reported_seconds": S_used,
        "cap_seconds": REPORTED["qpu_second_cap"],
        "overrun_seconds": _f(S_used - REPORTED["qpu_second_cap"]),
        "overrun_fraction": _f(S_used / REPORTED["qpu_second_cap"], 3),
        "executions_implied_by_39s": _f(N_from_time, 0),
        "estimate_for_prereg_200k_shots": _f(est_at_planned, 1),
        "observed_below_published_estimate": bool(S_used < est_at_planned),
        "implied_us_per_execution_if_200k_shots": _f((S_used - 2.0) / planned_shots * 1e6, 1),
        "published_us_per_execution": 350.0,
        "block_estimate_200k_assumed": 534.0,
    }
    # Re-cost the whole 1.52e6-shot block under the two readings.
    out["block_recost"] = {
        "at_published_350us": _f(2.0 + 0.00035 * block_shots, 0),
        "at_implied_185us_if_200k_ran": _f(2.0 + 0.000185 * block_shots, 0),
        "fits_in_300s_if_implied_rate": bool(2.0 + 0.000185 * block_shots <= 300.0),
        "caveat": "ONE data point. 39 s is consistent with (a) ~106k executions at default "
                  "metering, or (b) the planned 200k at a reduced rep_delay (~85 us). The job "
                  "JSON separates them. Do not book a plan on this yet.",
    }
    return out


def main():
    out = {
        "analysis_of": "docs/qpu/results/ibm_fez_run-001/provenance.json",
        "status_of_inputs": "OWNER-REPORTED. No job IDs, no job JSON, no shot counts. "
                            "Nothing here is [measured] and nothing here may be upgraded "
                            "without the job artifacts.",
        "reported_inputs": REPORTED,
        "tsirelson": _f(TSIRELSON),
        "chsh": chsh_section(),
        "drift": drift_section(),
        "metering": metering_section(),
    }
    print(json.dumps(out, indent=2))
    with open("analysis_output.json", "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    return out


if __name__ == "__main__":
    main()
