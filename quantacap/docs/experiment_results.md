# Quantacap Experiment Results Snapshot

This document captures the quantitative highlights from the latest synthetic quantum experiments. Each section cites the artifact that records the underlying data so runs can be reproduced or replayed directly from the saved adapters.

## CHSH Bell Inequality

The clean Bell test reaches the quantum-optimal violation of the CHSH bound with an observed value of \(S = 2.8284\). Each correlator remains at approximately \(\pm \tfrac{\sqrt{2}}{2}\) with 50,000 shots per measurement setting, resulting in balanced outcome counts for the four basis choices.【F:quantacap/artifacts/chsh_clean.json†L1-L32】

Injecting depolarising noise with strength \(p = 0.1\) lowers the correlators to about 0.573 and reduces the aggregate Bell parameter to \(S \approx 2.291\), still above the classical limit of 2 but clearly demonstrating the visibility loss introduced by the channel.【F:quantacap/artifacts/chsh_depol_0_1.json†L1-L32】

## Atom-1D Density Reconstruction

The synthetic atom experiment stores a Gaussian-like wavefunction over an 8-qubit grid. The replayed density remains sharply peaked yet normalised, with probabilities spanning from \(1.67\times10^{-16}\) to \(2.53\times10^{-2}\), a mean of \(3.91\times10^{-3}\), and variance \(5.46\times10^{-5}\).【F:quantacap/artifacts/summary_results.json†L1-L13】

## π-Phase Stability Run

A 100,000-rotation π-phase stabilisation run (precision \(10^{-12}\)) maintains a mean rotation equal to π with a standard deviation near \(1.02\times10^{-12}\). Roughly 69% of steps fall within the precision window, 99.8% remain locked inside a \(\pm10^{-11}\) band, and the RMS phase error stays at \(3.13\times10^{-12}\). The run is flagged as stabilised with the logged stability metrics preserved for replay.【F:quantacap/artifacts/pi_phase_demo.json†L1-L24】

---

To regenerate these numbers, rerun the corresponding CLI commands and inspect the referenced artifacts. Each experiment also persists adapter records so outcomes can be replayed without recomputation.
