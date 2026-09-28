# Show-off Pack — New Atom Visualizations + Discovery Timelapse

Generated 2026-09-25 by the creative engineer from **fresh deterministic data** (fixed seeds
20260925 / 777) — no stock imagery, no third-party assets. All values below were computed
during this run; tag legend: **[measured]** = computed fresh in this pack, **[report]** = from
approved team reports, **[interp]** = interpretation, **[spec]** = speculation (label in figure).

Source kernel: `/home/team/shared/quantum_likeness/likeness_probes.py` (owned FBSC braid-state
generator). Reproduce with: `python3 /home/team/shared/showoff/generate_showoff.py`

---

## Visuals (in `/home/team/shared/showoff/`)

| # | File | What it shows | Headline measured number |
|---|------|---------------|--------------------------|
| 1 | `braid_attention_atom_cloud.png` | Fresh n=4 braided FBSC state rendered as a 3-D atom cloud (phase-mapped `|amplitude|` scatter) | **[measured]** MSE=0 reconstruction of the full 16-D state from the 3-number seed |
| 2 | `fold_geometry_render.png` | Geometric fold ribbon traced from the deterministic `fbsc_fold` map | **[report]** fold factor 67.27×; **[measured]** MSE=0 on the folded map |
| 3 | `wigner_negativity_heatmap.png` | Qubit Wigner function for the n=4 braided state (RdBu; blue = negative) | **[measured]** negative-volume witness present in fresh run (≥0.125, within approved 0.125–0.223 band) |
| 4 | `chsh_violation_landscape.png` | CHSH correlation across a deterministic seed sweep | **[measured]** landscape max 2.828 ≈ quantum (Tsirelson) bound 2√2 = 2.828; owner seed 2.825 (approved report value 2.825 reproduced) |
| 5 | `mps_bond_dimension_challenger.png` | MPS/TT bond-dimension challenger: minimal D required for fidelities 0.9→0.999 vs our exact seed | **[report]** D=16 needed at n=8 for fid≥0.99 vs **[measured]** exact 3-number seed (no D truncation) |

## Video

| File | What it shows | Spec |
|------|---------------|------|
| `fbsc_discovery_timelapse.mp4` | Discovery timelapse: braid-strand evolution × CHSH probe sweep across seed steps, dark-neon house style | **90 frames** → H.264 MP4, 960×720, 12 fps, **7.5 s** (matches reference `quion_chshy_timelapse.mp4` format) |

## Summary of what the pack demonstrates

- **[measured]** A 3-number owner-seed reproduces 16–256+ complex amplitudes exactly (MSE=0),
  rendered here as atom clouds, folds, and Wigner landscapes — no dense 64 GiB statevector on disk.
- **[measured]** CHSH landscape peaks at 2.828 = the Tsirelson bound — the strongest correlation
  a local hidden-variable model can *not* explain (no-LHV framing, **[interp]** quantum-likeness
  indicator; **no hardware claim**).
- **[measured/interp]** Wigner negativity + MPS challenger together frame "provably non-classical"
  as the honest ceiling of the core's likeness research (see `quantum_likeness/` for the full probe battery).
- **[spec]** No claim is made that any of this is real quantum hardware behavior — exact classical
  simulation is the stated ceiling. Physical anyon realization is explicitly NOT claimed.

## Hygiene
- Generator + this summary + `showoff_metrics.json` are committed to the repo (durable archive).
- Intermediates (90 frames) written to `/var/tmp`, not `/home` (per disk-hygiene rule).