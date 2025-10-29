"""Run a bundle of flagship experiments and collate their summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from quantacap.experiments.chsh import run_chsh
from quantacap.experiments.pi.couple import run_pi_coupling
from quantacap.experiments.pi.entropy import run_pi_entropy_control
from quantacap.experiments.pi.noise import run_pi_entropy_collapse
from quantacap.experiments.atom1d import run_atom1d
from quantacap.experiments.med.docking import run_search
from quantacap.astro.lensing import run_lens_atom_equivalence
from quantacap.viz3d.fieldmap import build_field_series
from quantacap.viz3d.scene import export_scene


def _ensure_artifacts() -> Path:
    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> Dict[str, object]:
    _ensure_artifacts()
    summary: Dict[str, object] = {}

    summary["chsh_clean"] = run_chsh(shots=20000, depol=0.0, seed=424242)
    summary["pi_couple"] = run_pi_coupling(kappa=0.03, steps=2000, adapter_id="pi.couple.bundle")
    summary["pi_entropy"] = run_pi_entropy_control(steps=2000, adapter_id="pi.entropy.bundle")
    summary["pi_collapse"] = run_pi_entropy_collapse(
        kappa=0.02,
        sigma_min=1e-9,
        sigma_max=1e-7,
        stages=8,
        stage_length=64,
        adapter_id="pi.collapse.bundle",
    )

    atom = run_atom1d(n=8, L=4.0, sigma=0.7, adapter_id="atom.bundle")
    summary["atom1d"] = atom

    summary["med"] = run_search(
        "ACE2",
        cycles=200,
        topk=5,
        seed=4242,
        adapter_id="med.bundle",
        pi_adapter="pi.entropy.bundle",
    )

    summary["astro_equivalence"] = run_lens_atom_equivalence(
        resolution=32,
        impact_min=2.5,
        impact_max=6.0,
        atom_artifact=atom["artifact"],
        adapter_id="astro.equiv.bundle",
    )

    frames = list(
        build_field_series(
            source="adapter:pi.collapse.bundle",
            field="phase",
            grid=(16, 16, 16),
            steps=12,
        )
    )
    viz_meta = export_scene(frames, out_prefix="artifacts/master_compmap")
    summary["compmap"] = viz_meta

    summary_path = Path("artifacts/master_discovery_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    try:
        import runpy

        runpy.run_module("quantacap.scripts.report_discoveries", run_name="__main__")
    except Exception:
        # The report script already prints detailed diagnostics; failure here is non-fatal.
        pass

    print(json.dumps({"summary_path": str(summary_path), "keys": list(summary.keys())}, indent=2))
    return summary


if __name__ == "__main__":
    main()
