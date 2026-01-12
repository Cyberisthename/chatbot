"""Unit tests for real protein folding engine.

Uses only Python standard library unittest (no pytest dependency).
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import src...` works when tests are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiversal.protein_folding_engine import (
    AminoAcid,
    FoldingParameters,
    ProteinFoldingEngine,
    ProteinStructure,
    _bond_angle,
    _dist,
    _rotate_segment,
    _update_torsions_from_coords,
    _wrap_angle,
)


class TestAminoAcid(unittest.TestCase):
    def test_charge_acidic(self):
        self.assertEqual(AminoAcid("D").charge, -1.0)

    def test_charge_basic(self):
        self.assertEqual(AminoAcid("K").charge, 1.0)

    def test_charge_neutral(self):
        self.assertEqual(AminoAcid("A").charge, 0.0)

    def test_hydrophobicity(self):
        self.assertGreater(AminoAcid("I").hydrophobicity, 1.0)
        self.assertLess(AminoAcid("K").hydrophobicity, -1.0)


class TestGeometry(unittest.TestCase):
    def test_distance(self):
        a = (0.0, 0.0, 0.0)
        b = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(_dist(a, b), 5.0, places=6)

    def test_bond_angle_90deg(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        c = (1.0, 1.0, 0.0)
        self.assertAlmostEqual(_bond_angle(a, b, c), math.pi / 2, places=6)

    def test_wrap_angle(self):
        self.assertAlmostEqual(_wrap_angle(3.5), 3.5 - 2 * math.pi, places=9)
        self.assertAlmostEqual(_wrap_angle(-4.0), -4.0 + 2 * math.pi, places=9)
        self.assertAlmostEqual(_wrap_angle(0.5), 0.5, places=9)


class TestProteinStructure(unittest.TestCase):
    def test_to_dict(self):
        s = ProteinStructure(
            sequence="ABC",
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
            phi=[0.1, 0.2, 0.3],
            psi=[0.4, 0.5, 0.6],
        )
        d = s.to_dict()
        self.assertEqual(d["sequence"], "ABC")
        self.assertEqual(len(d["coords"]), 3)
        self.assertEqual(len(d["phi"]), 3)


class TestFoldingParameters(unittest.TestCase):
    def test_defaults(self):
        params = FoldingParameters()
        self.assertGreater(params.bond_length, 0)
        self.assertGreater(params.bond_k, 0)
        self.assertGreater(params.lj_epsilon, 0)


class TestProteinFoldingEngine(unittest.TestCase):
    def test_initialize_extended_chain(self):
        engine = ProteinFoldingEngine()
        structure = engine.initialize_extended_chain("ACDE", seed=42)
        self.assertEqual(structure.sequence, "ACDE")
        self.assertEqual(len(structure.coords), 4)
        self.assertEqual(len(structure.phi), 4)
        self.assertEqual(len(structure.psi), 4)

    def test_torsion_pivot_changes_geometry_without_breaking_bonds(self):
        engine = ProteinFoldingEngine()
        structure = engine.initialize_extended_chain("ACDEFGH", seed=7)

        coords_before = list(structure.coords)
        bond_before = [_dist(coords_before[i], coords_before[i + 1]) for i in range(len(coords_before) - 1)]

        # Rotate the tail around the bond between residues 1 and 2.
        _rotate_segment(structure.coords, start=3, axis_i=1, axis_j=2, angle=0.7)
        _update_torsions_from_coords(structure)

        self.assertGreater(_dist(structure.coords[4], coords_before[4]), 1e-6)

        bond_after = [_dist(structure.coords[i], structure.coords[i + 1]) for i in range(len(structure.coords) - 1)]
        for b0, b1 in zip(bond_before, bond_after):
            self.assertAlmostEqual(b0, b1, places=6)

    def test_energy_finite(self):
        engine = ProteinFoldingEngine()
        structure = engine.initialize_extended_chain("ACDE", seed=42)
        e = engine.energy(structure)
        self.assertTrue(math.isfinite(e))

    def test_bond_energy_stretched_is_positive(self):
        engine = ProteinFoldingEngine()
        params = engine.params
        coords = [(0.0, 0.0, 0.0), (params.bond_length * 1.5, 0.0, 0.0)]
        structure = ProteinStructure(sequence="AA", coords=coords, phi=[0.0, 0.0], psi=[0.0, 0.0])
        self.assertGreater(engine.energy(structure), 0.0)

    def test_nonbonded_repulsion_close_contact(self):
        engine = ProteinFoldingEngine()
        coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
            (0.5, 0.0, 0.0),
        ]
        structure = ProteinStructure(sequence="A" * 6, coords=coords, phi=[0.0] * 6, psi=[0.0] * 6)
        self.assertGreater(engine.energy(structure), 10.0)

    def test_anneal_tracks_best_energy(self):
        engine = ProteinFoldingEngine()
        initial = engine.initialize_extended_chain("ACDEFG", seed=100)
        e0 = engine.energy(initial)
        result = engine.metropolis_anneal(initial, steps=600, t_start=2.0, t_end=0.2, seed=100, log_every=300)
        self.assertTrue(math.isfinite(result["best_energy"]))
        self.assertLessEqual(result["best_energy"], e0 + 1e-6)

    def test_save_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ProteinFoldingEngine(artifacts_dir=tmpdir)
            structure = engine.initialize_extended_chain("ACE", seed=42)
            path = engine.save_artifact(run_id="test_run", payload={"test": "data", "best_structure": structure})

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["test"], "data")
            self.assertIn("best_structure", data)
            self.assertEqual(data["best_structure"]["sequence"], "ACE")


class TestRealComputation(unittest.TestCase):
    def test_different_seeds_give_different_results(self):
        engine = ProteinFoldingEngine()
        sequence = "ACDEFGH"

        energies = []
        for s in (1, 2, 3):
            r = engine.metropolis_anneal(engine.initialize_extended_chain(sequence, seed=s), steps=300, seed=s)
            energies.append(float(r["best_energy"]))

        # Stochastic search can occasionally collide; this makes the test far less flaky.
        self.assertGreater(len({round(e, 4) for e in energies}), 1)

    def test_longer_run_not_worse(self):
        engine = ProteinFoldingEngine()
        sequence = "ACDEFGH"
        initial = engine.initialize_extended_chain(sequence, seed=99)

        short = engine.metropolis_anneal(initial, steps=200, seed=99)
        long = engine.metropolis_anneal(initial, steps=1200, seed=99)

        self.assertLessEqual(long["best_energy"], short["best_energy"] + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
