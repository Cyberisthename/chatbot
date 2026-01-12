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
    _angular_distance_deg,
    _bond_angle,
    _dist,
    _ramachandran_prior_energy,
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

    def test_angular_distance_deg(self):
        # Test basic distance
        self.assertAlmostEqual(_angular_distance_deg(30, 60), 30, places=6)
        # Test wrap-around at 360
        self.assertAlmostEqual(_angular_distance_deg(350, 10), 20, places=6)
        # Test wrap-around at -360
        self.assertAlmostEqual(_angular_distance_deg(-10, 10), 20, places=6)


class TestRamachandranPriors(unittest.TestCase):
    def test_alpha_basin_favored(self):
        """Alpha-helix basin should be favored for general residues."""
        phi = math.radians(-60)  # Alpha basin center
        psi = math.radians(-45)

        e = _ramachandran_prior_energy(phi, psi, "general")

        # Energy should be relatively low (favored)
        # Compare to energy at a disfavored region
        e_disfavored = _ramachandran_prior_energy(math.radians(0), math.radians(0), "general")
        self.assertLess(e, e_disfavored)

    def test_beta_basin_favored(self):
        """Beta-sheet basin should be favored for general residues."""
        phi = math.radians(-135)  # Beta basin center
        psi = math.radians(135)

        e = _ramachandran_prior_energy(phi, psi, "general")

        # Energy should be relatively low (favored)
        # Compare to energy at a disfavored region
        e_disfavored = _ramachandran_prior_energy(math.radians(0), math.radians(0), "general")
        self.assertLess(e, e_disfavored)

    def test_disfavored_region(self):
        """Regions outside favored basins should have higher energy."""
        # Pick a random point outside known basins
        phi = math.radians(0)
        psi = math.radians(0)

        e_general = _ramachandran_prior_energy(phi, psi, "general")
        e_alpha = _ramachandran_prior_energy(phi, psi, "general")  # Same

        # Should be higher than the favored basin energies
        e_alpha_basin = _ramachandran_prior_energy(math.radians(-60), math.radians(-45), "general")
        self.assertGreater(e_general, e_alpha_basin)

    def test_glycine_more_permissive(self):
        """Glycine should have more permissive priors."""
        phi = math.radians(60)
        psi = math.radians(0)

        e_gly = _ramachandran_prior_energy(phi, psi, "glycine")
        e_general = _ramachandran_prior_energy(phi, psi, "general")

        # Glycine energy should be lower (more favorable) for this configuration
        self.assertLess(e_gly, e_general)

    def test_proline_restricted(self):
        """Proline should have restricted conformational space."""
        phi = math.radians(-75)
        psi = math.radians(150)  # PPII-like region

        e_pro = _ramachandran_prior_energy(phi, psi, "proline")
        e_general = _ramachandran_prior_energy(phi, psi, "general")

        # Proline energy should be lower (more favorable) in PPII region
        self.assertLess(e_pro, e_general)


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

    def test_internal_coordinate_propagation(self):
        """Test that changing torsion angles actually updates coordinates."""
        engine = ProteinFoldingEngine()
        structure = engine.initialize_extended_chain("AAAAAA", seed=42)

        # Save original coordinates
        orig_coords = list(structure.coords)

        # Apply a pivot move
        import random
        rng = random.Random(123)
        engine._pivot_move(structure, pivot_idx=2, max_delta=math.radians(30.0), rng=rng)

        # Coordinates should have changed
        new_coords = list(structure.coords)
        # At least some coordinates should be different
        changed = False
        for i in range(len(structure.sequence)):
            if i < 2:
                # Before pivot, coords should be same
                self.assertAlmostEqual(new_coords[i][0], orig_coords[i][0], places=6)
                self.assertAlmostEqual(new_coords[i][1], orig_coords[i][1], places=6)
                self.assertAlmostEqual(new_coords[i][2], orig_coords[i][2], places=6)
            else:
                # After pivot, coords may be different
                dx = abs(new_coords[i][0] - orig_coords[i][0])
                dy = abs(new_coords[i][1] - orig_coords[i][1])
                dz = abs(new_coords[i][2] - orig_coords[i][2])
                if dx > 1e-6 or dy > 1e-6 or dz > 1e-6:
                    changed = True

        self.assertTrue(changed, "Coordinates should have changed after pivot move")

    def test_crankshaft_preserves_connectivity(self):
        """Test that crankshaft move maintains bond lengths."""
        engine = ProteinFoldingEngine()
        structure = engine.initialize_extended_chain("AAAAAAAA", seed=42)

        import random
        rng = random.Random(456)

        # Get initial bond lengths
        n = len(structure.sequence)
        orig_bond_lengths = []
        for i in range(n - 1):
            orig_bond_lengths.append(_dist(structure.coords[i], structure.coords[i + 1]))

        # Apply crankshaft move
        engine._crankshaft_move(structure, start_idx=2, end_idx=5, max_delta=math.radians(15.0), rng=rng)

        # Check bond lengths are still close to target
        for i in range(n - 1):
            bl = _dist(structure.coords[i], structure.coords[i + 1])
            # Bond lengths should be approximately preserved (within small tolerance)
            # The rebuild process maintains bond length exactly
            self.assertAlmostEqual(bl, engine.params.bond_length, places=5)


class TestRealComputation(unittest.TestCase):
    def test_different_seeds_give_different_results(self):
        """Test that different random seeds produce different folding outcomes.

        This runs 3 different seeds and checks that not all produce the same result.
        More robust than comparing just 2 seeds (which could coincidentally be similar).
        """
        engine = ProteinFoldingEngine()
        sequence = "ACDEFGH"

        results = []
        for seed in [1, 2, 3]:
            r = engine.metropolis_anneal(
                engine.initialize_extended_chain(sequence, seed=seed),
                steps=500,
                seed=seed,
            )
            results.append(r["best_energy"])

        # Check that not all energies are the same (within a small tolerance)
        # This is statistically unlikely for true randomness
        e0, e1, e2 = results
        # At least one pair should be significantly different
        self.assertTrue(
            abs(e0 - e1) > 1e-3 or abs(e0 - e2) > 1e-3 or abs(e1 - e2) > 1e-3,
            f"All three seeds produced similar energies: {e0}, {e1}, {e2}",
        )

    def test_longer_run_not_worse(self):
        engine = ProteinFoldingEngine()
        sequence = "ACDEFGH"
        initial = engine.initialize_extended_chain(sequence, seed=99)

        short = engine.metropolis_anneal(initial, steps=200, seed=99)
        long = engine.metropolis_anneal(initial, steps=1200, seed=99)

        # Longer optimization should not be significantly worse (may be equal)
        self.assertLessEqual(long["best_energy"], short["best_energy"] + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
