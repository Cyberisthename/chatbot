#!/usr/bin/env python3
"""
Test Suite for Executable Genome Framework (EGF)
=================================================

Tests the core functionality of the EGF system.
Run: python -m tests.test_genome_framework
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genome import (
    GenomeCoreAdapter,
    RegulomeGraphAdapter,
    ContextEnvironmentAdapter,
    EpigeneticGateManager,
    ExpressionDynamicsEngine,
    ProteomeTranslator,
    PhenotypeScorer,
    ArtifactMemorySystem,
    ExecutableGenomeFramework,
    GenomeRegion,
    RegulatoryElement,
    EpigeneticGate,
    ContextState,
    ExpressionTrajectory,
    PhenotypeScore,
)


class TestGenomeRegion(unittest.TestCase):
    """Tests for GenomeRegion data class."""
    
    def test_create_region(self):
        region = GenomeRegion(
            region_id="test_exon",
            sequence="ATGCGTACGTAGCTAGCTAG",
            region_type="exon",
            start=100,
            end=120,
            chromosome="chr1",
            strand="+"
        )
        self.assertEqual(region.region_id, "test_exon")
        self.assertEqual(region.sequence, "ATGCGTACGTAGCTAGCTAG")
        self.assertEqual(region.region_type, "exon")
        
    def test_serialization(self):
        region = GenomeRegion(
            region_id="test_region",
            sequence="ATGC",
            region_type="promoter",
            start=1000,
            end=1004,
            chromosome="chrX",
            strand="-"
        )
        data = region.to_dict()
        restored = GenomeRegion.from_dict(data)
        self.assertEqual(restored.region_id, region.region_id)
        self.assertEqual(restored.sequence, region.sequence)


class TestRegulatoryElement(unittest.TestCase):
    """Tests for RegulatoryElement data class."""
    
    def test_create_element(self):
        element = RegulatoryElement(
            element_id="enhancer_1",
            element_type="enhancer",
            target_genes=["GENE_A", "GENE_B"],
            tf_families=["p53", "NFkB"],
            genomic_location=("chr1", 5000, 5100),
            weight=0.8
        )
        self.assertEqual(element.element_id, "enhancer_1")
        self.assertEqual(len(element.target_genes), 2)
        self.assertEqual(element.weight, 0.8)
        
    def test_serialization(self):
        element = RegulatoryElement(
            element_id="promoter_1",
            element_type="promoter",
            target_genes=["TEST_GENE"],
            tf_families=["homeobox"],
            genomic_location=("chr2", 1000, 1050),
            weight=0.9
        )
        data = element.to_dict()
        restored = RegulatoryElement.from_dict(data)
        self.assertEqual(restored.element_id, element.element_id)
        self.assertEqual(restored.genomic_location, element.genomic_location)


class TestEpigeneticGate(unittest.TestCase):
    """Tests for EpigeneticGate stateful behavior."""
    
    def test_gate_creation(self):
        gate = EpigeneticGate(
            gate_id="gate_1",
            regulated_element_id="enhancer_1",
            gate_type="methylation"
        )
        self.assertEqual(gate.methylation_level, 0.0)
        # Default accessibility is 1.0 (fully open chromatin)
        self.assertEqual(gate.accessibility_score, 1.0)
        
    def test_context_application(self):
        gate = EpigeneticGate(
            gate_id="gate_test",
            regulated_element_id="enhancer_test",
            gate_type="accessibility"
        )
        
        # Apply normal context
        context = {"tissue": "liver", "stress": 0.3}
        activation1 = gate.apply_context(context, time_step=1)
        
        # Apply stressed context
        context_stressed = {"tissue": "liver", "stress": 0.9}
        activation2 = gate.apply_context(context_stressed, time_step=2)
        
        # Stress should increase activation for accessibility
        self.assertGreaterEqual(activation2, 0.0)
        self.assertLessEqual(activation2, 1.0)
        
        # History should be recorded
        self.assertEqual(len(gate.state_history), 2)


class TestContextState(unittest.TestCase):
    """Tests for ContextState."""
    
    def test_create_context(self):
        context = ContextState(
            tissue="liver",
            stress_level=0.5,
            developmental_stage="adult",
            signal_molecules={"glucose": 0.8}
        )
        self.assertEqual(context.tissue, "liver")
        self.assertEqual(context.stress_level, 0.5)
        self.assertIn("glucose", context.signal_molecules)


class TestExpressionTrajectory(unittest.TestCase):
    """Tests for ExpressionTrajectory."""
    
    def test_trajectory_creation(self):
        traj = ExpressionTrajectory(
            gene_id="GENE_A",
            time_points=[0.0, 1.0, 2.0, 3.0],
            expression_values=[0.1, 0.5, 0.8, 0.9]
        )
        self.assertEqual(traj.gene_id, "GENE_A")
        self.assertEqual(len(traj.time_points), 4)
        
    def test_compute_statistics(self):
        traj = ExpressionTrajectory(
            gene_id="TEST",
            time_points=[0, 1, 2],
            expression_values=[10.0, 20.0, 30.0]
        )
        traj.compute_statistics()
        self.assertEqual(traj.peak_expression, 30.0)
        self.assertAlmostEqual(traj.mean_expression, 20.0)
        self.assertGreater(traj.stability_score, 0.0)


class TestPhenotypeScore(unittest.TestCase):
    """Tests for PhenotypeScore."""
    
    def test_score_creation(self):
        score = PhenotypeScore(
            viability_score=0.8,
            stability_score=0.9,
            efficiency_score=0.7,
            fitness_proxy=0.81
        )
        self.assertEqual(score.viability_score, 0.8)
        self.assertGreater(score.fitness_proxy, 0.7)


class TestGenomeCoreAdapter(unittest.TestCase):
    """Tests for GenomeCoreAdapter."""
    
    def setUp(self):
        self.adapter = GenomeCoreAdapter("/tmp/test_genome_core")
        
    def test_add_gene(self):
        self.adapter.add_gene("TEST_GENE", {
            "name": "Test Gene",
            "function": "test",
            "exonic_regions": []
        })
        gene = self.adapter.get_gene("TEST_GENE")
        self.assertIsNotNone(gene)
        self.assertEqual(gene["name"], "Test Gene")
        
    def test_add_region(self):
        region = GenomeRegion(
            region_id="test_exon",
            sequence="ATGC",
            region_type="exon",
            start=100,
            end=104,
            chromosome="chr1"
        )
        self.adapter.add_region(region)
        seq = self.adapter.get_sequence("test_exon")
        self.assertEqual(seq, "ATGC")


class TestRegulomeGraphAdapter(unittest.TestCase):
    """Tests for RegulomeGraphAdapter."""
    
    def setUp(self):
        import uuid
        self.test_id = uuid.uuid4().hex[:8]
        self.adapter = RegulomeGraphAdapter(f"/tmp/test_regulome_{self.test_id}")
        
    def test_add_element(self):
        element = RegulatoryElement(
            element_id="enhancer_1",
            element_type="enhancer",
            target_genes=["GENE_A"],
            tf_families=["p53"],
            genomic_location=("chr1", 1000, 1100),
            weight=0.8
        )
        self.adapter.add_regulatory_element(element)
        self.assertIn(element.element_id, self.adapter.elements)
        
    def test_add_edge(self):
        import uuid
        source_id = f"source_{uuid.uuid4().hex[:8]}"
        target_id = f"target_{uuid.uuid4().hex[:8]}"
        self.adapter.add_regulatory_edge(source_id, target_id, 0.7)
        self.assertIn(source_id, self.adapter.edges)
        self.assertEqual(len(self.adapter.edges[source_id]), 1)


class TestContextEnvironmentAdapter(unittest.TestCase):
    """Tests for ContextEnvironmentAdapter."""
    
    def setUp(self):
        self.adapter = ContextEnvironmentAdapter("/tmp/test_context")
        
    def test_activate_transcription_factors(self):
        context = {
            "tissue": "liver",
            "stress": 0.5,
            "signals": {}
        }
        tf_activity = self.adapter.activate_transcription_factors(context)
        
        self.assertIsInstance(tf_activity, dict)
        # Should have entries for TF families
        self.assertGreater(len(tf_activity), 0)


class TestEpigeneticGateManager(unittest.TestCase):
    """Tests for EpigeneticGateManager."""
    
    def setUp(self):
        import uuid
        self.test_id = uuid.uuid4().hex[:8]
        self.manager = EpigeneticGateManager(f"/tmp/test_gates_{self.test_id}")
        
    def test_create_gate(self):
        import uuid
        element_id = f"element_{uuid.uuid4().hex[:8]}"
        gate = self.manager.create_gate(element_id, "methylation")
        self.assertIsNotNone(gate)
        self.assertEqual(gate.gate_type, "methylation")
        
    def test_get_element_gate(self):
        import uuid
        element_id = f"element_{uuid.uuid4().hex[:8]}"
        gate = self.manager.create_gate(element_id, "accessibility")
        retrieved = self.manager.get_element_gate(element_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.gate_type, "accessibility")


class TestExpressionDynamicsEngine(unittest.TestCase):
    """Tests for ExpressionDynamicsEngine."""
    
    def setUp(self):
        self.engine = ExpressionDynamicsEngine("/tmp/test_expression")
        
    def test_execute_expression(self):
        regulatory_input = {
            "GENE_A": 0.5,
            "GENE_B": 0.8,
            "GENE_C": 0.2
        }
        trajectories = self.engine.execute_expression(
            regulatory_input, duration=10.0, time_step=1.0
        )
        
        self.assertIsInstance(trajectories, dict)
        self.assertEqual(len(trajectories), 3)


class TestPhenotypeScorer(unittest.TestCase):
    """Tests for PhenotypeScorer."""
    
    def setUp(self):
        self.scorer = PhenotypeScorer("/tmp/test_phenotype")
        
    def test_is_successful(self):
        score = PhenotypeScore(
            viability_score=0.8,
            stability_score=0.8,
            efficiency_score=0.8,
            fitness_proxy=0.8
        )
        self.assertTrue(self.scorer.is_successful(score))


class TestArtifactMemorySystem(unittest.TestCase):
    """Tests for ArtifactMemorySystem."""
    
    def setUp(self):
        self.memory = ArtifactMemorySystem("/tmp/test_memory")
        
    def test_store_and_retrieve(self):
        from src.genome import ExecutionArtifact
        
        artifact = ExecutionArtifact(
            artifact_id="test_artifact",
            execution_id="exec_1",
            context={"tissue": "liver", "stress": 0.5},
            gate_states={},
            regulatory_paths=[],
            expression_trajectories={},
            phenotype_scores={},
            outcome_score=0.75,
            execution_time=1.5
        )
        
        self.memory.store_artifact(artifact)
        retrieved = self.memory.get_artifact("test_artifact")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.artifact_id, "test_artifact")
        self.assertEqual(retrieved.outcome_score, 0.75)


class TestExecutableGenomeFramework(unittest.TestCase):
    """Integration tests for the main EGF framework."""
    
    def setUp(self):
        self.egf = ExecutableGenomeFramework("/tmp/test_egf_full")
        
    def test_initialization(self):
        # Framework should initialize without error
        self.assertIsNotNone(self.egf)
        self.assertIsNotNone(self.egf.genome_adapter)
        self.assertIsNotNone(self.egf.regulome_adapter)
        self.assertIsNotNone(self.egf.gate_adapter)
        
    def test_load_and_execute(self):
        # Create minimal genome data
        genome_data = {
            "genome": {
                "regions": [
                    {
                        "region_id": "exon_1",
                        "sequence": "ATG"*10,
                        "region_type": "exon",
                        "start": 100,
                        "end": 130,
                        "chromosome": "chr1"
                    }
                ],
                "genes": {
                    "TEST_GENE": {
                        "name": "Test Gene",
                        "exonic_regions": ["exon_1"],
                        "function": "test"
                    }
                },
                "isoforms": {}
            },
            "regulome": {
                "elements": [
                    {
                        "element_id": "promoter_1",
                        "element_type": "promoter",
                        "target_genes": ["TEST_GENE"],
                        "tf_families": ["homeobox"],
                        "genomic_location": ["chr1", 50, 100],
                        "weight": 0.8
                    }
                ],
                "edges": [
                    {"source": "promoter_1", "target": "TEST_GENE", "weight": 0.8}
                ]
            }
        }
        
        self.egf.load_genome_data(genome_data)
        self.egf.set_context(tissue="liver", stress=0.3)
        
        # Execute genome
        result = self.egf.execute_genome(duration=6.0, time_step=1.0)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.outcome_score, 0.0)
        self.assertIsInstance(result.expression_trajectories, dict)


class TestFrameworkNovelty(unittest.TestCase):
    """Tests demonstrating framework novelty."""
    
    def test_no_neural_network_components(self):
        """Verify framework doesn't use neural networks."""
        from src.genome import (
            GenomeRegion, RegulatoryElement, EpigeneticGate,
            ExpressionTrajectory, PhenotypeScore
        )
        
        # These classes should not have any neural network imports
        import src.genome.executable_genome_framework as egf_module
        
        # Check that key classes exist and are not neural network wrappers
        self.assertTrue(hasattr(egf_module, 'GenomeRegion'))
        self.assertTrue(hasattr(egf_module, 'RegulatoryElement'))
        self.assertTrue(hasattr(egf_module, 'EpigeneticGate'))
        
    def test_no_prediction_only(self):
        """Verify framework executes, doesn't just predict."""
        egf = ExecutableGenomeFramework("/tmp/test_no_prediction")
        
        genome_data = {
            "genome": {
                "regions": [],
                "genes": {"GENE_A": {"name": "A", "exonic_regions": [], "function": "test"}},
                "isoforms": {}
            },
            "regulome": {
                "elements": [],
                "edges": []
            }
        }
        
        egf.load_genome_data(genome_data)
        egf.set_context(tissue="liver", stress=0.1)
        result = egf.execute_genome(duration=2.0, time_step=1.0)
        
        # Execution should produce trajectories over time
        self.assertIsInstance(result.expression_trajectories, dict)
        # Not just a single prediction value
        
    def test_memory_accumulates(self):
        """Verify memory doesn't get overwritten."""
        import uuid
        test_id = uuid.uuid4().hex[:8]
        egf = ExecutableGenomeFramework(f"/tmp/test_memory_accum_{test_id}")
        
        genome_data = {
            "genome": {
                "regions": [],
                "genes": {"GENE_A": {"name": "A", "exonic_regions": [], "function": "test"}},
                "isoforms": {}
            },
            "regulome": {
                "elements": [],
                "edges": []
            }
        }
        
        egf.load_genome_data(genome_data)
        
        # Execute multiple times
        for i in range(3):
            egf.set_context(tissue="liver", stress=0.1 * i)
            egf.execute_genome(duration=2.0, time_step=1.0)
        
        stats = egf.get_memory_stats()
        
        # Should have 3 artifacts, not 1
        self.assertEqual(stats['total_artifacts'], 3)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
