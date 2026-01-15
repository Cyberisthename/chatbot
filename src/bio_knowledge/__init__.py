"""
Biological Knowledge Integration for Cancer Hypothesis Generation

This module integrates real biological data with Thought-Compression Language (TCL)
and Quantum H-bond protein folding to generate novel cancer treatment hypotheses.

WARNING: This is real scientific software. All data and calculations are based on
actual biochemical principles and published scientific knowledge.
"""

from .biological_database import (
    BiologicalKnowledgeBase,
    Protein,
    CancerPathway,
    Drug,
    MolecularInteraction
)

from .cancer_hypothesis_generator import (
    CancerHypothesisGenerator,
    Hypothesis,
    HypothesisMetrics,
    CausalChain
)

from .tcl_quantum_integrator import (
    TCLQuantumIntegrator,
    QuantumProteinAnalysis,
    TCLCausalChain
)

from .dna_generator import DNAGenerator, DNASequence
from .digital_pipeline import run_pipeline

__all__ = [
    "BiologicalKnowledgeBase",
    "Protein",
    "CancerPathway",
    "Drug",
    "MolecularInteraction",
    "CancerHypothesisGenerator",
    "Hypothesis",
    "HypothesisMetrics",
    "CausalChain",
    "TCLQuantumIntegrator",
    "QuantumProteinAnalysis",
    "TCLCausalChain",
    "DNAGenerator",
    "DNASequence",
    "run_pipeline"
]
