"""
Biological Knowledge Integration for Cancer Hypothesis Generation

This module integrates real biological data with Thought-Compression Language (TCL)
and Quantum H-bond protein folding to generate novel cancer treatment hypotheses.

NEW: Virtual Cancer Cell Simulator - Test treatments digitally!

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

from .dna_sequence_retriever import (
    DNASequenceRetriever,
    GeneStructure,
    GenomicRegion
)

from .quantum_dna_optimizer import (
    QuantumDNAOptimizer,
    OptimizedDNA,
    DNAQuantumAnalysis
)

from .virtual_cancer_cell_simulator import (
    VirtualCancerCellSimulator,
    VirtualCellState,
    TreatmentOutcome,
    CellState
)

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
    "DNASequenceRetriever",
    "GeneStructure",
    "GenomicRegion",
    "QuantumDNAOptimizer",
    "OptimizedDNA",
    "DNAQuantumAnalysis",
    "VirtualCancerCellSimulator",
    "VirtualCellState",
    "TreatmentOutcome",
    "CellState"
]
