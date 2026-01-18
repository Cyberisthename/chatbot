"""
Genome Module - Executable Genome Framework (EGF)
=================================================

A novel computational paradigm representing genomes as executable, modular,
memory-preserving programs rather than static data or retrained predictive models.

Core Principle: DNA is not merely stored or predicted upon.
The genome itself becomes an executable system that:
  - Runs under context
  - Maintains persistent regulatory state  
  - Learns without catastrophic forgetting
  - Improves through replayable biological "experiences"

This is NOT:
  - A neural network trained end-to-end
  - A static simulator
  - AlphaFold-style prediction

This IS:
  - Genome → Program → Execution → Memory → Knowledge Accumulation
  - A new way of computing biological systems
  - Where regulation behaves as a stateful executable process

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    EXECUTABLE GENOME FRAMEWORK                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
  │  │    CONTEXT   │  │   GENOME     │  │      REGULOME        │   │
  │  │  ADAPTER     │→ │  CORE        │→ │      ADAPTER         │   │
  │  │ (Inputs)     │  │  (DNA)       │  │  (Regulatory Graph)  │   │
  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │
  │         │                                    │                   │
  │         ↓                                    ↓                   │
  │  ┌──────────────┐              ┌──────────────────────┐         │
  │  │ EPIGENETIC   │←────────────→│   EXPRESSION         │         │
  │  │ GATE ADAPTER │              │   DYNAMICS ADAPTER   │         │
  │  │ (Stateful)   │              │   (Temporal Exec)    │         │
  │  └──────────────┘              └──────────────────────┘         │
  │         │                                    │                   │
  │         ↓                                    ↓                   │
  │  ┌─────────────────────────────────────────────────────────┐    │
  │  │              ARTIFACT MEMORY SYSTEM                      │    │
  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │    │
  │  │  │  Context   │ │  Gate      │ │  Regulatory Paths  │   │    │
  │  │  │  States    │ │  States    │ │  & Expression      │   │    │
  │  │  └────────────┘ └────────────┘ └────────────────────┘   │    │
  │  └─────────────────────────────────────────────────────────┘    │
  │         │                                    │                   │
  │         ↓                                    ↓                   │
  │  ┌──────────────┐              ┌──────────────────────┐         │
  │  │   PHENOTYPE  │              │      PROTEOME        │         │
  │  │   ADAPTER    │              │      ADAPTER         │         │
  │  │  (Outcomes)  │              │  (Translation)       │         │
  │  └──────────────┘              └──────────────────────┘         │
  └─────────────────────────────────────────────────────────────────┘

Key Differentiators:
  1. Executable genome logic - DNA as source code
  2. Persistent epigenetic state - stateful gates
  3. Modular replayable memory - artifact system
  4. Context-dependent execution - environment-aware
  5. No catastrophic forgetting - memory accumulates
  6. No single monolithic model - modular adapters

The system NEVER forgets. Learning occurs by:
  - Storing successful regulatory executions as artifacts
  - Reusing prior biological experiences
  - Expanding memory rather than overwriting

This enables cumulative biological understanding, not statistical drift.

Usage:
    from src.genome import ExecutableGenomeFramework
    
    # Initialize framework
    egf = ExecutableGenomeFramework("/path/to/storage")
    
    # Load genome data
    egf.load_genome_data(genome_data)
    
    # Set biological context
    egf.set_context(tissue="liver", stress=0.3)
    
    # Execute genome
    result = egf.execute_genome(duration=24.0, time_step=1.0)
    
    # Access results
    print(result.phenotype_scores)
"""

from .executable_genome_framework import (
    # Core types
    GenomeRegion,
    RegulatoryElement,
    EpigeneticGate,
    ContextState,
    ExpressionTrajectory,
    PhenotypeScore,
    ExecutionArtifact,
    ExecutionStatus,
    
    # Adapter interfaces
    GenomeAdapter,
    RegulomeAdapter,
    EpigeneticGateAdapter,
    ContextAdapter,
    ExpressionDynamicsAdapter,
    ProteomeAdapter,
    PhenotypeAdapter,
    ArtifactMemoryAdapter,
    
    # Concrete implementations
    GenomeCoreAdapter,
    RegulomeGraphAdapter,
    EpigeneticGateManager,
    ContextEnvironmentAdapter,
    ExpressionDynamicsEngine,
    ProteomeTranslator,
    PhenotypeScorer,
    ArtifactMemorySystem,
    
    # Main framework
    ExecutableGenomeFramework,
)

__all__ = [
    # Core types
    "GenomeRegion",
    "RegulatoryElement",
    "EpigeneticGate",
    "ContextState",
    "ExpressionTrajectory",
    "PhenotypeScore",
    "ExecutionArtifact",
    "ExecutionStatus",
    
    # Adapter interfaces
    "GenomeAdapter",
    "RegulomeAdapter",
    "EpigeneticGateAdapter",
    "ContextAdapter",
    "ExpressionDynamicsAdapter",
    "ProteomeAdapter",
    "PhenotypeAdapter",
    "ArtifactMemoryAdapter",
    
    # Concrete implementations
    "GenomeCoreAdapter",
    "RegulomeGraphAdapter",
    "EpigeneticGateManager",
    "ContextEnvironmentAdapter",
    "ExpressionDynamicsEngine",
    "ProteomeTranslator",
    "PhenotypeScorer",
    "ArtifactMemorySystem",
    
    # Main framework
    "ExecutableGenomeFramework",
]
