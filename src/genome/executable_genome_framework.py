"""
Executable Genome Framework (EGF)
=================================

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
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable


class ExecutionStatus(Enum):
    """Status of a biological execution episode."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    STABLE = "stable"  # Expression reached stable state
    UNSTABLE = "unstable"  # Expression diverged
    FAILED = "failed"


@dataclass
class GenomeRegion:
    """Represents a DNA region in the genome core."""
    region_id: str
    sequence: str  # DNA sequence (A, T, G, C)
    region_type: str  # exon, intron, promoter, enhancer, utr, etc.
    start: int
    end: int
    chromosome: str
    strand: str = "+"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "sequence": self.sequence,
            "region_type": self.region_type,
            "start": self.start,
            "end": self.end,
            "chromosome": self.chromosome,
            "strand": self.strand,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenomeRegion":
        return cls(
            region_id=data["region_id"],
            sequence=data["sequence"],
            region_type=data["region_type"],
            start=data["start"],
            end=data["end"],
            chromosome=data["chromosome"],
            strand=data.get("strand", "+"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegulatoryElement:
    """A regulatory element in the regulome graph."""
    element_id: str
    element_type: str  # promoter, enhancer, silencer, insulator, tf_binding_site
    target_genes: List[str]  # Genes regulated by this element
    tf_families: List[str]  # Transcription factor families that bind here
    genomic_location: Tuple[str, int, int]  # (chromosome, start, end)
    weight: float = 1.0  # Default regulatory strength
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "target_genes": self.target_genes,
            "tf_families": self.tf_families,
            "genomic_location": self.genomic_location,
            "weight": self.weight,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegulatoryElement":
        return cls(
            element_id=data["element_id"],
            element_type=data["element_type"],
            target_genes=data["target_genes"],
            tf_families=data.get("tf_families", []),
            genomic_location=tuple(data["genomic_location"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpigeneticGate:
    """Stateful gate mimicking methylation/chromatin accessibility.
    
    Unlike static weights, these gates have:
    - PERSISTENT state (like methylation patterns)
    - CONTEXT-dependent activation
    - HISTORY-awareness (prior states influence current)
    """
    gate_id: str
    regulated_element_id: str
    gate_type: str  # "methylation", "accessibility", "histone_mod"
    
    # Persistent state - this is the key innovation
    methylation_level: float = 0.0  # 0 = unmethylated (active), 1 = methylated (silent)
    accessibility_score: float = 1.0  # 0 = closed, 1 = open chromatin
    histone_marks: Dict[str, float] = field(default_factory=dict)  # H3K4me3, H3K27ac, etc.
    
    # History for stateful behavior
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context sensitivity
    tissue_specificity: Dict[str, float] = field(default_factory=dict)  # TF binding tissue preference
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "regulated_element_id": self.regulated_element_id,
            "gate_type": self.gate_type,
            "methylation_level": self.methylation_level,
            "accessibility_score": self.accessibility_score,
            "histone_marks": self.histone_marks,
            "state_history": self.state_history,
            "tissue_specificity": self.tissue_specificity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpigeneticGate":
        return cls(
            gate_id=data["gate_id"],
            regulated_element_id=data["regulated_element_id"],
            gate_type=data["gate_type"],
            methylation_level=data.get("methylation_level", 0.0),
            accessibility_score=data.get("accessibility_score", 1.0),
            histone_marks=data.get("histone_marks", {}),
            state_history=data.get("state_history", []),
            tissue_specificity=data.get("tissue_specificity", {}),
        )
    
    def apply_context(self, context: Dict[str, Any], time_step: int) -> float:
        """Apply context to gate and return activation level.
        
        This is where context-dependent, stateful behavior emerges.
        """
        tissue = context.get("tissue", "generic")
        stress_level = context.get("stress", 0.0)
        signal_input = context.get("signals", {})
        
        # Tissue-specific activation
        tissue_factor = self.tissue_specificity.get(tissue, 0.5)
        
        # Stress response
        stress_response = 1.0 - min(stress_level, 1.0)  # High stress → lower activation
        
        # Signal-dependent modulation
        signal_modulation = 1.0
        for signal, strength in signal_input.items():
            signal_modulation *= (1.0 + strength * 0.1)
        
        # Calculate new accessibility based on context
        new_accessibility = (
            tissue_factor * 0.4 +
            stress_response * 0.3 +
            min(signal_modulation, 1.5) * 0.3
        )
        
        # Methylation can change slowly based on context (hysteresis)
        if context.get("inducer", None):
            # Some treatments can demethylate
            self.methylation_level = max(0.0, self.methylation_level - 0.1)
        elif context.get("repressor", None):
            # Some treatments can methylate
            self.methylation_level = min(1.0, self.methylation_level + 0.1)
        
        # State persistence: gradual return to baseline
        baseline = 0.5
        persistence_rate = 0.1
        self.accessibility_score = (
            self.accessibility_score * (1 - persistence_rate) +
            new_accessibility * persistence_rate
        )
        
        # Record state history for replay
        self.state_history.append({
            "time_step": time_step,
            "context": {"tissue": tissue, "stress": stress_level},
            "accessibility": self.accessibility_score,
            "methylation": self.methylation_level,
        })
        
        # Return effective activation (1 = fully active, 0 = fully silent)
        return (1.0 - self.methylation_level) * self.accessibility_score


@dataclass
class ContextState:
    """Biological context/input state."""
    tissue: str = "generic"
    developmental_stage: str = "adult"
    stress_level: float = 0.0
    nutrient_status: str = "normal"
    signal_molecules: Dict[str, float] = field(default_factory=dict)
    environmental_conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tissue": self.tissue,
            "developmental_stage": self.developmental_stage,
            "stress_level": self.stress_level,
            "nutrient_status": self.nutrient_status,
            "signal_molecules": self.signal_molecules,
            "environmental_conditions": self.environmental_conditions,
            "timestamp": self.timestamp,
        }


@dataclass
class ExpressionTrajectory:
    """Gene expression over time."""
    gene_id: str
    time_points: List[float]
    expression_values: List[float]
    trajectory_type: str = "continuous"  # "continuous", "pulsatile", "oscillatory"
    stability_score: float = 0.0
    peak_expression: float = 0.0
    mean_expression: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "time_points": self.time_points,
            "expression_values": self.expression_values,
            "trajectory_type": self.trajectory_type,
            "stability_score": self.stability_score,
            "peak_expression": self.peak_expression,
            "mean_expression": self.mean_expression,
        }
    
    def compute_statistics(self):
        """Compute summary statistics for this trajectory."""
        if not self.expression_values:
            return
        self.peak_expression = max(self.expression_values)
        self.mean_expression = sum(self.expression_values) / len(self.expression_values)
        # Stability: low variance = high stability
        if len(self.expression_values) > 1:
            variance = sum((x - self.mean_expression) ** 2 for x in self.expression_values) / len(self.expression_values)
            self.stability_score = 1.0 / (1.0 + variance)
        else:
            self.stability_score = 1.0


@dataclass
class PhenotypeScore:
    """Biological outcome/phenotype scores."""
    viability_score: float = 0.0
    stability_score: float = 0.0
    efficiency_score: float = 0.0
    fitness_proxy: float = 0.0
    pathway_activity: Dict[str, float] = field(default_factory=dict)
    cellular_response: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "viability_score": self.viability_score,
            "stability_score": self.stability_score,
            "efficiency_score": self.efficiency_score,
            "fitness_proxy": self.fitness_proxy,
            "pathway_activity": self.pathway_activity,
            "cellular_response": self.cellular_response,
        }


@dataclass
class ExecutionArtifact:
    """Complete record of a biological execution episode.
    
    This is the key to non-destructive learning. Unlike neural networks
    that overwrite weights, we store entire execution episodes.
    """
    artifact_id: str
    execution_id: str
    context: Dict[str, Any]
    gate_states: Dict[str, Dict[str, Any]]
    regulatory_paths: List[List[str]]  # Paths through the regulatory graph
    expression_trajectories: Dict[str, Dict[str, Any]]  # gene_id -> trajectory
    phenotype_scores: Dict[str, Any]
    outcome_score: float  # Overall success measure
    execution_time: float
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "execution_id": self.execution_id,
            "context": self.context,
            "gate_states": self.gate_states,
            "regulatory_paths": self.regulatory_paths,
            "expression_trajectories": self.expression_trajectories,
            "phenotype_scores": self.phenotype_scores,
            "outcome_score": self.outcome_score,
            "execution_time": self.execution_time,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionArtifact":
        return cls(
            artifact_id=data["artifact_id"],
            execution_id=data["execution_id"],
            context=data["context"],
            gate_states=data["gate_states"],
            regulatory_paths=data["regulatory_paths"],
            expression_trajectories=data["expression_trajectories"],
            phenotype_scores=data["phenotype_scores"],
            outcome_score=data["outcome_score"],
            execution_time=data["execution_time"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# ADAPTER BASE CLASSES
# ============================================================================

class GenomeAdapter(ABC):
    """Abstract base for Genome Core Adapter.
    
    Stores DNA sequences, variants, isoforms as immutable biological source code.
    """
    
    @abstractmethod
    def load_genome(self, genome_data: Dict[str, Any]) -> None:
        """Load genome data into the adapter."""
        pass
    
    @abstractmethod
    def get_sequence(self, region_id: str) -> Optional[str]:
        """Retrieve DNA sequence for a region."""
        pass
    
    @abstractmethod
    def get_gene(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get gene structure and metadata."""
        pass
    
    @abstractmethod
    def get_isoforms(self, gene_id: str) -> List[Dict[str, Any]]:
        """Get all isoforms for a gene."""
        pass


class RegulomeAdapter(ABC):
    """Abstract base for Regulome Adapter.
    
    Represents promoters, enhancers, silencers, transcription factors
    as an executable graph where edges represent regulatory influence.
    """
    
    @abstractmethod
    def add_regulatory_element(self, element: RegulatoryElement) -> None:
        """Add a regulatory element to the graph."""
        pass
    
    @abstractmethod
    def add_regulatory_edge(self, source_id: str, target_id: str, weight: float) -> None:
        """Add regulatory influence edge."""
        pass
    
    @abstractmethod
    def get_regulatory_influence(self, element_id: str, context: Dict[str, Any]) -> float:
        """Compute regulatory influence for an element under context."""
        pass
    
    @abstractmethod
    def get_regulatory_pathways(self, start_element: str, target_gene: str) -> List[List[str]]:
        """Find all regulatory paths from element to gene."""
        pass


class EpigeneticGateAdapter(ABC):
    """Abstract base for Epigenetic Gate Adapter.
    
    Stateful gates controlling regulatory edges with context-dependent,
    persistent, history-aware behavior mimicking methylation/accessibility.
    """
    
    @abstractmethod
    def create_gate(self, element_id: str, gate_type: str) -> EpigeneticGate:
        """Create a new epigenetic gate for an element."""
        pass
    
    @abstractmethod
    def apply_context_to_gates(self, context: Dict[str, Any], time_step: int) -> Dict[str, float]:
        """Apply context to all gates and return activation levels."""
        pass
    
    @abstractmethod
    def get_gate_state(self, gate_id: str) -> Optional[EpigeneticGate]:
        """Get current state of a gate."""
        pass


class ContextAdapter(ABC):
    """Abstract base for Context/Environment Adapter.
    
    Handles tissue identity, stress, signals, nutrients, conditions.
    """
    
    @abstractmethod
    def set_context(self, context: ContextState) -> None:
        """Set the current biological context."""
        pass
    
    @abstractmethod
    def get_context(self) -> ContextState:
        """Get current context."""
        pass
    
    @abstractmethod
    def activate_transcription_factors(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Activate transcription factors based on context."""
        pass


class ExpressionDynamicsAdapter(ABC):
    """Abstract base for Expression Dynamics Adapter.
    
    Executes the regulatory graph over time, produces continuous
    gene expression trajectories. Learns which paths are stable/unstable.
    """
    
    @abstractmethod
    def execute_expression(self, regulatory_input: Dict[str, float], 
                          duration: float, time_step: float) -> Dict[str, ExpressionTrajectory]:
        """Execute expression dynamics over time."""
        pass
    
    @abstractmethod
    def get_stable_states(self) -> List[Dict[str, Any]]:
        """Get identified stable expression states."""
        pass


class ProteomeAdapter(ABC):
    """Abstract base for Proteome Adapter.
    
    Translates expression into protein abundance and functional embeddings.
    """
    
    @abstractmethod
    def translate_expression(self, expression_trajectories: Dict[str, ExpressionTrajectory],
                           genome_adapter: GenomeAdapter) -> Dict[str, Any]:
        """Translate mRNA expression to protein abundance."""
        pass
    
    @abstractmethod
    def get_protein_embeddings(self, protein_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate functional embeddings for proteins."""
        pass


class PhenotypeAdapter(ABC):
    """Abstract base for Outcome/Phenotype Adapter.
    
    Scores biological outcomes, defines experiment success criteria.
    """
    
    @abstractmethod
    def score_phenotype(self, expression_data: Dict[str, ExpressionTrajectory],
                       protein_data: Dict[str, Any]) -> PhenotypeScore:
        """Score the biological outcome."""
        pass
    
    @abstractmethod
    def is_successful(self, score: PhenotypeScore) -> bool:
        """Determine if experiment was successful."""
        pass


class ArtifactMemoryAdapter(ABC):
    """Abstract base for Artifact Memory System.
    
    Stores complete biological execution episodes for replay and reuse.
    """
    
    @abstractmethod
    def store_artifact(self, artifact: ExecutionArtifact) -> None:
        """Store an execution artifact."""
        pass
    
    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[ExecutionArtifact]:
        """Retrieve an artifact by ID."""
        pass
    
    @abstractmethod
    def find_similar_artifacts(self, context: Dict[str, Any], 
                               min_score: float) -> List[ExecutionArtifact]:
        """Find artifacts with similar contexts/outcomes."""
        pass
    
    @abstractmethod
    def replay_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Replay an artifact to regenerate its execution state."""
        pass


# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class GenomeCoreAdapter(GenomeAdapter):
    """Concrete Genome Core Adapter implementation.
    
    Stores DNA as immutable biological source code.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.regions: Dict[str, GenomeRegion] = {}
        self.genes: Dict[str, Dict[str, Any]] = {}
        self.isoforms: Dict[str, List[Dict[str, Any]]] = {}
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load persisted genome data."""
        genome_file = self.storage_path / "genome.json"
        if genome_file.exists():
            with open(genome_file, "r") as f:
                data = json.load(f)
                for region_data in data.get("regions", []):
                    region = GenomeRegion.from_dict(region_data)
                    self.regions[region.region_id] = region
                self.genes = data.get("genes", {})
                self.isoforms = data.get("isoforms", {})
    
    def _save_to_disk(self):
        """Persist genome data."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        genome_file = self.storage_path / "genome.json"
        data = {
            "regions": [r.to_dict() for r in self.regions.values()],
            "genes": self.genes,
            "isoforms": self.isoforms,
        }
        with open(genome_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_genome(self, genome_data: Dict[str, Any]) -> None:
        """Load complete genome data."""
        # Load regions
        for region_data in genome_data.get("regions", []):
            region = GenomeRegion.from_dict(region_data)
            self.regions[region.region_id] = region
        
        # Load genes
        self.genes.update(genome_data.get("genes", {}))
        
        # Load isoforms
        for gene_id, isoforms_data in genome_data.get("isoforms", {}).items():
            if gene_id not in self.isoforms:
                self.isoforms[gene_id] = []
            self.isoforms[gene_id].extend(isoforms_data)
        
        self._save_to_disk()
    
    def add_gene(self, gene_id: str, gene_data: Dict[str, Any]) -> None:
        """Add a gene to the genome."""
        self.genes[gene_id] = gene_data
        self._save_to_disk()
    
    def add_region(self, region: GenomeRegion) -> None:
        """Add a genomic region."""
        self.regions[region.region_id] = region
        self._save_to_disk()
    
    def add_isoform(self, gene_id: str, isoform_data: Dict[str, Any]) -> None:
        """Add an isoform for a gene."""
        if gene_id not in self.isoforms:
            self.isoforms[gene_id] = []
        self.isoforms[gene_id].append(isoform_data)
        self._save_to_disk()
    
    def get_sequence(self, region_id: str) -> Optional[str]:
        """Retrieve DNA sequence for a region."""
        if region_id in self.regions:
            return self.regions[region_id].sequence
        return None
    
    def get_gene(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get gene structure and metadata."""
        return self.genes.get(gene_id)
    
    def get_isoforms(self, gene_id: str) -> List[Dict[str, Any]]:
        """Get all isoforms for a gene."""
        return self.isoforms.get(gene_id, [])
    
    def get_sequences_for_genes(self, gene_ids: List[str]) -> Dict[str, str]:
        """Get exonic sequences for multiple genes."""
        sequences = {}
        for gene_id in gene_ids:
            gene = self.get_gene(gene_id)
            if gene:
                # Collect exonic sequences
                exonic_regions = gene.get("exonic_regions", [])
                full_seq = ""
                for region_id in exonic_regions:
                    seq = self.get_sequence(region_id)
                    if seq:
                        full_seq += seq
                sequences[gene_id] = full_seq
        return sequences


class RegulomeGraphAdapter(RegulomeAdapter):
    """Concrete Regulome Adapter with executable graph.
    
    Edges represent regulatory influence, NOT static equations.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.elements: Dict[str, RegulatoryElement] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}  # source -> [(target, weight), ...]
        self.reverse_edges: Dict[str, List[Tuple[str, float]]] = {}  # target <- [(source, weight), ...]
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load persisted regulome data."""
        regulome_file = self.storage_path / "regulome.json"
        if regulome_file.exists():
            with open(regulome_file, "r") as f:
                data = json.load(f)
                for elem_data in data.get("elements", []):
                    element = RegulatoryElement.from_dict(elem_data)
                    self.elements[element.element_id] = element
                self.edges = data.get("edges", {})
                self.reverse_edges = data.get("reverse_edges", {})
    
    def _save_to_disk(self):
        """Persist regulome data."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        regulome_file = self.storage_path / "regulome.json"
        data = {
            "elements": [e.to_dict() for e in self.elements.values()],
            "edges": self.edges,
            "reverse_edges": self.reverse_edges,
        }
        with open(regulome_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_regulatory_element(self, element: RegulatoryElement) -> None:
        """Add a regulatory element to the graph."""
        self.elements[element.element_id] = element
        self._save_to_disk()
    
    def add_regulatory_edge(self, source_id: str, target_id: str, weight: float) -> None:
        """Add regulatory influence edge."""
        if source_id not in self.edges:
            self.edges[source_id] = []
        self.edges[source_id].append((target_id, weight))
        
        if target_id not in self.reverse_edges:
            self.reverse_edges[target_id] = []
        self.reverse_edges[target_id].append((source_id, weight))
        
        self._save_to_disk()
    
    def get_regulatory_influence(self, element_id: str, context: Dict[str, Any]) -> float:
        """Compute regulatory influence for an element under context.
        
        Unlike static equations, this is EXECUTABLE logic:
        - Sum of all incoming regulatory influences
        - Weighted by TF activity from context
        - Modified by element-specific properties
        """
        if element_id not in self.elements:
            return 0.0
        
        element = self.elements[element_id]
        total_influence = 0.0
        
        # Sum influences from all regulatory sources
        if element_id in self.reverse_edges:
            for source_id, base_weight in self.reverse_edges[element_id]:
                source_element = self.elements.get(source_id)
                if source_element:
                    # Get TF activity from context
                    tf_activity = context.get("tf_activity", {})
                    tf_factor = 1.0
                    
                    # Check if any TFs in source element match context
                    for tf_family in source_element.tf_families:
                        if tf_family in tf_activity:
                            tf_factor *= (1.0 + tf_activity[tf_family])
                    
                    total_influence += base_weight * tf_factor * element.weight
        
        # Element's own regulatory capacity
        element_factor = element.weight
        
        return total_influence * element_factor
    
    def get_regulatory_pathways(self, start_element: str, target_gene: str) -> List[List[str]]:
        """Find all regulatory paths from element to gene.
        
        This discovers which regulatory cascades are possible.
        """
        if start_element not in self.elements:
            return []
        
        pathways = []
        visited = set()
        
        def dfs(current: str, path: List[str]):
            if current == target_gene:
                pathways.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            
            if current in self.edges:
                for target, weight in self.edges[current]:
                    if weight > 0.1:  # Only follow strong regulatory links
                        path.append(target)
                        dfs(target, path)
                        path.pop()
        
        dfs(start_element, [start_element])
        return pathways
    
    def get_upstream_regulators(self, element_id: str, max_depth: int = 3) -> List[str]:
        """Get all regulators upstream of an element."""
        regulators = []
        visited = set()
        
        def traverse(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            
            if current in self.reverse_edges:
                for source, _ in self.reverse_edges[current]:
                    if source not in visited:
                        regulators.append(source)
                        traverse(source, depth + 1)
        
        traverse(element_id, 0)
        return regulators
    
    def get_downstream_targets(self, element_id: str, max_depth: int = 3) -> List[str]:
        """Get all targets downstream of an element."""
        targets = []
        visited = set()
        
        def traverse(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            
            if current in self.edges:
                for target, _ in self.edges[current]:
                    if target not in visited:
                        targets.append(target)
                        traverse(target, depth + 1)
        
        traverse(element_id, 0)
        return targets


class EpigeneticGateManager(EpigeneticGateAdapter):
    """Concrete Epigenetic Gate Adapter.
    
    Stateful gates with persistent, context-dependent, history-aware behavior.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.gates: Dict[str, EpigeneticGate] = {}
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load persisted gate states."""
        gates_file = self.storage_path / "gates.json"
        if gates_file.exists():
            with open(gates_file, "r") as f:
                data = json.load(f)
                for gate_data in data.get("gates", []):
                    gate = EpigeneticGate.from_dict(gate_data)
                    self.gates[gate.gate_id] = gate
    
    def _save_to_disk(self):
        """Persist gate states."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        gates_file = self.storage_path / "gates.json"
        data = {
            "gates": [g.to_dict() for g in self.gates.values()],
        }
        with open(gates_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_gate(self, element_id: str, gate_type: str) -> EpigeneticGate:
        """Create a new epigenetic gate for an element."""
        gate_id = f"gate_{uuid.uuid4().hex[:8]}"
        gate = EpigeneticGate(
            gate_id=gate_id,
            regulated_element_id=element_id,
            gate_type=gate_type,
        )
        self.gates[gate_id] = gate
        self._save_to_disk()
        return gate
    
    def apply_context_to_gates(self, context: Dict[str, Any], time_step: int) -> Dict[str, float]:
        """Apply context to all gates and return activation levels."""
        activations = {}
        for gate_id, gate in self.gates.items():
            activation = gate.apply_context(context, time_step)
            activations[gate_id] = activation
        self._save_to_disk()
        return activations
    
    def get_gate_state(self, gate_id: str) -> Optional[EpigeneticGate]:
        """Get current state of a gate."""
        return self.gates.get(gate_id)
    
    def get_element_gate(self, element_id: str) -> Optional[EpigeneticGate]:
        """Get gate for a specific element."""
        for gate in self.gates.values():
            if gate.regulated_element_id == element_id:
                return gate
        return None
    
    def set_tissue_specificity(self, gate_id: str, tissue: str, specificity: float) -> None:
        """Set tissue-specific activation for a gate."""
        if gate_id in self.gates:
            self.gates[gate_id].tissue_specificity[tissue] = specificity
            self._save_to_disk()
    
    def reset_gate_state(self, gate_id: str) -> None:
        """Reset a gate to default state."""
        if gate_id in self.gates:
            self.gates[gate_id].methylation_level = 0.0
            self.gates[gate_id].accessibility_score = 0.5
            self.gates[gate_id].state_history = []
            self._save_to_disk()


class ContextEnvironmentAdapter(ContextAdapter):
    """Concrete Context/Environment Adapter.
    
    Handles tissue identity, stress, signals, nutrients, conditions.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.current_context = ContextState()
        self.tf_database: Dict[str, Dict[str, Any]] = {}  # TF family -> properties
        self.context_history: List[Dict[str, Any]] = []
        self._load_tf_database()
    
    def _load_tf_database(self):
        """Load transcription factor database."""
        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        tf_file = self.storage_path / "tf_database.json"
        if tf_file.exists():
            with open(tf_file, "r") as f:
                self.tf_database = json.load(f)
        else:
            # Initialize with common TF families
            self.tf_database = {
                "homeobox": {
                    "targets": ["HOX genes", "developmental genes"],
                    "tissue_specificity": {"embryonic": 0.9, "adult": 0.3},
                },
                "bHLH": {
                    "targets": ["MYC", "MAX", "cell_cycle_genes"],
                    "tissue_specificity": {"proliferating": 0.9, "quiescent": 0.2},
                },
                "nuclear_receptor": {
                    "targets": ["metabolic_genes", "differentiation_genes"],
                    "tissue_specificity": {"liver": 0.8, "adipose": 0.7},
                },
                "zinc_finger": {
                    "targets": ["KRAB", "C2H2_genes"],
                    "tissue_specificity": {"generic": 0.5},
                },
                "AP1": {
                    "targets": ["stress_response", "proliferation"],
                    "tissue_specificity": {"stressed": 0.9, "normal": 0.3},
                },
                "p53": {
                    "targets": ["DNA_repair", "apoptosis", "cell_cycle_arrest"],
                    "tissue_specificity": {"stressed": 0.95, "normal": 0.1},
                },
                "NFkB": {
                    "targets": ["inflammation", "immune_response"],
                    "tissue_specificity": {"immune": 0.9, "other": 0.2},
                },
                "STAT": {
                    "targets": ["cytokine_response", "immune_genes"],
                    "tissue_specificity": {"immune": 0.85, "stressed": 0.6},
                },
            }
            with open(tf_file, "w") as f:
                json.dump(self.tf_database, f, indent=2)
    
    def set_context(self, context: ContextState) -> None:
        """Set the current biological context."""
        self.current_context = context
        self.context_history.append({
            "timestamp": time.time(),
            "context": context.to_dict(),
        })
        # Keep only last 100 contexts
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
    
    def get_context(self) -> ContextState:
        """Get current context."""
        return self.current_context
    
    def activate_transcription_factors(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Activate transcription factors based on context.
        
        This is where context → TF activity mapping happens.
        """
        tf_activity = {}
        
        tissue = context.get("tissue", "generic")
        stress = context.get("stress", 0.0)
        signals = context.get("signals", {})
        developmental_stage = context.get("developmental_stage", "adult")
        
        for tf_family, properties in self.tf_database.items():
            activity = 0.0
            
            # Tissue-specific activation
            tissue_spec = properties.get("tissue_specificity", {})
            if tissue in tissue_spec:
                activity += tissue_spec[tissue] * 0.4
            else:
                activity += tissue_spec.get("generic", 0.3) * 0.4
            
            # Stress response
            if tf_family in ["AP1", "p53", "NFkB"]:
                activity += stress * 0.5  # Stress strongly activates these
            
            # Signal-dependent activation
            for signal, strength in signals.items():
                if signal in properties.get("targets", []):
                    activity += strength * 0.3
            
            # Developmental regulation
            if developmental_stage == "embryonic":
                if tf_family == "homeobox":
                    activity += 0.3
            elif developmental_stage == "adult":
                if tf_family in ["nuclear_receptor", "PPAR"]:
                    activity += 0.2
            
            # Normalize to [0, 1]
            tf_activity[tf_family] = min(1.0, max(0.0, activity))
        
        return tf_activity
    
    def add_tf_family(self, tf_family: str, properties: Dict[str, Any]) -> None:
        """Add a new TF family to the database."""
        self.tf_database[tf_family] = properties
    
    def get_available_tf_families(self) -> List[str]:
        """Get list of all TF families."""
        return list(self.tf_database.keys())


class ExpressionDynamicsEngine(ExpressionDynamicsAdapter):
    """Concrete Expression Dynamics Adapter.
    
    Executes the regulatory graph over time, produces continuous
    gene expression trajectories. Identifies stable/unstable states.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.stable_states: List[Dict[str, Any]] = []
        self.expression_history: List[Dict[str, Any]] = []
        self._load_stable_states()
    
    def _load_stable_states(self):
        """Load previously identified stable states."""
        states_file = self.storage_path / "stable_states.json"
        if states_file.exists():
            with open(states_file, "r") as f:
                self.stable_states = json.load(f)
    
    def _save_stable_states(self):
        """Persist stable states."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        states_file = self.storage_path / "stable_states.json"
        with open(states_file, "w") as f:
            json.dump(self.stable_states, f, indent=2)
    
    def execute_expression(self, regulatory_input: Dict[str, float],
                          duration: float, time_step: float) -> Dict[str, ExpressionTrajectory]:
        """Execute expression dynamics over time.
        
        For each gene, we simulate:
        - Transcription rate from regulatory inputs
        - mRNA degradation
        - Translation (to protein, handled by ProteomeAdapter)
        - Feedback loops
        """
        trajectories = {}
        n_steps = int(duration / time_step)
        time_points = [i * time_step for i in range(n_steps + 1)]
        
        # Initialize expression levels
        expression = {gene: 0.0 for gene in regulatory_input.keys()}
        
        # Decay rate for mRNA (typical half-life ~10 hours → decay ~0.07/hr)
        decay_rate = 0.07
        
        for step in range(n_steps + 1):
            # Record current state
            for gene, level in expression.items():
                if gene not in trajectories:
                    trajectories[gene] = ExpressionTrajectory(
                        gene_id=gene,
                        time_points=[],
                        expression_values=[],
                    )
                trajectories[gene].time_points.append(time_points[step])
                trajectories[gene].expression_values.append(level)
            
            # Update expression for next step
            next_expression = {}
            for gene, current_level in expression.items():
                regulatory_signal = regulatory_input.get(gene, 0.0)
                
                # Gene expression ODE: dE/dt = production - decay
                # Production depends on regulatory input
                production_rate = regulatory_signal * 10.0  # Max expression level
                
                # Update equation
                delta = (production_rate - decay_rate * current_level) * time_step
                next_expression[gene] = max(0.0, current_level + delta)
            
            expression = next_expression
        
        # Compute statistics for each trajectory
        for gene, traj in trajectories.items():
            traj.compute_statistics()
        
        # Identify stable states
        self._identify_stable_states(trajectories)
        
        return trajectories
    
    def _identify_stable_states(self, trajectories: Dict[str, ExpressionTrajectory]):
        """Identify stable expression states from trajectories."""
        for gene, traj in trajectories.items():
            if traj.stability_score > 0.8:  # Very stable
                state = {
                    "gene_id": gene,
                    "expression_level": traj.mean_expression,
                    "stability": traj.stability_score,
                    "trajectory_type": traj.trajectory_type,
                    "peak": traj.peak_expression,
                }
                # Check if similar state already exists
                is_new = True
                for existing in self.stable_states:
                    if (existing["gene_id"] == gene and 
                        abs(existing["expression_level"] - state["expression_level"]) < 0.1):
                        is_new = False
                        break
                if is_new:
                    self.stable_states.append(state)
        
        self._save_stable_states()
    
    def get_stable_states(self) -> List[Dict[str, Any]]:
        """Get identified stable expression states."""
        return self.stable_states
    
    def get_unstable_genes(self, trajectories: Dict[str, ExpressionTrajectory], 
                          stability_threshold: float = 0.5) -> List[str]:
        """Get genes with unstable expression."""
        unstable = []
        for gene, traj in trajectories.items():
            if traj.stability_score < stability_threshold:
                unstable.append(gene)
        return unstable


class ProteomeTranslator(ProteomeAdapter):
    """Concrete Proteome Adapter.
    
    Translates expression into protein abundance and functional embeddings.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.translation_efficiency: Dict[str, float] = {}  # gene -> efficiency
        self.protein_half_lives: Dict[str, float] = {}  # protein -> half-life (hours)
        self._load_parameters()
    
    def _load_parameters(self):
        """Load translation parameters."""
        params_file = self.storage_path / "translation_params.json"
        if params_file.exists():
            with open(params_file, "r") as f:
                data = json.load(f)
                self.translation_efficiency = data.get("translation_efficiency", {})
                self.protein_half_lives = data.get("protein_half_lives", {})
    
    def _save_parameters(self):
        """Persist translation parameters."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        params_file = self.storage_path / "translation_params.json"
        data = {
            "translation_efficiency": self.translation_efficiency,
            "protein_half_lives": self.protein_half_lives,
        }
        with open(params_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def translate_expression(self, expression_trajectories: Dict[str, ExpressionTrajectory],
                           genome_adapter: GenomeAdapter) -> Dict[str, Any]:
        """Translate mRNA expression to protein abundance.
        
        Uses a simple translation model:
        - Translation efficiency varies by gene
        - Protein degradation follows exponential decay
        """
        protein_data = {}
        
        for gene_id, expr_traj in expression_trajectories.items():
            # Get translation efficiency
            efficiency = self.translation_efficiency.get(gene_id, 0.5)  # Default 50%
            
            # Translate each time point
            protein_time = []
            protein_values = []
            
            for i, (t, mrna_level) in enumerate(zip(expr_traj.time_points, 
                                                      expr_traj.expression_values)):
                protein_level = mrna_level * efficiency * 10.0  # Scale factor
                protein_time.append(t)
                protein_values.append(protein_level)
            
            protein_data[gene_id] = {
                "time_points": protein_time,
                "abundance": protein_values,
                "mean_abundance": sum(protein_values) / len(protein_values),
                "peak_abundance": max(protein_values),
                "translation_efficiency": efficiency,
            }
        
        return protein_data
    
    def get_protein_embeddings(self, protein_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate functional embeddings for proteins.
        
        Creates a simple embedding based on:
        - Abundance level
        - Temporal dynamics
        - Functional category (inferred from gene name)
        """
        embeddings = {}
        
        for gene_id, data in protein_data.items():
            embedding = []
            
            # Abundance features (normalized)
            mean_abund = data.get("mean_abundance", 0)
            embedding.append(min(1.0, mean_abund / 100.0))  # Normalized mean
            embedding.append(min(1.0, data.get("peak_abundance", 0) / 100.0))  # Peak
            
            # Dynamics features
            if len(data.get("abundance", [])) > 1:
                values = data["abundance"]
                variance = sum((x - mean_abund) ** 2 for x in values) / len(values)
                embedding.append(min(1.0, variance / 100.0))  # Variability
            else:
                embedding.append(0.0)
            
            # Pad to fixed size embedding
            while len(embedding) < 16:
                embedding.append(0.0)
            
            # Truncate if too long
            embedding = embedding[:16]
            
            embeddings[gene_id] = embedding
        
        return embeddings
    
    def set_translation_efficiency(self, gene_id: str, efficiency: float) -> None:
        """Set translation efficiency for a gene."""
        self.translation_efficiency[gene_id] = max(0.0, min(1.0, efficiency))
        self._save_parameters()


class PhenotypeScorer(PhenotypeAdapter):
    """Concrete Outcome/Phenotype Adapter.
    
    Scores biological outcomes, defines experiment success criteria.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.success_criteria: Dict[str, float] = {}
        self.score_history: List[Dict[str, Any]] = []
        self._load_criteria()
    
    def _load_criteria(self):
        """Load success criteria."""
        criteria_file = self.storage_path / "success_criteria.json"
        if criteria_file.exists():
            with open(criteria_file, "r") as f:
                self.success_criteria = json.load(f)
        else:
            # Default criteria
            self.success_criteria = {
                "min_viability": 0.5,
                "min_stability": 0.6,
                "min_efficiency": 0.4,
            }
            self._save_criteria()
    
    def _save_criteria(self):
        """Persist success criteria."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        criteria_file = self.storage_path / "success_criteria.json"
        with open(criteria_file, "w") as f:
            json.dump(self.success_criteria, f, indent=2)
    
    def score_phenotype(self, expression_data: Dict[str, ExpressionTrajectory],
                       protein_data: Dict[str, Any]) -> PhenotypeScore:
        """Score the biological outcome."""
        score = PhenotypeScore()
        
        # Calculate viability: based on overall expression levels
        # Too low = cells might not be functioning
        # Too high = might indicate dysregulation
        if expression_data:
            mean_expressions = [traj.mean_expression for traj in expression_data.values()]
            overall_mean = sum(mean_expressions) / len(mean_expressions)
            
            # Optimal expression range
            if 20.0 <= overall_mean <= 80.0:
                score.viability_score = 1.0
            elif 10.0 <= overall_mean <= 100.0:
                score.viability_score = 0.7
            else:
                score.viability_score = 0.4
        
        # Calculate stability: average of trajectory stabilities
        if expression_data:
            stabilities = [traj.stability_score for traj in expression_data.values()]
            score.stability_score = sum(stabilities) / len(stabilities)
        
        # Calculate efficiency: protein output per mRNA input
        if protein_data:
            efficiencies = []
            for gene_id, pdata in protein_data.items():
                if gene_id in expression_data:
                    traj = expression_data[gene_id]
                    if traj.mean_expression > 0:
                        efficiency = pdata["mean_abundance"] / (traj.mean_expression * 10)
                        efficiencies.append(min(1.0, efficiency))
            if efficiencies:
                score.efficiency_score = sum(efficiencies) / len(efficiencies)
        
        # Fitness proxy: weighted combination
        score.fitness_proxy = (
            score.viability_score * 0.4 +
            score.stability_score * 0.4 +
            score.efficiency_score * 0.2
        )
        
        # Pathway activity (simplified)
        for gene_id in expression_data.keys():
            if "ribosomal" in gene_id.lower():
                score.pathway_activity["translation"] = score.pathway_activity.get("translation", 0) + 0.1
            elif "metabolic" in gene_id.lower() or "oxido" in gene_id.lower():
                score.pathway_activity["metabolism"] = score.pathway_activity.get("metabolism", 0) + 0.1
            elif "stress" in gene_id.lower() or "hsp" in gene_id.lower():
                score.pathway_activity["stress_response"] = score.pathway_activity.get("stress_response", 0) + 0.1
            elif "cell_cycle" in gene_id.lower() or "cycline" in gene_id.lower():
                score.pathway_activity["proliferation"] = score.pathway_activity.get("proliferation", 0) + 0.1
        
        # Normalize pathway scores
        for pathway in score.pathway_activity:
            score.pathway_activity[pathway] = min(1.0, score.pathway_activity[pathway])
        
        return score
    
    def is_successful(self, score: PhenotypeScore) -> bool:
        """Determine if experiment was successful."""
        return (
            score.viability_score >= self.success_criteria.get("min_viability", 0.5) and
            score.stability_score >= self.success_criteria.get("min_stability", 0.6) and
            score.efficiency_score >= self.success_criteria.get("min_efficiency", 0.4)
        )
    
    def set_success_criteria(self, criteria: Dict[str, float]) -> None:
        """Update success criteria."""
        self.success_criteria.update(criteria)
        self._save_criteria()


class ArtifactMemorySystem(ArtifactMemoryAdapter):
    """Concrete Artifact Memory System.
    
    Stores complete biological execution episodes for replay and reuse.
    
    KEY INNOVATION: This system enables cumulative learning WITHOUT
    catastrophic forgetting. Each successful execution is stored as an
    artifact and can be replayed, but never overwrites previous knowledge.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.artifacts: Dict[str, ExecutionArtifact] = {}
        self.execution_index: Dict[str, List[str]] = {}  # context_hash -> artifact_ids
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load persisted artifacts."""
        artifacts_file = self.storage_path / "artifacts.json"
        if artifacts_file.exists():
            with open(artifacts_file, "r") as f:
                data = json.load(f)
                for artifact_data in data.get("artifacts", []):
                    artifact = ExecutionArtifact.from_dict(artifact_data)
                    self.artifacts[artifact.artifact_id] = artifact
                self.execution_index = data.get("execution_index", {})
    
    def _save_artifacts(self):
        """Persist artifacts."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        artifacts_file = self.storage_path / "artifacts.json"
        data = {
            "artifacts": [a.to_dict() for a in self.artifacts.values()],
            "execution_index": self.execution_index,
        }
        with open(artifacts_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def store_artifact(self, artifact: ExecutionArtifact) -> None:
        """Store an execution artifact."""
        self.artifacts[artifact.artifact_id] = artifact
        
        # Index by context for fast retrieval
        context_hash = self._hash_context(artifact.context)
        if context_hash not in self.execution_index:
            self.execution_index[context_hash] = []
        self.execution_index[context_hash].append(artifact.artifact_id)
        
        self._save_artifacts()
    
    def get_artifact(self, artifact_id: str) -> Optional[ExecutionArtifact]:
        """Retrieve an artifact by ID."""
        return self.artifacts.get(artifact_id)
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash of context for indexing."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:16]
    
    def find_similar_artifacts(self, context: Dict[str, Any],
                               min_score: float = 0.5) -> List[ExecutionArtifact]:
        """Find artifacts with similar contexts/outcomes."""
        context_hash = self._hash_context(context)
        candidates = []
        
        # First, try exact context match
        if context_hash in self.execution_index:
            for artifact_id in self.execution_index[context_hash]:
                artifact = self.artifacts.get(artifact_id)
                if artifact and artifact.outcome_score >= min_score:
                    candidates.append(artifact)
        
        # If not enough, find partial matches
        if len(candidates) < 3:
            for artifact in self.artifacts.values():
                if artifact.outcome_score >= min_score and artifact not in candidates:
                    # Check context similarity
                    similarity = self._context_similarity(context, artifact.context)
                    if similarity > 0.5:
                        candidates.append(artifact)
        
        # Sort by outcome score
        candidates.sort(key=lambda a: a.outcome_score, reverse=True)
        return candidates[:10]  # Return top 10
    
    def _context_similarity(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        if not ctx1 or not ctx2:
            return 0.0
        
        similarity = 0.0
        weights = 0.0
        
        # Tissue match
        if ctx1.get("tissue") == ctx2.get("tissue"):
            similarity += 0.3
        weights += 0.3
        
        # Stress level similarity
        s1 = ctx1.get("stress_level", 0)
        s2 = ctx2.get("stress_level", 0)
        similarity += 1.0 - abs(s1 - s2)
        weights += 0.3
        
        # Developmental stage match
        if ctx1.get("developmental_stage") == ctx2.get("developmental_stage"):
            similarity += 0.2
        weights += 0.2
        
        # Signal overlap
        sig1 = set(ctx1.get("signal_molecules", {}).keys())
        sig2 = set(ctx2.get("signal_molecules", {}).keys())
        if sig1 and sig2:
            overlap = len(sig1 & sig2) / len(sig1 | sig2)
            similarity += overlap * 0.2
        weights += 0.2
        
        return similarity / weights if weights > 0 else 0.0
    
    def replay_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Replay an artifact to regenerate its execution state.
        
        This is key for reusing prior biological "experiences".
        """
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            return {"error": "Artifact not found"}
        
        # Return stored execution data
        return {
            "artifact_id": artifact.artifact_id,
            "execution_id": artifact.execution_id,
            "context": artifact.context,
            "gate_states": artifact.gate_states,
            "regulatory_paths": artifact.regulatory_paths,
            "expression_trajectories": artifact.expression_trajectories,
            "outcome_score": artifact.outcome_score,
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the artifact memory."""
        if not self.artifacts:
            return {"total_artifacts": 0}
        
        outcome_scores = [a.outcome_score for a in self.artifacts.values()]
        return {
            "total_artifacts": len(self.artifacts),
            "avg_outcome_score": sum(outcome_scores) / len(outcome_scores),
            "high_success_count": sum(1 for s in outcome_scores if s >= 0.8),
            "context_categories": len(self.execution_index),
        }
    
    def prune_low_value_artifacts(self, threshold: float = 0.3) -> int:
        """Remove artifacts with low outcome scores."""
        to_remove = []
        for artifact_id, artifact in self.artifacts.items():
            if artifact.outcome_score < threshold:
                to_remove.append(artifact_id)
        
        for artifact_id in to_remove:
            del self.artifacts[artifact_id]
        
        self._save_artifacts()
        return len(to_remove)


# ============================================================================
# MAIN EXECUTABLE GENOME FRAMEWORK
# ============================================================================

class ExecutableGenomeFramework:
    """Main framework orchestrating all adapters.
    
    This is the entry point for running biological "experiments"
    using the executable genome paradigm.
    
    Usage:
        framework = ExecutableGenomeFramework("/path/to/storage")
        
        # Set context (tissue, stress, signals, etc.)
        framework.set_context(tissue="liver", stress=0.3)
        
        # Execute the genome
        result = framework.execute_genome(duration=24.0, time_step=1.0)
        
        # Access results
        print(result.phenotype_scores)
        print(result.expression_trajectories)
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize all adapters
        self.genome_adapter = GenomeCoreAdapter(str(self.storage_path / "genome"))
        self.regulome_adapter = RegulomeGraphAdapter(str(self.storage_path / "regulome"))
        self.gate_adapter = EpigeneticGateManager(str(self.storage_path / "gates"))
        self.context_adapter = ContextEnvironmentAdapter(str(self.storage_path / "context"))
        self.expression_adapter = ExpressionDynamicsEngine(str(self.storage_path / "expression"))
        self.proteome_adapter = ProteomeTranslator(str(self.storage_path / "proteome"))
        self.phenotype_adapter = PhenotypeScorer(str(self.storage_path / "phenotype"))
        self.memory_adapter = ArtifactMemorySystem(str(self.storage_path / "memory"))
        
        # Current execution state
        self.current_execution_id: Optional[str] = None
        self.current_artifact: Optional[ExecutionArtifact] = None
        
        print(f"🧬 Executable Genome Framework initialized at {self.storage_path}")
    
    def set_context(self, tissue: str = "generic", stress: float = 0.0,
                   developmental_stage: str = "adult",
                   signals: Dict[str, float] = None,
                   **kwargs) -> None:
        """Set the biological context for execution."""
        context = ContextState(
            tissue=tissue,
            stress_level=stress,
            developmental_stage=developmental_stage,
            signal_molecules=signals or {},
            environmental_conditions=kwargs,
        )
        self.context_adapter.set_context(context)
    
    def load_genome_data(self, genome_data: Dict[str, Any]) -> None:
        """Load genome, regulome, and related data."""
        # Load genome core
        if "genome" in genome_data:
            self.genome_adapter.load_genome(genome_data["genome"])
        
        # Load regulatory elements
        if "regulome" in genome_data:
            for element_data in genome_data["regulome"].get("elements", []):
                element = RegulatoryElement.from_dict(element_data)
                self.regulome_adapter.add_regulatory_element(element)
            
            # Load regulatory edges
            for edge in genome_data["regulome"].get("edges", []):
                self.regulome_adapter.add_regulatory_edge(
                    edge["source"], edge["target"], edge["weight"]
                )
        
        # Create epigenetic gates for regulatory elements
        for element in self.regulome_adapter.elements.values():
            if self.gate_adapter.get_element_gate(element.element_id) is None:
                self.gate_adapter.create_gate(element.element_id, "methylation")
        
        print(f"✅ Loaded genome with {len(self.genome_adapter.genes)} genes, "
              f"{len(self.regulome_adapter.elements)} regulatory elements")
    
    def execute_genome(self, duration: float = 24.0, time_step: float = 1.0) -> ExecutionArtifact:
        """Execute the genome under current context.
        
        This is the main execution method. It:
        1. Activates transcription factors based on context
        2. Applies context to epigenetic gates
        3. Computes regulatory influence through the graph
        4. Executes expression dynamics over time
        5. Translates to proteins
        6. Scores phenotypes
        7. Stores artifact in memory
        
        Returns the complete execution artifact.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        print(f"🚀 Starting execution {execution_id}")
        print(f"   Context: {self.context_adapter.current_context.to_dict()}")
        
        # Step 1: Activate transcription factors
        context = self.context_adapter.current_context.to_dict()
        tf_activity = self.context_adapter.activate_transcription_factors(context)
        print(f"   TF activity: {sum(tf_activity.values())/len(tf_activity):.2f} avg")
        
        # Step 2: Apply context to epigenetic gates
        gate_activations = {}
        for time_idx in range(int(duration / time_step) + 1):
            t = time_idx * time_step
            activations = self.gate_adapter.apply_context_to_gates(context, t)
            gate_activations.update(activations)
        
        # Step 3: Compute regulatory influence
        regulatory_input = {}
        for gene_id in self.genome_adapter.genes.keys():
            # Get upstream regulatory elements
            influence = 0.0
            for element in self.regulome_adapter.elements.values():
                if gene_id in element.target_genes:
                    # Get gate activation for this element
                    element_gate = self.gate_adapter.get_element_gate(element.element_id)
                    gate_factor = 1.0
                    if element_gate:
                        gate_factor = gate_activations.get(element_gate.gate_id, 0.5)
                    
                    # Get regulatory influence
                    reg_influence = self.regulome_adapter.get_regulatory_influence(
                        element.element_id, {"tf_activity": tf_activity}
                    )
                    influence += reg_influence * gate_factor
            
            regulatory_input[gene_id] = influence
        
        # Step 4: Execute expression dynamics
        expression_trajectories = self.expression_adapter.execute_expression(
            regulatory_input, duration, time_step
        )
        print(f"   Expressed {len(expression_trajectories)} genes")
        
        # Step 5: Translate to proteins
        protein_data = self.proteome_adapter.translate_expression(
            expression_trajectories, self.genome_adapter
        )
        
        # Step 6: Score phenotype
        phenotype_score = self.phenotype_adapter.score_phenotype(
            expression_trajectories, protein_data
        )
        print(f"   Phenotype: viability={phenotype_score.viability_score:.2f}, "
              f"stability={phenotype_score.stability_score:.2f}")
        
        # Step 7: Create and store artifact
        execution_time = time.time() - start_time
        
        # Get regulatory pathways
        regulatory_paths = []
        for element in self.regulome_adapter.elements.values():
            for target_gene in element.target_genes:
                paths = self.regulome_adapter.get_regulatory_pathways(
                    element.element_id, target_gene
                )
                regulatory_paths.extend(paths)
        
        artifact = ExecutionArtifact(
            artifact_id=f"artifact_{uuid.uuid4().hex[:12]}",
            execution_id=execution_id,
            context=context,
            gate_states={
                gate_id: {
                    "methylation": gate.methylation_level,
                    "accessibility": gate.accessibility_score,
                }
                for gate_id, gate in self.gate_adapter.gates.items()
            },
            regulatory_paths=regulatory_paths,
            expression_trajectories={
                gene_id: traj.to_dict()
                for gene_id, traj in expression_trajectories.items()
            },
            phenotype_scores=phenotype_score.to_dict(),
            outcome_score=phenotype_score.fitness_proxy,
            execution_time=execution_time,
        )
        
        self.memory_adapter.store_artifact(artifact)
        self.current_execution_id = execution_id
        self.current_artifact = artifact
        
        print(f"✅ Execution {execution_id} complete in {execution_time:.2f}s")
        print(f"   Outcome score: {phenotype_score.fitness_proxy:.3f}")
        
        return artifact
    
    def find_similar_experiments(self, context: Dict[str, Any],
                                min_score: float = 0.5) -> List[ExecutionArtifact]:
        """Find similar prior experiments from memory."""
        return self.memory_adapter.find_similar_artifacts(context, min_score)
    
    def replay_experiment(self, artifact_id: str) -> Dict[str, Any]:
        """Replay a prior experiment."""
        return self.memory_adapter.replay_artifact(artifact_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about accumulated biological knowledge."""
        return self.memory_adapter.get_memory_stats()
    
    def get_stable_states(self) -> List[Dict[str, Any]]:
        """Get identified stable expression states."""
        return self.expression_adapter.get_stable_states()
    
    def add_regulatory_knowledge(self, element: RegulatoryElement) -> None:
        """Add new regulatory knowledge to the system."""
        self.regulome_adapter.add_regulatory_element(element)
        
        # Create gate for new element
        if self.gate_adapter.get_element_gate(element.element_id) is None:
            self.gate_adapter.create_gate(element.element_id, "methylation")
        
        print(f"🧬 Added regulatory element: {element.element_id}")
    
    def export_execution_report(self, artifact_id: str) -> Dict[str, Any]:
        """Generate a detailed execution report."""
        artifact = self.memory_adapter.get_artifact(artifact_id)
        if not artifact:
            return {"error": "Artifact not found"}
        
        return {
            "execution_summary": {
                "execution_id": artifact.execution_id,
                "outcome_score": artifact.outcome_score,
                "execution_time": artifact.execution_time,
            },
            "context": artifact.context,
            "expression_summary": {
                "genes_expressed": len(artifact.expression_trajectories),
                "trajectories": artifact.expression_trajectories,
            },
            "phenotype": artifact.phenotype_scores,
            "regulatory_paths": len(artifact.regulatory_paths),
            "gate_states": artifact.gate_states,
        }


# ============================================================================
# FRAMEWORK DEFINITION & SCIENTIFIC CONTEXT
# ============================================================================

"""
SCIENTIFIC FRAMING
==================

The Executable Genome Framework (EGF) represents a fundamental shift in 
computational biology: treating biological regulation as executable code 
rather than static equations or learned statistical patterns.

Key Theoretical Contributions:

1. GENOME-AS-PROGRAM PARADIGM
   - DNA is immutable "source code"
   - Regulation is "executable logic" executed under context
   - Gene expression is "program output"
   
2. STATEFUL REGULATORY EXECUTION
   - Epigenetic gates have persistent, history-dependent state
   - Context modifies gate behavior dynamically
   - State is never lost, only accumulated

3. NON-DESTRUCTIVE BIOLOGICAL LEARNING
   - Successful executions stored as artifacts
   - No weight updates, no catastrophic forgetting
   - Knowledge accumulates as execution episodes

4. EXECUTABLE REGULATORY GRAPHS
   - Regulatory relationships are executable logic
   - Edges represent causal influence, not correlation
   - Paths can be discovered and replayed

What Would Falsify This Framework:
- Demonstrating that biological regulation CANNOT be captured by 
  executable, stateful rules
- Showing that the artifact memory approach fails to improve 
  predictions over time
- Proving that context-dependent execution adds no predictive value

Key Differentiators from Existing Approaches:

                    EGF              Neural Nets        Simulators
                    ---              ----------         ---------
    Representation  Executable code  Learned weights    Differential eq
    Learning        Store episodes   Gradient descent   Parameter fit
    Memory          Cumulative       Catastrophic       Fixed
    Context         Dynamic          Input encoding     Fixed params
    State           Persistent       Stateless          Initial cond.
    Replay          Full traces      Implicit           Re-init only

The framework bridges:
- Systems biology (regulatory networks)
- AI memory architectures (episode-based learning)  
- Computational graphs (execution traces)
- Epigenetics (conceptually, not chemically)

This is not AlphaFold. AlphaFold predicts structure from sequence.
EGF represents regulation as an executable process that runs under context.
"""

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
