"""
Executable Genome Framework (EGF)
Formalizing Genome-as-Program Modeling

This module implements a novel computational paradigm where biological systems 
are represented as executable, modular, memory-preserving programs.
"""

import hashlib
import json
import time
import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.adapter_engine import Adapter, AdapterEngine, QuantumArtifact


@dataclass
class BiologicalArtifact:
    """Stores a complete 'biological execution episode'"""
    artifact_id: str
    raw_context: Dict[str, Any]
    processed_context: Dict[str, Any]
    context_hash: str
    gate_states: Dict[str, float]
    regulatory_paths: List[Tuple[str, str, float]]
    expression_results: Dict[str, float]
    outcome_scores: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "raw_context": self.raw_context,
            "processed_context": self.processed_context,
            "context_hash": self.context_hash,
            "gate_states": self.gate_states,
            "regulatory_paths": self.regulatory_paths,
            "expression_results": self.expression_results,
            "outcome_scores": self.outcome_scores,
            "timestamp": self.timestamp
        }


class GenomeCoreAdapter:
    """
    Stores DNA sequences, variants, and isoforms.
    Acts as the immutable biological source code.
    """
    def __init__(self, sequence_data: Dict[str, str]):
        self.sequence_data = sequence_data
        self.version = "1.0.0"
        self.memory: List[Any] = []

    def get_sequence(self, gene_id: str) -> Optional[str]:
        return self.sequence_data.get(gene_id)


class EpigeneticGateAdapter:
    """
    Stateful gates controlling regulatory edges.
    Context-dependent, persistent, history-aware.
    Mimics methylation / chromatin accessibility behavior.
    """
    def __init__(self):
        self.gate_states: Dict[str, float] = {}  # 0.0 (closed) to 1.0 (open)
        self.memory: List[Dict[str, float]] = []

    def update_gates(self, context: Dict[str, Any]):
        """Update gates based on tissue, signals, and past history"""
        # Logic to mimic chromatin accessibility
        for gate_id, current_state in self.gate_states.items():
            # Example: persistent state with context influence
            stress = context.get("stress", 0.0)
            tissue_bias = context.get("tissue_gates", {}).get(gate_id, 0.5)
            
            # Non-linear update
            new_state = (current_state * 0.8) + (tissue_bias * 0.1) - (stress * 0.05)
            self.gate_states[gate_id] = max(0.0, min(1.0, new_state))
        
        self.memory.append(self.gate_states.copy())

    def get_gate_multiplier(self, edge_id: str) -> float:
        return self.gate_states.get(edge_id, 1.0)


class RegulomeAdapter:
    """
    Represents promoters, enhancers, silencers, transcription factors.
    Implemented as an executable graph.
    """
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[Tuple[str, float]]] = {} # src -> [(dest, weight)]
        self.memory: List[Any] = []

    def add_regulatory_link(self, source: str, target: str, influence: float):
        self.nodes.add(source)
        self.nodes.add(target)
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append((target, influence))

    def execute_logic(self, active_tfs: Dict[str, float], gates: EpigeneticGateAdapter) -> Dict[str, float]:
        """Runs the regulatory logic to determine activation levels"""
        activations = active_tfs.copy()
        
        # Simple iterative activation spread (mimics regulatory cascade)
        for _ in range(3): # Depth of cascade
            new_activations = activations.copy()
            for src, targets in self.edges.items():
                if src in activations:
                    src_level = activations[src]
                    for dest, weight in targets:
                        gate_mult = gates.get_gate_multiplier(f"{src}->{dest}")
                        effective_influence = src_level * weight * gate_mult
                        new_activations[dest] = new_activations.get(dest, 0.0) + effective_influence
            activations = new_activations
            
        return activations


class ContextAdapter:
    """
    Inputs: tissue identity, stress, signals, nutrients, conditions.
    Activates transcription factors and regulatory pathways.
    """
    def __init__(self):
        self.current_context: Dict[str, Any] = {}
        self.memory: List[Dict[str, Any]] = []

    def process_environment(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.current_context = raw_inputs
        # Logic to activate initial TFs based on environment
        signals = raw_inputs.get("initial_signals", {}).copy()
        
        # Example: high glucose in environment activates TF_Glucose
        if raw_inputs.get("glucose_level", 0) > 0.5:
            signals["TF_Glucose"] = signals.get("TF_Glucose", 0.0) + 0.5
            
        self.current_context["processed_signals"] = signals
        self.memory.append(self.current_context.copy())
        return self.current_context


class ExpressionDynamicsAdapter:
    """
    Executes the regulatory graph over time.
    Produces continuous gene expression trajectories.
    """
    def __init__(self, regulome: RegulomeAdapter):
        self.regulome = regulome
        self.trajectories: List[Dict[str, float]] = []
        self.memory: List[List[Dict[str, float]]] = []

    def run_trajectory(self, initial_state: Dict[str, float], gates: EpigeneticGateAdapter, steps: int = 10):
        current_state = initial_state
        self.trajectories = [current_state]
        
        for _ in range(steps):
            current_state = self.regulome.execute_logic(current_state, gates)
            # Apply decay/homeostasis
            current_state = {k: v * 0.9 for k, v in current_state.items()}
            self.trajectories.append(current_state)
        
        self.memory.append(self.trajectories)
        return self.trajectories


class ProteomeAdapter:
    """
    Translates expression into protein abundance and functional embeddings.
    """
    def __init__(self):
        self.protein_abundance: Dict[str, float] = {}
        self.memory: List[Dict[str, float]] = []

    def translate(self, expression: Dict[str, float]):
        # Translation efficiency and half-life simulation
        self.protein_abundance = {gene: val * 1.2 for gene, val in expression.items()}
        self.memory.append(self.protein_abundance.copy())
        return self.protein_abundance


class PhenotypeAdapter:
    """
    Scores biological outcomes (stability, efficiency, viability proxies).
    """
    def __init__(self):
        self.memory: List[Dict[str, float]] = []

    def score_outcome(self, protein_levels: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
        # Example: scoring based on required proteins for a tissue
        target_proteins = context.get("required_proteins", {})
        stability = 1.0
        efficiency = 0.0
        
        for p, target in target_proteins.items():
            actual = protein_levels.get(p, 0.0)
            diff = abs(actual - target)
            stability -= (diff * 0.1)
            efficiency += actual
        
        score = {
            "stability": max(0.0, stability),
            "efficiency": efficiency,
            "viability": 1.0 if stability > 0.5 else 0.0
        }
        self.memory.append(score)
        return score


class ExecutableGenomeFramework:
    """
    The master framework that orchestrates biological execution and learning.
    """
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Adapters
        self.genome = GenomeCoreAdapter({})
        self.regulome = RegulomeAdapter()
        self.epigenetics = EpigeneticGateAdapter()
        self.context_engine = ContextAdapter()
        self.expression_engine = ExpressionDynamicsAdapter(self.regulome)
        self.proteome = ProteomeAdapter()
        self.phenotype = PhenotypeAdapter()
        
        # Artifact Memory
        self.memory: List[BiologicalArtifact] = []
        self._load_memory()

    def _load_memory(self):
        memory_file = self.storage_path / "biological_memory.json"
        if memory_file.exists():
            with open(memory_file, "r") as f:
                data = json.load(f)
                self.memory = []
                for a in data:
                    # Migration/Compatibility
                    if "context" in a and "raw_context" not in a:
                        a["raw_context"] = a.pop("context")
                        a["processed_context"] = a["raw_context"]
                    if "context_hash" not in a:
                        a["context_hash"] = self._compute_context_hash(a["raw_context"])
                    
                    # Ensure all fields are present for dataclass unpacking
                    try:
                        self.memory.append(BiologicalArtifact(**a))
                    except TypeError:
                        # Skip corrupted or incompatible entries
                        continue

    def _compute_context_hash(self, context: Dict[str, Any]) -> str:
        """
        Compute a stable hash of the context, ignoring transient fields.
        """
        # Create a copy to avoid mutating the original
        clean_ctx = copy.deepcopy(context)
        
        # Fields to ignore
        ignore_fields = ["timestamp", "processed_signals", "artifact_id", "runtime_ms", "name"]
        for field in ignore_fields:
            if field in clean_ctx:
                del clean_ctx[field]
        
        # Stable sort for consistent hashing
        ctx_json = json.dumps(clean_ctx, sort_keys=True)
        return hashlib.md5(ctx_json.encode()).hexdigest()

    def _save_memory(self):
        memory_file = self.storage_path / "biological_memory.json"
        with open(memory_file, "w") as f:
            json.dump([a.to_dict() for a in self.memory], f, indent=2)

    def execute(self, raw_context: Dict[str, Any]) -> BiologicalArtifact:
        """
        Runs a biological execution episode.
        Genome -> Program -> Execution -> Memory
        """
        start_time = time.time()
        
        # 0. Replay Lookup
        context_hash = self._compute_context_hash(raw_context)
        cached_artifact = next((a for a in self.memory if a.context_hash == context_hash), None)
        
        if cached_artifact:
            duration_ms = (time.time() - start_time) * 1000
            print(f"✅ REPLAY_HIT: {cached_artifact.artifact_id} (hash: {context_hash}) [{duration_ms:.2f}ms]")
            return cached_artifact
            
        print(f"🔄 REPLAY_MISS: Executing biological program... (hash: {context_hash})")

        # 1. Process environment/context (work on a deep copy)
        processed_context = self.context_engine.process_environment(copy.deepcopy(raw_context))
        
        # 2. Update epigenetic state based on context
        self.epigenetics.update_gates(processed_context)
        
        # 3. Get initial signals from context (Transcription Factors)
        initial_signals = processed_context.get("processed_signals", {})
        
        # 4. Run expression dynamics
        trajectory = self.expression_engine.run_trajectory(initial_signals, self.epigenetics)
        final_expression = trajectory[-1]
        
        # 5. Translate to proteome
        protein_levels = self.proteome.translate(final_expression)
        
        # 6. Score phenotype
        scores = self.phenotype.score_outcome(protein_levels, processed_context)
        
        # 7. Create Artifact
        artifact_id = f"episode_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        artifact = BiologicalArtifact(
            artifact_id=artifact_id,
            raw_context=raw_context,
            processed_context=processed_context,
            context_hash=context_hash,
            gate_states=self.epigenetics.gate_states.copy(),
            regulatory_paths=[], # Would be populated by graph traversal
            expression_results=final_expression,
            outcome_scores=scores
        )
        
        # 8. Learn: Store execution (unconditional storage for experiments)
        self.memory.append(artifact)
        self._save_memory()
            
        duration_ms = (time.time() - start_time) * 1000
        print(f"🎬 Execution complete in {duration_ms:.2f}ms. Artifact: {artifact_id}")
        return artifact

    def replay_experience(self, artifact_id: str):
        """Replays a previous biological episode to reinforce or adapt"""
        artifact = next((a for a in self.memory if a.artifact_id == artifact_id), None)
        if not artifact:
            return None
            
        # Set state from artifact
        self.epigenetics.gate_states = artifact.gate_states.copy()
        # Re-execute with same context
        return self.execute(artifact.raw_context)


if __name__ == "__main__":
    # Example setup
    egf = ExecutableGenomeFramework("./egf_data")
    
    # Configure some regulatory links
    egf.regulome.add_regulatory_link("TF_A", "GENE_X", 1.5)
    egf.regulome.add_regulatory_link("GENE_X", "GENE_Y", 0.8)
    egf.epigenetics.gate_states["TF_A->GENE_X"] = 1.0
    egf.epigenetics.gate_states["GENE_X->GENE_Y"] = 0.5
    
    # Run an execution
    context = {
        "tissue": "liver",
        "stress": 0.1,
        "initial_signals": {"TF_A": 1.0},
        "required_proteins": {"GENE_Y": 1.0}
    }
    
    result = egf.execute(context)
    print(f"Executed Episode: {result.artifact_id}")
    print(f"Outcome Scores: {result.outcome_scores}")
    print(f"Final Expression: {result.expression_results}")
