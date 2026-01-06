"""
Executable Genome Framework (EGF)
A novel computational paradigm where biological systems are represented as executable, 
modular, memory-preserving programs rather than static data or retrained models.

This framework demonstrates that genomes can be executed as programs that:
- Run under biological context
- Maintain persistent regulatory state
- Learn without catastrophic forgetting
- Improve through replayable biological "experiences"
"""

import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from pathlib import Path
from enum import Enum

from ..core.adapter_engine import AdapterEngine, QuantumArtifact


class BiologicalState(Enum):
    """Biological execution states"""
    DORMANT = "dormant"
    ACTIVE = "active"
    REGULATING = "regulating"
    EXPRESSING = "expressing"
    ADAPTING = "adapting"


class RegulatoryMode(Enum):
    """Gene regulatory modes"""
    BASAL = "basal"
    INDUCED = "induced"
    REPRESSED = "repressed"
    CONDITIONAL = "conditional"


class ExecutionContext(Enum):
    """Biological context types"""
    NORMAL = "normal"
    STRESS = "stress"
    DEVELOPMENT = "development"
    DISEASE = "disease"
    TREATMENT = "treatment"


@dataclass
class EpigeneticGate:
    """Stateful gate controlling regulatory edges"""
    gate_id: str
    current_state: bool = False
    methylation_level: float = 0.0
    chromatin_accessibility: float = 1.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity: float = 1.0
    hysteresis_memory: float = 0.0
    
    def update_state(self, input_signal: float, context: ExecutionContext) -> bool:
        """Update gate state based on input signal and context"""
        # Apply context-dependent sensitivity
        effective_signal = input_signal * self.sensitivity
        
        # Apply hysteresis for memory effect
        if abs(effective_signal) < self.hysteresis_memory:
            effective_signal *= 0.1
        
        # Update methylation level (persistent memory)
        if effective_signal > 0.5:
            self.methylation_level = min(1.0, self.methylation_level + 0.1)
        elif effective_signal < -0.5:
            self.methylation_level = max(0.0, self.methylation_level - 0.05)
        
        # Update chromatin accessibility
        self.chromatin_accessibility = 1.0 - self.methylation_level * 0.8
        
        # Store in history
        self.history.append({
            "timestamp": time.time(),
            "input_signal": input_signal,
            "context": context.value,
            "methylation_level": self.methylation_level,
            "chromatin_accessibility": self.chromatin_accessibility
        })
        
        # Calculate new state
        threshold = 0.3 + (self.methylation_level * 0.4)
        new_state = effective_signal > threshold
        
        # Apply accessibility filter
        if self.chromatin_accessibility < 0.2:
            new_state = False
        
        self.current_state = new_state
        return new_state


@dataclass
class RegulatoryEdge:
    """Executable regulatory influence"""
    edge_id: str
    source_gene: str
    target_gene: str
    weight: float
    gate: EpigeneticGate
    regulatory_mode: RegulatoryMode = RegulatoryMode.BASAL
    context_dependencies: Set[ExecutionContext] = field(default_factory=set)
    
    def execute(self, context: ExecutionContext, source_expression: float) -> float:
        """Execute regulatory influence"""
        if context not in self.context_dependencies and self.context_dependencies:
            return 0.0
        
        gate_output = self.gate.update_state(source_expression, context)
        
        if not gate_output:
            return 0.0
        
        # Apply regulatory mode
        if self.regulatory_mode == RegulatoryMode.REPRESSED:
            return -self.weight * source_expression
        else:
            return self.weight * source_expression


class GenomeCoreAdapter:
    """
    Genome Core Adapter - Stores DNA sequences, variants, isoforms
    Acts as immutable biological source code
    """
    
    def __init__(self, genome_id: str, sequence_data: Dict[str, Any]):
        self.genome_id = genome_id
        self.sequence_data = sequence_data
        self.variants = sequence_data.get("variants", [])
        self.isoforms = sequence_data.get("isoforms", [])
        self.created_at = time.time()
        self.modifications = []
    
    def get_gene_sequence(self, gene_id: str) -> Optional[str]:
        """Retrieve gene sequence by ID"""
        return self.sequence_data.get("genes", {}).get(gene_id, {}).get("sequence")
    
    def get_isoform_variants(self, gene_id: str) -> List[str]:
        """Get all isoform variants for a gene"""
        return self.isoforms.get(gene_id, [])
    
    def add_genomic_variant(self, variant_data: Dict[str, Any]):
        """Add new genomic variant (immutable learning)"""
        self.variants.append({
            **variant_data,
            "discovered_at": time.time()
        })
        self.modifications.append({
            "type": "variant_added",
            "variant": variant_data,
            "timestamp": time.time()
        })


class RegulomeAdapter:
    """
    Regulome Adapter - Represents promoters, enhancers, silencers, transcription factors
    Implemented as an executable graph
    """
    
    def __init__(self, regulome_id: str):
        self.regulome_id = regulome_id
        self.regulatory_network = {}
        self.transcription_factors = {}
        self.regulatory_elements = {}
        self.execution_history = []
    
    def add_regulatory_element(self, element_id: str, element_type: str, 
                             sequence: str, position: int):
        """Add regulatory element to network"""
        self.regulatory_elements[element_id] = {
            "type": element_type,
            "sequence": sequence,
            "position": position,
            "binding_proteins": [],
            "activity_history": []
        }
    
    def add_transcription_factor(self, tf_id: str, binding_motifs: List[str]):
        """Add transcription factor"""
        self.transcription_factors[tf_id] = {
            "binding_motifs": binding_motifs,
            "target_genes": [],
            "activity_state": False,
            "expression_level": 0.0
        }
    
    def create_regulatory_edge(self, source_gene: str, target_gene: str, 
                             weight: float, gate: EpigeneticGate) -> RegulatoryEdge:
        """Create executable regulatory edge"""
        edge_id = f"{source_gene}->{target_gene}"
        
        edge = RegulatoryEdge(
            edge_id=edge_id,
            source_gene=source_gene,
            target_gene=target_gene,
            weight=weight,
            gate=gate
        )
        
        self.regulatory_network[edge_id] = edge
        return edge
    
    def execute_regulatory_network(self, context: ExecutionContext, 
                                 gene_expressions: Dict[str, float]) -> Dict[str, float]:
        """Execute regulatory network computation"""
        new_expressions = gene_expressions.copy()
        execution_log = {
            "timestamp": time.time(),
            "context": context.value,
            "input_expressions": gene_expressions.copy(),
            "regulatory_updates": {}
        }
        
        # Execute each regulatory edge
        for edge_id, edge in self.regulatory_network.items():
            if edge.source_gene in gene_expressions:
                regulatory_output = edge.execute(context, gene_expressions[edge.source_gene])
                
                if edge.target_gene not in execution_log["regulatory_updates"]:
                    execution_log["regulatory_updates"][edge.target_gene] = []
                
                execution_log["regulatory_updates"][edge.target_gene].append({
                    "source": edge.source_gene,
                    "output": regulatory_output,
                    "gate_state": edge.gate.current_state
                })
                
                # Apply regulatory effect
                if edge.target_gene in new_expressions:
                    new_expressions[edge.target_gene] += regulatory_output
        
        self.execution_history.append(execution_log)
        return new_expressions


class EpigeneticGateAdapter:
    """
    Epigenetic Gate Adapter - Stateful gates controlling regulatory edges
    Context-dependent, persistent, history-aware
    """
    
    def __init__(self, gate_id: str, sensitivity: float = 1.0):
        self.gate_id = gate_id
        self.epigenetic_gates = {}
        self.global_methylation_state = 0.0
        self.chromatin_states = {}
        self.memory_preservation_factor = 0.95
        
    def create_epigenetic_gate(self, target_gene: str, initial_state: bool = False,
                             sensitivity: float = 1.0) -> EpigeneticGate:
        """Create epigenetic gate for a gene"""
        gate = EpigeneticGate(
            gate_id=f"{self.gate_id}_{target_gene}",
            current_state=initial_state,
            sensitivity=sensitivity
        )
        
        self.epigenetic_gates[target_gene] = gate
        return gate
    
    def update_global_state(self, stress_level: float, treatment_effects: Dict[str, float]):
        """Update global epigenetic state"""
        self.global_methylation_state += stress_level * 0.1
        
        for gene, effect in treatment_effects.items():
            if gene in self.epigenetic_gates:
                gate = self.epigenetic_gates[gene]
                gate.methylation_level += effect * 0.2
                gate.chromatin_accessibility = max(0.0, 1.0 - gate.methylation_level * 0.8)
    
    def get_gate_state(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get current epigenetic gate state"""
        if gene_id in self.epigenetic_gates:
            gate = self.epigenetic_gates[gene_id]
            return {
                "state": gate.current_state,
                "methylation_level": gate.methylation_level,
                "chromatin_accessibility": gate.chromatin_accessibility,
                "history_length": len(gate.history)
            }
        return None


class ContextEnvironmentAdapter:
    """
    Context/Environment Adapter
    Inputs: tissue identity, stress, signals, nutrients, conditions
    Activates transcription factors and regulatory pathways
    """
    
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.environmental_conditions = {}
        self.tissue_contexts = {}
        self.signal_cascade = {}
        self.activation_history = []
        
    def set_environmental_condition(self, condition: str, intensity: float):
        """Set environmental condition intensity"""
        self.environmental_conditions[condition] = intensity
    
    def set_tissue_context(self, tissue_type: str, context_data: Dict[str, Any]):
        """Set tissue-specific context"""
        self.tissue_contexts[tissue_type] = {
            **context_data,
            "timestamp": time.time()
        }
    
    def activate_pathway(self, pathway_id: str, activation_level: float):
        """Activate biological pathway"""
        self.signal_cascade[pathway_id] = {
            "activation_level": activation_level,
            "timestamp": time.time(),
            "duration": 0.0
        }
    
    def get_context_vector(self, tissue_type: str) -> Dict[str, float]:
        """Get context activation vector"""
        context_vector = {}
        
        # Environmental contributions
        for condition, intensity in self.environmental_conditions.items():
            context_vector[f"env_{condition}"] = intensity
        
        # Tissue-specific contributions
        if tissue_type in self.tissue_contexts:
            tissue_data = self.tissue_contexts[tissue_type]
            for key, value in tissue_data.items():
                if isinstance(value, (int, float)):
                    context_vector[f"tissue_{key}"] = float(value)
        
        # Signal cascade contributions
        for pathway, signal_data in self.signal_cascade.items():
            context_vector[f"signal_{pathway}"] = signal_data["activation_level"]
        
        return context_vector


class ExpressionDynamicsAdapter:
    """
    Expression Dynamics Adapter
    Executes the regulatory graph over time
    Produces continuous gene expression trajectories
    """
    
    def __init__(self, dynamics_id: str):
        self.dynamics_id = dynamics_id
        self.expression_trajectories = {}
        self.stability_matrix = {}
        self.oscillation_patterns = {}
        
    def simulate_expression_dynamics(self, initial_expressions: Dict[str, float],
                                   regulome: RegulomeAdapter,
                                   context: ExecutionContext,
                                   time_steps: int = 100,
                                   dt: float = 0.1) -> Dict[str, List[float]]:
        """Simulate gene expression dynamics over time"""
        trajectories = {gene: [expr] for gene, expr in initial_expressions.items()}
        
        current_expressions = initial_expressions.copy()
        
        for step in range(time_steps):
            # Execute regulatory network
            new_expressions = regulome.execute_regulatory_network(context, current_expressions)
            
            # Add noise and degradation
            for gene in new_expressions:
                degradation = current_expressions.get(gene, 0.0) * 0.05
                noise = (hashlib.md5(f"{gene}_{step}_{time.time()}".encode()).digest()[0] - 128) * 0.01
                new_expressions[gene] = max(0.0, new_expressions[gene] - degradation + noise)
            
            # Store trajectory
            for gene, expr in new_expressions.items():
                if gene not in trajectories:
                    trajectories[gene] = [expr]
                trajectories[gene].append(expr)
            
            current_expressions = new_expressions
        
        return trajectories
    
    def analyze_stability(self, trajectories: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Analyze expression stability and oscillations"""
        stability_analysis = {}
        
        for gene, expression_series in trajectories.items():
            if len(expression_series) < 10:
                continue
                
            # Calculate stability metrics
            mean_expr = sum(expression_series) / len(expression_series)
            variance = sum((x - mean_expr) ** 2 for x in expression_series) / len(expression_series)
            
            # Detect oscillations
            oscillation_strength = 0.0
            if len(expression_series) > 20:
                # Simple period detection
                periods = []
                for i in range(10, len(expression_series) - 10):
                    if expression_series[i] > expression_series[i-1] and expression_series[i] > expression_series[i+1]:
                        periods.append(i)
                
                if len(periods) > 2:
                    period_diffs = [periods[i+1] - periods[i] for i in range(len(periods)-1)]
                    if period_diffs:
                        oscillation_strength = 1.0 / (sum(period_diffs) / len(period_diffs))
            
            stability_analysis[gene] = {
                "mean_expression": mean_expr,
                "variance": variance,
                "stability_score": 1.0 / (1.0 + variance),
                "oscillation_strength": oscillation_strength,
                "steady_state": variance < 0.01
            }
        
        return stability_analysis


class ProteomeAdapter:
    """
    Proteome Adapter
    Translates expression into protein abundance and functional embeddings
    """
    
    def __init__(self, proteome_id: str):
        self.proteome_id = proteome_id
        self.protein_abundance_map = {}
        self.functional_embeddings = {}
        self.structure_predictions = {}
        
    def translate_expression_to_proteins(self, gene_expressions: Dict[str, float],
                                       genome_core: GenomeCoreAdapter) -> Dict[str, float]:
        """Translate gene expressions to protein abundances"""
        protein_abundances = {}
        
        for gene_id, expression_level in gene_expressions.items():
            # Get gene sequence
            gene_sequence = genome_core.get_gene_sequence(gene_id)
            if not gene_sequence:
                continue
            
            # Calculate translation efficiency
            sequence_length = len(gene_sequence)
            translation_efficiency = min(1.0, 1000.0 / sequence_length)
            
            # Apply post-translational modifiers
            protein_id = f"protein_{gene_id}"
            abundance = expression_level * translation_efficiency
            
            # Store result
            protein_abundances[protein_id] = abundance
            
            # Update abundance map
            if gene_id not in self.protein_abundance_map:
                self.protein_abundance_map[gene_id] = []
            self.protein_abundance_map[gene_id].append({
                "timestamp": time.time(),
                "abundance": abundance,
                "expression_level": expression_level
            })
        
        return protein_abundances
    
    def generate_functional_embedding(self, protein_abundances: Dict[str, float]) -> Dict[str, Any]:
        """Generate functional embedding of proteome state"""
        # Simple functional categorization
        functional_categories = {
            "metabolic": 0.0,
            "signaling": 0.0,
            "structural": 0.0,
            "regulatory": 0.0,
            "stress_response": 0.0
        }
        
        total_abundance = sum(protein_abundances.values())
        if total_abundance == 0:
            return {"functional_profile": functional_categories, "complexity_score": 0.0}
        
        # Categorize proteins based on naming conventions
        for protein_id, abundance in protein_abundances.items():
            normalized_abundance = abundance / total_abundance
            
            if "kinase" in protein_id.lower() or "phosphatase" in protein_id.lower():
                functional_categories["signaling"] += normalized_abundance
            elif "structural" in protein_id.lower() or "actin" in protein_id.lower():
                functional_categories["structural"] += normalized_abundance
            elif "regulatory" in protein_id.lower() or "tf" in protein_id.lower():
                functional_categories["regulatory"] += normalized_abundance
            elif "stress" in protein_id.lower() or "heat" in protein_id.lower():
                functional_categories["stress_response"] += normalized_abundance
            else:
                functional_categories["metabolic"] += normalized_abundance
        
        # Calculate complexity score
        non_zero_categories = sum(1 for score in functional_categories.values() if score > 0)
        complexity_score = non_zero_categories / len(functional_categories)
        
        return {
            "functional_profile": functional_categories,
            "complexity_score": complexity_score,
            "total_abundance": total_abundance,
            "dominant_category": max(functional_categories.items(), key=lambda x: x[1])[0]
        }


class OutcomePhenotypeAdapter:
    """
    Outcome/Phenotype Adapter
    Scores biological outcomes and defines experiment success criteria
    """
    
    def __init__(self, outcome_id: str):
        self.outcome_id = outcome_id
        self.phenotype_scores = {}
        self.success_criteria = {}
        self.evaluation_history = []
        
    def define_success_criteria(self, criteria: Dict[str, Any]):
        """Define experiment success criteria"""
        self.success_criteria = {
            **criteria,
            "defined_at": time.time()
        }
    
    def evaluate_phenotype(self, proteome_state: Dict[str, Any],
                          expression_stability: Dict[str, Dict[str, float]],
                          context: ExecutionContext) -> Dict[str, float]:
        """Evaluate phenotype based on biological state"""
        phenotype_scores = {}
        
        # Stability score
        stable_genes = sum(1 for gene_data in expression_stability.values() 
                         if gene_data.get("steady_state", False))
        total_genes = len(expression_stability)
        phenotype_scores["stability"] = stable_genes / max(1, total_genes) if total_genes > 0 else 0.0
        
        # Efficiency score (based on dominant functional category)
        functional_profile = proteome_state.get("functional_profile", {})
        dominant_category_score = max(functional_profile.values()) if functional_profile else 0.0
        phenotype_scores["efficiency"] = dominant_category_score
        
        # Complexity score
        phenotype_scores["complexity"] = proteome_state.get("complexity_score", 0.0)
        
        # Context adaptation score
        context_adaptation = 1.0
        if context == ExecutionContext.STRESS:
            # Higher stress response indicates better adaptation
            stress_response = functional_profile.get("stress_response", 0.0)
            context_adaptation = stress_response
        elif context == ExecutionContext.DEVELOPMENT:
            # Balanced profile indicates proper development
            context_adaptation = 1.0 - abs(0.5 - phenotype_scores["complexity"])
        
        phenotype_scores["adaptation"] = context_adaptation
        
        # Overall viability proxy
        phenotype_scores["viability"] = (
            phenotype_scores["stability"] * 0.3 +
            phenotype_scores["efficiency"] * 0.3 +
            phenotype_scores["complexity"] * 0.2 +
            phenotype_scores["adaptation"] * 0.2
        )
        
        return phenotype_scores
    
    def is_successful_experiment(self, phenotype_scores: Dict[str, float]) -> bool:
        """Determine if experiment meets success criteria"""
        if not self.success_criteria:
            return True  # No criteria = always successful
        
        for metric, threshold in self.success_criteria.items():
            if metric in phenotype_scores:
                if phenotype_scores[metric] < threshold:
                    return False
        
        return True


class BiologicalExecutionArtifact:
    """
    Artifact Memory System
    Stores complete "biological execution episodes" for replay and learning
    """
    
    def __init__(self, artifact_id: str, experiment_type: str):
        self.artifact_id = artifact_id
        self.experiment_type = experiment_type
        self.execution_episode = {
            "context": {},
            "initial_state": {},
            "regulatory_execution": [],
            "expression_trajectories": {},
            "final_state": {},
            "phenotype_scores": {},
            "success": False,
            "timestamp": time.time()
        }
        self.replay_count = 0
        self.learning_value = 0.0
    
    def record_execution_episode(self, context: ExecutionContext,
                                genome_state: Dict[str, Any],
                                regulatory_execution: List[Dict[str, Any]],
                                expression_trajectories: Dict[str, List[float]],
                                final_state: Dict[str, Any],
                                phenotype_scores: Dict[str, float],
                                success: bool):
        """Record complete biological execution episode"""
        self.execution_episode.update({
            "context": context.value,
            "genome_state": genome_state,
            "regulatory_execution": regulatory_execution,
            "expression_trajectories": expression_trajectories,
            "final_state": final_state,
            "phenotype_scores": phenotype_scores,
            "success": success
        })
        
        # Calculate learning value
        self.learning_value = self._calculate_learning_value(phenotype_scores, success)
    
    def replay_episode(self) -> Dict[str, Any]:
        """Replay biological execution episode"""
        self.replay_count += 1
        return {
            "artifact_id": self.artifact_id,
            "replay_count": self.replay_count,
            "execution_episode": self.execution_episode.copy()
        }
    
    def _calculate_learning_value(self, phenotype_scores: Dict[str, float], success: bool) -> float:
        """Calculate learning value of episode"""
        base_value = 1.0 if success else 0.5
        
        # Boost value based on phenotype scores
        if "viability" in phenotype_scores:
            base_value *= (0.5 + phenotype_scores["viability"])
        
        # Boost for novel context combinations
        complexity_factor = len(phenotype_scores) / 5.0  # Normalize to typical 5 metrics
        base_value *= min(1.0, complexity_factor)
        
        return min(1.0, base_value)


class ExecutableGenomeFramework:
    """
    Main Executable Genome Framework
    Coordinates all adapters for biological program execution
    """
    
    def __init__(self, framework_id: str, adapter_engine: AdapterEngine):
        self.framework_id = framework_id
        self.adapter_engine = adapter_engine
        
        # Initialize all adapters
        self.genome_core = None
        self.regulome = None
        self.epigenetic_gates = None
        self.context_environment = None
        self.expression_dynamics = None
        self.proteome = None
        self.phenotype_outcome = None
        
        # Artifact memory system
        self.biological_artifacts = {}
        self.artifacts_path = Path("./biological_artifacts")
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Execution state
        self.current_state = BiologicalState.DORMANT
        self.execution_history = []
    
    def initialize_genome_system(self, genome_data: Dict[str, Any]) -> bool:
        """Initialize complete genome system"""
        try:
            # Create genome core
            self.genome_core = GenomeCoreAdapter(
                genome_id=genome_data.get("genome_id", "default_genome"),
                sequence_data=genome_data
            )
            
            # Create regulome
            self.regulome = RegulomeAdapter("regulome_v1")
            
            # Create epigenetic gates
            self.epigenetic_gates = EpigeneticGateAdapter("epigenetic_v1")
            
            # Create context environment
            self.context_environment = ContextEnvironmentAdapter("context_v1")
            
            # Create expression dynamics
            self.expression_dynamics = ExpressionDynamicsAdapter("dynamics_v1")
            
            # Create proteome
            self.proteome = ProteomeAdapter("proteome_v1")
            
            # Create phenotype outcome
            self.phenotype_outcome = OutcomePhenotypeAdapter("phenotype_v1")
            
            self.current_state = BiologicalState.ACTIVE
            return True
            
        except Exception as e:
            print(f"Failed to initialize genome system: {e}")
            return False
    
    def execute_biological_program(self, context: ExecutionContext,
                                 environmental_conditions: Dict[str, float],
                                 tissue_type: str,
                                 initial_gene_expressions: Dict[str, float],
                                 time_steps: int = 100) -> Dict[str, Any]:
        """Execute complete biological program"""
        
        if self.current_state != BiologicalState.ACTIVE:
            return {"error": "Genome system not initialized"}
        
        self.current_state = BiologicalState.REGULATING
        
        try:
            # Step 1: Set context environment
            for condition, intensity in environmental_conditions.items():
                self.context_environment.set_environmental_condition(condition, intensity)
            
            self.context_environment.set_tissue_context(tissue_type, {
                "type": tissue_type,
                "activation_level": 1.0
            })
            
            # Step 2: Simulate expression dynamics
            trajectories = self.expression_dynamics.simulate_expression_dynamics(
                initial_gene_expressions, self.regulome, context, time_steps
            )
            
            # Step 3: Analyze stability
            final_expressions = {gene: expr_series[-1] for gene, expr_series in trajectories.items()}
            stability_analysis = self.expression_dynamics.analyze_stability(trajectories)
            
            # Step 4: Translate to proteome
            protein_abundances = self.proteome.translate_expression_to_proteins(
                final_expressions, self.genome_core
            )
            
            functional_embedding = self.proteome.generate_functional_embedding(protein_abundances)
            
            # Step 5: Evaluate phenotype
            phenotype_scores = self.phenotype_outcome.evaluate_phenotype(
                functional_embedding, stability_analysis, context
            )
            
            # Step 6: Determine success
            success = self.phenotype_outcome.is_successful_experiment(phenotype_scores)
            
            # Step 7: Create biological artifact
            artifact = BiologicalExecutionArtifact(
                artifact_id=f"bio_exp_{uuid.uuid4().hex[:8]}",
                experiment_type="genome_execution"
            )
            
            artifact.record_execution_episode(
                context=context,
                genome_state={"gene_expressions": final_expressions},
                regulatory_execution=self.regulome.execution_history.copy(),
                expression_trajectories=trajectories,
                final_state={
                    "protein_abundances": protein_abundances,
                    "functional_embedding": functional_embedding
                },
                phenotype_scores=phenotype_scores,
                success=success
            )
            
            # Store artifact
            self.biological_artifacts[artifact.artifact_id] = artifact
            self._save_artifact(artifact)
            
            self.current_state = BiologicalState.ACTIVE
            
            return {
                "execution_id": artifact.artifact_id,
                "success": success,
                "context": context.value,
                "final_expressions": final_expressions,
                "protein_abundances": protein_abundances,
                "functional_embedding": functional_embedding,
                "phenotype_scores": phenotype_scores,
                "trajectories": trajectories,
                "stability_analysis": stability_analysis,
                "artifacts_created": 1
            }
            
        except Exception as e:
            self.current_state = BiologicalState.DORMANT
            return {"error": f"Biological program execution failed: {e}"}
    
    def learn_from_experiments(self) -> Dict[str, Any]:
        """Learn from successful biological experiments"""
        successful_artifacts = [
            artifact for artifact in self.biological_artifacts.values()
            if artifact.learning_value > 0.7
        ]
        
        if not successful_artifacts:
            return {"message": "No high-value learning artifacts found"}
        
        # Analyze successful patterns
        context_patterns = {}
        regulatory_patterns = {}
        
        for artifact in successful_artifacts:
            episode = artifact.execution_episode
            context = episode.get("context", "unknown")
            
            if context not in context_patterns:
                context_patterns[context] = []
            context_patterns[context].append(artifact.learning_value)
        
        # Generate learning insights
        insights = {
            "successful_contexts": list(context_patterns.keys()),
            "total_learning_artifacts": len(successful_artifacts),
            "average_learning_value": sum(a.learning_value for a in successful_artifacts) / len(successful_artifacts),
            "cumulative_knowledge": len(self.biological_artifacts)
        }
        
        return insights
    
    def replay_successful_patterns(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Replay successful biological patterns for current context"""
        relevant_artifacts = []
        
        for artifact in self.biological_artifacts.values():
            if (artifact.execution_episode.get("context") == context.value and 
                artifact.learning_value > 0.5):
                relevant_artifacts.append(artifact.replay_episode())
        
        return relevant_artifacts
    
    def _save_artifact(self, artifact: BiologicalExecutionArtifact):
        """Save biological artifact to disk"""
        artifact_path = self.artifacts_path / f"{artifact.artifact_id}.json"
        with open(artifact_path, 'w') as f:
            json.dump({
                "artifact_id": artifact.artifact_id,
                "experiment_type": artifact.experiment_type,
                "execution_episode": artifact.execution_episode,
                "replay_count": artifact.replay_count,
                "learning_value": artifact.learning_value
            }, f, indent=2)


__all__ = [
    "ExecutableGenomeFramework",
    "BiologicalState",
    "ExecutionContext",
    "RegulatoryMode",
    "EpigeneticGate",
    "RegulatoryEdge",
    "GenomeCoreAdapter",
    "RegulomeAdapter", 
    "EpigeneticGateAdapter",
    "ContextEnvironmentAdapter",
    "ExpressionDynamicsAdapter",
    "ProteomeAdapter",
    "OutcomePhenotypeAdapter",
    "BiologicalExecutionArtifact"
]