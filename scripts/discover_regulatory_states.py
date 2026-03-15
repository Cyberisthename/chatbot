#!/usr/bin/env python3
"""
Regulatory State Discovery Experiment
======================================

This script demonstrates how the EGF framework can discover regulatory states
through constraint-based exploration, similar to the negative information
experiment in the quantum module.

The key insight is that by exploring what ISN'T expressed (constraints),
the system can discover more about regulatory logic than by observing
positive expression alone.

Run: python scripts/discover_regulatory_states.py
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genome import (
    ExecutableGenomeFramework,
    GenomeRegion,
    RegulatoryElement,
    ExecutionArtifact,
)


@dataclass
class RegulatoryConstraint:
    """Represents a constraint on regulatory state."""
    constraint_id: str
    gene_id: str
    constraint_type: str  # "must_not_express", "must_express", "max_expression", "min_expression"
    threshold: float
    reason: str


class RegulatoryStateDiscoverer:
    """Discovers regulatory states through constraint-based exploration.
    
    This implements the insight that "negative information" (what should NOT
    happen) is often more informative than positive information.
    
    Three experimental branches:
    1. BASELINE: Standard execution without constraints
    2. CONSTRAINT-ONLY: Apply negative constraints (what to avoid)
    3. DIRECT-MEASUREMENT: Collapse to satisfying constraints
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize EGF framework
        self.egf = ExecutableGenomeFramework(str(self.storage_path / "egf"))
        
        # Setup genome
        self._setup_genome()
        
        # Experiment tracking
        self.branch_results: Dict[str, Dict[str, Any]] = {}
        self.constraints: List[RegulatoryConstraint] = []
        
    def _setup_genome(self):
        """Setup a test genome with regulatory network."""
        genome_data = {
            "genome": {
                "regions": [
                    {"region_id": f"exon_{gene}", "sequence": "ATG"*100, 
                     "region_type": "exon", "start": 1000, "end": 1300, "chromosome": "chr1"}
                    for gene in ["GATA1", "KLF1", "BCL11A", "MYC", "MDM2", "P53"]
                ],
                "genes": {
                    "GATA1": {"name": "GATA1", "exonic_regions": ["exon_GATA1"], "function": "erythroid"},
                    "KLF1": {"name": "KLF1", "exonic_regions": ["exon_KLF1"], "function": "erythroid"},
                    "BCL11A": {"name": "BCL11A", "exonic_regions": ["exon_BCL11A"], "function": "repressor"},
                    "MYC": {"name": "MYC", "exonic_regions": ["exon_MYC"], "function": "proliferation"},
                    "MDM2": {"name": "MDM2", "exonic_regions": ["exon_MDM2"], "function": "p53_regulator"},
                    "P53": {"name": "TP53", "exonic_regions": ["exon_P53"], "function": "tumor_suppressor"},
                },
                "isoforms": {}
            },
            "regulome": {
                "elements": [
                    {"element_id": "enhancer_P53", "element_type": "enhancer",
                     "target_genes": ["P53"], "tf_families": ["p53"], 
                     "genomic_location": ["chr1", 5000, 5100], "weight": 0.9},
                    {"element_id": "enhancer_MDM2", "element_type": "enhancer", 
                     "target_genes": ["MDM2"], "tf_families": ["p53"], 
                     "genomic_location": ["chr1", 5200, 5300], "weight": 0.7},
                    {"element_id": "promoter_GATA1", "element_type": "promoter",
                     "target_genes": ["GATA1"], "tf_families": ["GATA"], 
                     "genomic_location": ["chr1", 6000, 6100], "weight": 0.8},
                    {"element_id": "enhancer_BCL11A", "element_type": "enhancer",
                     "target_genes": ["BCL11A"], "tf_families": ["BCL11A"], 
                     "genomic_location": ["chr1", 7000, 7100], "weight": 0.6},
                    {"element_id": "repressor_MYC", "element_type": "silencer",
                     "target_genes": ["MYC"], "tf_families": ["p53"], 
                     "genomic_location": ["chr1", 8000, 8100], "weight": 0.5},
                ],
                "edges": [
                    {"source": "enhancer_P53", "target": "P53", "weight": 0.9},
                    {"source": "enhancer_MDM2", "target": "MDM2", "weight": 0.7},
                    {"source": "promoter_GATA1", "target": "GATA1", "weight": 0.8},
                    {"source": "enhancer_BCL11A", "target": "BCL11A", "weight": 0.6},
                    {"source": "repressor_MYC", "target": "MYC", "weight": 0.5},
                ]
            }
        }
        
        self.egf.load_genome_data(genome_data)
        
    def add_constraint(self, gene_id: str, constraint_type: str, 
                      threshold: float, reason: str) -> RegulatoryConstraint:
        """Add a regulatory constraint."""
        constraint = RegulatoryConstraint(
            constraint_id=f"constr_{len(self.constraints)}",
            gene_id=gene_id,
            constraint_type=constraint_type,
            threshold=threshold,
            reason=reason
        )
        self.constraints.append(constraint)
        return constraint
        
    def run_branch_baseline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Branch A: Standard execution without constraints."""
        print("\n--- BRANCH A: BASELINE (No constraints) ---")
        
        self.egf.set_context(**context)
        artifact = self.egf.execute_genome(duration=24.0, time_step=1.0)
        
        return {
            "branch": "baseline",
            "artifact_id": artifact.artifact_id,
            "outcome_score": artifact.outcome_score,
            "expression": self._extract_expression(artifact),
            "entropy": self._compute_entropy(artifact),
            "support_size": len(artifact.expression_trajectories),
        }
    
    def run_branch_constraints(self, context: Dict[str, Any], 
                              max_iterations: int = 20) -> Dict[str, Any]:
        """Branch B: Execution with negative constraints.
        
        This branch explores what regulatory states should be AVOIDED
        (negative information) and learns from the constraints.
        """
        print("\n--- BRANCH B: CONSTRAINT-ONLY (Negative information) ---")
        
        # Set up constraint context
        constraint_context = context.copy()
        constraint_context["constraints"] = [
            {"gene": c.gene_id, "type": c.constraint_type, "threshold": c.threshold}
            for c in self.constraints
        ]
        
        self.egf.set_context(**constraint_context)
        
        # Track constraint satisfaction over iterations
        constraint_satisfaction = []
        expression_history = []
        
        for iteration in range(max_iterations):
            artifact = self.egf.execute_genome(duration=12.0, time_step=1.0)
            
            expression = self._extract_expression(artifact)
            expression_history.append(expression)
            
            # Check constraint satisfaction
            satisfied = 0
            total = 0
            for constraint in self.constraints:
                gene_expr = expression.get(constraint.gene_id, 0)
                total += 1
                
                if constraint.constraint_type == "must_not_express":
                    if gene_expr < constraint.threshold:
                        satisfied += 1
                elif constraint.constraint_type == "must_express":
                    if gene_expr > constraint.threshold:
                        satisfied += 1
                        
            satisfaction_rate = satisfied / total if total > 0 else 0
            constraint_satisfaction.append(satisfaction_rate)
            
            print(f"  Iteration {iteration+1}: constraint satisfaction = {satisfaction_rate:.2%}")
            
            # Check for saturation
            if len(constraint_satisfaction) > 5:
                recent = constraint_satisfaction[-5:]
                if max(recent) - min(recent) < 0.05:
                    print(f"  Saturation reached at iteration {iteration+1}")
                    break
        
        final_expression = expression_history[-1]
        
        # Compute metrics
        entropy = self._compute_entropy_from_expressions(expression_history)
        support_size = len(final_expression)
        
        return {
            "branch": "constraints",
            "iterations": len(expression_history),
            "final_expression": final_expression,
            "constraint_satisfaction": constraint_satisfaction,
            "saturation_point": len(constraint_satisfaction),
            "entropy": entropy,
            "support_size": support_size,
            "information_gain": self._compute_information_gain(
                expression_history, constraint_satisfaction
            ),
        }
    
    def run_branch_direct(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Branch C: Direct measurement with constraint collapse.
        
        This branch uses constraints to "collapse" the regulatory state
        to a configuration that satisfies all constraints.
        """
        print("\n--- BRANCH C: DIRECT MEASUREMENT (Constraint collapse) ---")
        
        # Set up with collapse-inducing context
        collapse_context = context.copy()
        collapse_context["collapse_mode"] = True
        collapse_context["target_constraints"] = [
            {"gene": c.gene_id, "type": c.constraint_type, "threshold": c.threshold}
            for c in self.constraints
        ]
        
        self.egf.set_context(**collapse_context)
        artifact = self.egf.execute_genome(duration=24.0, time_step=1.0)
        
        expression = self._extract_expression(artifact)
        
        # Measure constraint satisfaction after collapse
        satisfied = 0
        for constraint in self.constraints:
            gene_expr = expression.get(constraint.gene_id, 0)
            if constraint.constraint_type == "must_not_express":
                if gene_expr < constraint.threshold:
                    satisfied += 1
            elif constraint.constraint_type == "must_express":
                if gene_expr > constraint.threshold:
                    satisfied += 1
        
        satisfaction_rate = satisfied / len(self.constraints) if self.constraints else 1.0
        
        return {
            "branch": "direct",
            "artifact_id": artifact.artifact_id,
            "outcome_score": artifact.outcome_score,
            "expression": expression,
            "constraint_satisfaction_rate": satisfaction_rate,
            "satisfied_constraints": satisfied,
            "total_constraints": len(self.constraints),
        }
    
    def run_comparative_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all three branches and compare."""
        print("\n" + "="*60)
        print("REGULATORY STATE DISCOVERY - COMPARATIVE ANALYSIS")
        print("="*60)
        
        # Run baseline
        baseline = self.run_branch_baseline(context)
        
        # Run constraint exploration
        constraints = self.run_branch_constraints(context)
        
        # Run direct measurement
        direct = self.run_branch_direct(context)
        
        # Compute comparative metrics
        analysis = {
            "context": context,
            "branches": {
                "baseline": baseline,
                "constraints": constraints,
                "direct": direct,
            },
            "comparative_metrics": {
                "information_gain_constraints": constraints.get("information_gain", 0),
                "constraint_satisfaction_baseline": self._compute_constraint_satisfaction(
                    baseline["expression"], self.constraints
                ),
                "constraint_satisfaction_constraints": constraints.get(
                    "constraint_satisfaction", [0]
                )[-1],
                "constraint_satisfaction_direct": direct["constraint_satisfaction_rate"],
            },
            "insights": self._generate_insights(baseline, constraints, direct),
        }
        
        return analysis
    
    def _extract_expression(self, artifact: ExecutionArtifact) -> Dict[str, float]:
        """Extract mean expression levels from artifact."""
        expression = {}
        for gene_id, traj_data in artifact.expression_trajectories.items():
            if isinstance(traj_data, dict):
                expression[gene_id] = traj_data.get("mean_expression", 0)
            else:
                expression[gene_id] = 0
        return expression
    
    def _compute_entropy(self, artifact: ExecutionArtifact) -> float:
        """Compute entropy of expression distribution."""
        expression = self._extract_expression(artifact)
        values = list(expression.values())
        total = sum(values) if sum(values) > 0 else 1
        
        # Shannon entropy
        entropy = 0
        for v in values:
            if v > 0:
                p = v / total
                entropy -= p * (p + 1e-10).bit_length()  # log2
        
        return max(0, entropy)
    
    def _compute_entropy_from_expressions(self, expressions: List[Dict[str, float]]) -> float:
        """Compute average entropy across expression history."""
        if not expressions:
            return 0
        
        entropies = []
        for expr in expressions:
            total = sum(expr.values()) if sum(expr.values()) > 0 else 1
            entropy = 0
            for v in expr.values():
                if v > 0:
                    p = v / total
                    entropy -= p * (p + 1e-10).bit_length()
            entropies.append(max(0, entropy))
        
        return sum(entropies) / len(entropies)
    
    def _compute_information_gain(self, expressions: List[Dict[str, float]],
                                  satisfaction: List[float]) -> float:
        """Compute information gain from constraint exploration."""
        if len(expressions) < 2 or len(satisfaction) < 2:
            return 0
        
        # Information gain = reduction in uncertainty about constraints
        initial_uncertainty = 1.0 - satisfaction[0]
        final_uncertainty = 1.0 - satisfaction[-1]
        
        return max(0, initial_uncertainty - final_uncertainty)
    
    def _compute_constraint_satisfaction(self, expression: Dict[str, float],
                                         constraints: List[RegulatoryConstraint]) -> float:
        """Compute fraction of constraints satisfied by expression."""
        if not constraints:
            return 1.0
            
        satisfied = 0
        for constraint in constraints:
            gene_expr = expression.get(constraint.gene_id, 0)
            if constraint.constraint_type == "must_not_express":
                if gene_expr < constraint.threshold:
                    satisfied += 1
            elif constraint.constraint_type == "must_express":
                if gene_expr > constraint.threshold:
                    satisfied += 1
        
        return satisfied / len(constraints)
    
    def _generate_insights(self, baseline: Dict, constraints: Dict, 
                          direct: Dict) -> List[str]:
        """Generate insights from comparative analysis."""
        insights = []
        
        # Check if constraints improved outcome
        baseline_score = baseline.get("outcome_score", 0)
        direct_score = direct.get("outcome_score", 0)
        
        if direct_score > baseline_score:
            insights.append(
                f"Direct measurement (score={direct_score:.3f}) outperformed "
                f"baseline (score={baseline_score:.3f})"
            )
        
        # Check information gain
        info_gain = constraints.get("information_gain", 0)
        if info_gain > 0.1:
            insights.append(
                f"Constraint exploration achieved {info_gain:.3f} information gain"
            )
        
        # Check constraint satisfaction
        sat_direct = direct.get("constraint_satisfaction_rate", 0)
        sat_base = baseline_score  # Use baseline outcome as proxy
        if sat_direct > sat_base:
            insights.append(
                f"Constraint collapse achieved {sat_direct:.1%} constraint satisfaction"
            )
        
        return insights


def main():
    """Run regulatory state discovery experiment."""
    print("="*60)
    print("REGULATORY STATE DISCOVERY EXPERIMENT")
    print("Using Negative Information to Discover Regulatory Logic")
    print("="*60)
    
    discoverer = RegulatoryStateDiscoverer("/tmp/regulatory_discovery")
    
    # Define constraints based on biological knowledge
    # These represent "negative information" - what should NOT happen
    discoverer.add_constraint(
        gene_id="BCL11A",
        constraint_type="must_not_express",
        threshold=10.0,
        reason="BCL11A should be silenced in erythroid differentiation"
    )
    
    discoverer.add_constraint(
        gene_id="MDM2",
        constraint_type="max_expression",
        threshold=50.0,
        reason="MDM2 should be regulated, not overexpressed"
    )
    
    discoverer.add_constraint(
        gene_id="P53",
        constraint_type="must_express",
        threshold=20.0,
        reason="P53 tumor suppressor should be active"
    )
    
    print(f"\nAdded {len(discoverer.constraints)} regulatory constraints")
    for c in discoverer.constraints:
        print(f"  - {c.gene_id}: {c.constraint_type} < {c.threshold} ({c.reason})")
    
    # Run experiment under stress context
    context = {
        "tissue": "liver",
        "stress": 0.5,
        "developmental_stage": "adult",
        "signals": {}
    }
    
    # Run comparative analysis
    results = discoverer.run_comparative_analysis(context)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nBaseline Execution:")
    print(f"  Outcome score: {results['branches']['baseline']['outcome_score']:.3f}")
    print(f"  Entropy: {results['branches']['baseline']['entropy']:.3f}")
    
    print("\nConstraint Exploration:")
    print(f"  Iterations: {results['branches']['constraints']['iterations']}")
    print(f"  Information gain: {results['branches']['constraints']['information_gain']:.3f}")
    print(f"  Final constraint satisfaction: {results['branches']['constraints']['constraint_satisfaction'][-1]:.1%}")
    
    print("\nDirect Measurement:")
    print(f"  Outcome score: {results['branches']['direct']['outcome_score']:.3f}")
    print(f"  Constraints satisfied: {results['branches']['direct']['satisfied_constraints']}/{results['branches']['direct']['total_constraints']}")
    
    print("\nComparative Metrics:")
    metrics = results['comparative_metrics']
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nInsights:")
    for insight in results.get('insights', []):
        print(f"  • {insight}")
    
    # Save results
    output_file = Path("/tmp/regulatory_discovery/results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        serializable = {
            "branches": {},
            "comparative_metrics": results["comparative_metrics"],
            "insights": results["insights"],
        }
        for branch_name, branch_data in results["branches"].items():
            serializable["branches"][branch_name] = branch_data
        
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
