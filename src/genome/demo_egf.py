#!/usr/bin/env python3
"""
Demo: Executable Genome Framework (EGF)
========================================

This script demonstrates the core functionality of the EGF system,
showing how genomes can be treated as executable programs.

Run: python src/genome/demo_egf.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
# Since we're in src/genome/, we need to go up two levels
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct import from the executable_genome_framework module
from src.genome.executable_genome_framework import (
    ExecutableGenomeFramework,
    GenomeRegion,
    RegulatoryElement,
    ContextState,
    ExpressionTrajectory,
    PhenotypeScore,
)


def create_sample_genome():
    """Create a sample genome for demonstration."""
    return {
        "genome": {
            "regions": [
                {
                    "region_id": "promoter_TFAP2A",
                    "sequence": "ATGCATGCATGCATGCATGCATGCATGCATGC",
                    "region_type": "promoter",
                    "start": 1000,
                    "end": 1032,
                    "chromosome": "chr1",
                    "strand": "+"
                },
                {
                    "region_id": "exon_TFAP2A_1",
                    "sequence": "ATGCCGGCTAACTCGGTGCTGATCGATCGAT",
                    "region_type": "exon",
                    "start": 1033,
                    "end": 1064,
                    "chromosome": "chr1",
                    "strand": "+"
                },
                {
                    "region_id": "enhancer_P53_1",
                    "sequence": "GGATCCGGATCCGGATCCGGATCCGGATCCG",
                    "region_type": "enhancer",
                    "start": 5000,
                    "end": 5032,
                    "chromosome": "chr1",
                    "strand": "+"
                },
            ],
            "genes": {
                "TFAP2A": {
                    "name": "TFAP2A",
                    "description": "Transcription Factor AP-2 alpha",
                    "exonic_regions": ["exon_TFAP2A_1"],
                    "promoter_region": "promoter_TFAP2A",
                    "function": "transcription_factor",
                },
                "CDKN1A": {
                    "name": "CDKN1A",
                    "description": "Cyclin-dependent kinase inhibitor 1A",
                    "exonic_regions": ["exon_CDKN1A_1"],
                    "promoter_region": "promoter_CDKN1A",
                    "function": "cell_cycle_regulator",
                },
                "BAX": {
                    "name": "BAX",
                    "description": "BCL2 associated X protein",
                    "exonic_regions": ["exon_BAX_1"],
                    "promoter_region": "promoter_BAX",
                    "function": "apoptosis_regulator",
                },
                "MDM2": {
                    "name": "MDM2",
                    "description": "MDM2 proto-oncogene",
                    "exonic_regions": ["exon_MDM2_1"],
                    "promoter_region": "promoter_MDM2",
                    "function": "negative_p53_regulator",
                },
            },
            "isoforms": {
                "TFAP2A": [
                    {"isoform_id": "TFAP2A_v1", "exons": ["exon_TFAP2A_1"]},
                    {"isoform_id": "TFAP2A_v2", "exons": ["exon_TFAP2A_1"]},
                ]
            }
        },
        "regulome": {
            "elements": [
                {
                    "element_id": "enhancer_p53_1",
                    "element_type": "enhancer",
                    "target_genes": ["CDKN1A", "BAX"],
                    "tf_families": ["p53"],
                    "genomic_location": ["chr1", 5000, 5032],
                    "weight": 0.9,
                },
                {
                    "element_id": "enhancer_MDM2_1",
                    "element_type": "enhancer",
                    "target_genes": ["MDM2"],
                    "tf_families": ["p53"],
                    "genomic_location": ["chr1", 6000, 6032],
                    "weight": 0.7,
                },
                {
                    "element_id": "promoter_TFAP2A",
                    "element_type": "promoter",
                    "target_genes": ["TFAP2A"],
                    "tf_families": ["AP1", "homeobox"],
                    "genomic_location": ["chr1", 1000, 1032],
                    "weight": 0.8,
                },
                {
                    "element_id": "silencer_MDM2_1",
                    "element_type": "silencer",
                    "target_genes": ["MDM2"],
                    "tf_families": ["nuclear_receptor"],
                    "genomic_location": ["chr1", 7000, 7032],
                    "weight": 0.3,
                },
            ],
            "edges": [
                {"source": "enhancer_p53_1", "target": "CDKN1A", "weight": 0.9},
                {"source": "enhancer_p53_1", "target": "BAX", "weight": 0.85},
                {"source": "enhancer_MDM2_1", "target": "MDM2", "weight": 0.7},
                {"source": "promoter_TFAP2A", "target": "TFAP2A", "weight": 0.8},
            ]
        }
    }


def demo_basic_execution():
    """Demonstrate basic genome execution."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Genome Execution")
    print("="*70)
    
    # Initialize framework
    egf = ExecutableGenomeFramework("/tmp/egf_demo_basic")
    
    # Load genome data
    genome_data = create_sample_genome()
    egf.load_genome_data(genome_data)
    
    # Set context (stressed liver cells)
    egf.set_context(
        tissue="liver",
        stress=0.7,
        developmental_stage="adult",
        signals={"glucocorticoid": 0.5}
    )
    
    print("\nContext: Stressed liver cells with glucocorticoid signal")
    print("Executing genome for 24 hours...")
    
    # Execute genome
    start = time.time()
    result = egf.execute_genome(duration=24.0, time_step=1.0)
    elapsed = time.time() - start
    
    print(f"\nExecution completed in {elapsed:.3f}s")
    print(f"Outcome Score: {result.outcome_score:.3f}")
    print(f"\nPhenotype Scores:")
    print(f"  Viability: {result.phenotype_scores['viability_score']:.3f}")
    print(f"  Stability: {result.phenotype_scores['stability_score']:.3f}")
    print(f"  Efficiency: {result.phenotype_scores['efficiency_score']:.3f}")
    print(f"  Fitness Proxy: {result.phenotype_scores['fitness_proxy']:.3f}")
    
    print(f"\nExpressed {len(result.expression_trajectories)} genes")
    for gene_id, traj_data in list(result.expression_trajectories.items())[:3]:
        traj = ExpressionTrajectory(**traj_data)
        print(f"  {gene_id}: mean={traj.mean_expression:.2f}, "
              f"stability={traj.stability_score:.3f}")


def demo_context_dependence():
    """Demonstrate context-dependent execution."""
    print("\n" + "="*70)
    print("DEMO 2: Context-Dependent Execution")
    print("="*70)
    
    egf = ExecutableGenomeFramework("/tmp/egf_demo_context")
    egf.load_genome_data(create_sample_genome())
    
    contexts = [
        ("Normal liver", {"tissue": "liver", "stress": 0.0}),
        ("Stressed liver", {"tissue": "liver", "stress": 0.7}),
        ("Stressed + p53 signal", {"tissue": "liver", "stress": 0.7, 
                                    "signals": {"p53激活": 0.8}}),
        ("Immune cell, inflammation", {"tissue": "immune", "stress": 0.5,
                                        "signals": {"NFkB激活": 0.9}}),
    ]
    
    print("\nExecuting genome under different contexts:")
    
    for name, ctx_params in contexts:
        egf.set_context(**ctx_params)
        result = egf.execute_genome(duration=12.0, time_step=1.0)
        
        print(f"\n  {name}:")
        print(f"    Outcome: {result.outcome_score:.3f}")
        print(f"    Viability: {result.phenotype_scores['viability_score']:.3f}")
        
        # Show key pathway activations
        pathways = result.phenotype_scores.get('pathway_activity', {})
        for pathway, activity in list(pathways.items())[:2]:
            print(f"    {pathway}: {activity:.2f}")


def demo_memory_accumulation():
    """Demonstrate cumulative memory without forgetting."""
    print("\n" + "="*70)
    print("DEMO 3: Cumulative Memory (No Catastrophic Forgetting)")
    print("="*70)
    
    egf = ExecutableGenomeFramework("/tmp/egf_demo_memory")
    egf.load_genome_data(create_sample_genome())
    
    # Execute multiple experiments with different contexts
    experiments = [
        {"tissue": "liver", "stress": 0.2},
        {"tissue": "liver", "stress": 0.5},
        {"tissue": "liver", "stress": 0.8},
        {"tissue": "immune", "stress": 0.3},
        {"tissue": "brain", "stress": 0.1},
    ]
    
    print("\nExecuting 5 different experiments...")
    
    for i, ctx in enumerate(experiments):
        egf.set_context(**ctx)
        result = egf.execute_genome(duration=6.0, time_step=1.0)
        print(f"  Experiment {i+1} ({ctx['tissue']}, stress={ctx['stress']}): "
              f"outcome={result.outcome_score:.3f}")
    
    # Check memory statistics
    stats = egf.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  Total artifacts: {stats['total_artifacts']}")
    print(f"  Average outcome: {stats['avg_outcome_score']:.3f}")
    print(f"  High-success (≥0.8): {stats['high_success_count']}")
    print(f"  Context categories: {stats['context_categories']}")
    
    # Demonstrate knowledge retrieval
    print("\nFinding similar experiments to 'stressed liver':")
    similar = egf.find_similar_experiments(
        {"tissue": "liver", "stress": 0.6}, min_score=0.3
    )
    
    for artifact in similar[:3]:
        print(f"  - Similar context, outcome: {artifact.outcome_score:.3f}")
    
    # Verify no forgetting: early experiments are still accessible
    print("\nReplaying first experiment:")
    first_artifact_id = list(egf.memory_adapter.artifacts.keys())[0]
    replay = egf.replay_experiment(first_artifact_id)
    print(f"  Original context preserved: {replay['context']}")
    print(f"  Original outcome: {replay['outcome_score']:.3f}")


def demo_stable_states():
    """Demonstrate stable state discovery."""
    print("\n" + "="*70)
    print("DEMO 4: Stable State Discovery")
    print("="*70)
    
    egf = ExecutableGenomeFramework("/tmp/egf_demo_stable")
    egf.load_genome_data(create_sample_genome())
    
    # Execute with normal context
    egf.set_context(tissue="liver", stress=0.2)
    egf.execute_genome(duration=48.0, time_step=1.0)
    
    # Execute with stressed context  
    egf.set_context(tissue="liver", stress=0.8)
    egf.execute_genome(duration=48.0, time_step=1.0)
    
    # Get stable states
    stable_states = egf.get_stable_states()
    
    print(f"\nIdentified {len(stable_states)} stable expression states:")
    
    for state in stable_states:
        print(f"  {state['gene_id']}:")
        print(f"    Expression level: {state['expression_level']:.2f}")
        print(f"    Stability: {state['stability']:.3f}")
        print(f"    Type: {state.get('trajectory_type', 'continuous')}")


def demo_regulatory_pathways():
    """Demonstrate regulatory pathway discovery."""
    print("\n" + "="*70)
    print("DEMO 5: Regulatory Pathway Discovery")
    print("="*70)
    
    egf = ExecutableGenomeFramework("/tmp/egf_demo_pathways")
    egf.load_genome_data(create_sample_genome())
    
    # Execute genome
    egf.set_context(tissue="liver", stress=0.5)
    result = egf.execute_genome(duration=12.0, time_step=1.0)
    
    print(f"\nExecuted with {len(result.regulatory_paths)} regulatory pathways")
    
    # Show some pathways
    print("\nSample regulatory pathways:")
    for i, path in enumerate(result.regulatory_paths[:5]):
        print(f"  {' → '.join(path)}")
    
    # Show gate states
    print(f"\nEpigenetic gate states ({len(result.gate_states)} gates):")
    for gate_id, states in list(result.gate_states.items())[:3]:
        print(f"  {gate_id}:")
        print(f"    Methylation: {states.get('methylation', 'N/A'):.2f}")
        print(f"    Accessibility: {states.get('accessibility', 'N/A'):.2f}")


def demo_adapters_independently():
    """Demonstrate individual adapters working independently."""
    print("\n" + "="*70)
    print("DEMO 6: Individual Adapters")
    print("="*70)
    
    from src.genome import (
        GenomeCoreAdapter,
        RegulomeGraphAdapter,
        ContextEnvironmentAdapter,
    )
    
    print("\n1. Genome Core Adapter:")
    genome_adapter = GenomeCoreAdapter("/tmp/egf_demo_core")
    
    # Add a gene
    genome_adapter.add_region(GenomeRegion(
        region_id="test_exon",
        sequence="ATGCCCGGGAAATTTCCCGGG",
        region_type="exon",
        start=1000,
        end=1021,
        chromosome="chr1"
    ))
    
    genome_adapter.add_gene("TEST_GENE", {
        "name": "TEST_GENE",
        "description": "Test gene for demo",
        "exonic_regions": ["test_exon"],
        "function": "test",
    })
    
    seq = genome_adapter.get_sequence("test_exon")
    gene = genome_adapter.get_gene("TEST_GENE")
    print(f"  Added gene: {gene['name']}")
    print(f"  Sequence retrieved: {seq[:10]}...")
    
    print("\n2. Regulome Adapter:")
    regulome_adapter = RegulomeGraphAdapter("/tmp/egf_demo_regulome")
    
    element = RegulatoryElement(
        element_id="demo_enhancer",
        element_type="enhancer",
        target_genes=["TEST_GENE"],
        tf_families=["bHLH"],
        genomic_location=("chr1", 2000, 2030),
        weight=0.8
    )
    regulome_adapter.add_regulatory_element(element)
    regulome_adapter.add_regulatory_edge("demo_enhancer", "TEST_GENE", 0.7)
    
    influence = regulome_adapter.get_regulatory_influence(
        "demo_enhancer", {"tf_activity": {"bHLH": 0.6}}
    )
    print(f"  Added regulatory element: {element.element_type}")
    print(f"  Regulatory influence: {influence:.3f}")
    
    print("\n3. Context Adapter:")
    context_adapter = ContextEnvironmentAdapter("/tmp/egf_demo_context")
    
    tf_activity = context_adapter.activate_transcription_factors({
        "tissue": "liver",
        "stress": 0.5,
        "signals": {}
    })
    print(f"  TF families activated: {len(tf_activity)}")
    print(f"  Sample activity: p53={tf_activity.get('p53', 0):.2f}, "
          f"NFkB={tf_activity.get('NFkB', 0):.2f}")


def main():
    """Run all demonstrations."""
    print("="*70)
    print("EXECUTABLE GENOME FRAMEWORK (EGF) - Demonstration")
    print("="*70)
    print("\nThis demo shows how EGF treats genomes as executable programs")
    print("that run under context, maintain state, and accumulate memory.")
    
    # Run demos
    try:
        demo_basic_execution()
        demo_context_dependence()
        demo_memory_accumulation()
        demo_stable_states()
        demo_regulatory_pathways()
        demo_adapters_independently()
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("""
Key Takeaways:
1. Genome executes as program under biological context
2. Different contexts produce different expression outcomes
3. Memory accumulates without forgetting previous experiments
4. Stable expression states can be discovered
5. Regulatory pathways are executable logic, not static correlations
6. Adapters are modular and can be used independently

The EGF paradigm: Genome → Program → Execution → Memory → Knowledge
        """)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
