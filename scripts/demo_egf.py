#!/usr/bin/env python3
"""
Demo of the Executable Genome Framework (EGF)
"""
import sys
import os

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.egf.genome_program import ExecutableGenomeFramework

def run_demo():
    print("🧬 Initializing Executable Genome Framework (EGF)...")
    egf = ExecutableGenomeFramework("./egf_demo_data")
    
    # 1. Define the 'Regulome' (The program logic)
    # We define a simple metabolic response pathway
    print("🛠️  Configuring Regulome logic...")
    # TF_Glucose activates Gene_Insulin
    egf.regulome.add_regulatory_link("TF_Glucose", "GENE_Insulin", 2.0)
    # Gene_Insulin activates Gene_GlucoseUptake
    egf.regulome.add_regulatory_link("GENE_Insulin", "GENE_GlucoseUptake", 1.5)
    # TF_Stress inhibits the Insulin pathway via a gate
    egf.regulome.add_regulatory_link("TF_Stress", "GENE_Insulin", -1.0)
    
    # 2. Set initial Epigenetic States (Program Memory)
    # Open the gates for the insulin pathway
    egf.epigenetics.gate_states["TF_Glucose->GENE_Insulin"] = 0.8
    egf.epigenetics.gate_states["GENE_Insulin->GENE_GlucoseUptake"] = 1.0
    
    # 3. Execute in different contexts
    contexts = [
        {
            "name": "Normal High Glucose",
            "tissue": "pancreas",
            "stress": 0.0,
            "initial_signals": {"TF_Glucose": 1.0},
            "required_proteins": {"GENE_GlucoseUptake": 320.0}
        },
        {
            "name": "High Stress / Low Glucose",
            "tissue": "pancreas",
            "stress": 0.9,
            "initial_signals": {"TF_Glucose": 0.2, "TF_Stress": 0.8},
            "required_proteins": {"GENE_GlucoseUptake": -100.0}
        }
    ]
    
    for i, ctx in enumerate(contexts):
        print(f"\n🚀 Running Execution: {ctx['name']}")
        result = egf.execute(ctx)
        print(f"   Artifact ID: {result.artifact_id}")
        print(f"   Outcome: {result.outcome_scores}")
        print(f"   Key Expression: Insulin={result.expression_results.get('GENE_Insulin', 0):.4f}, "
              f"Uptake={result.expression_results.get('GENE_GlucoseUptake', 0):.4f}")
        
        if i == 0 and result.regulatory_paths:
            print(f"   Execution Trace (first 3): {result.regulatory_paths[:3]}")

    # 4. Show Memory (Non-destructive learning)
    print(f"\n📚 Total episodes in memory: {len(egf.memory)}")
    if egf.memory:
        last_id = egf.memory[-1].artifact_id
        print(f"🔄 Replaying last successful experience: {last_id}")
        replay_result = egf.replay_experience(last_id)
        print(f"   Replay Outcome: {replay_result.outcome_scores}")

if __name__ == "__main__":
    run_demo()
