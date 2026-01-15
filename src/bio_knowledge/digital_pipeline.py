"""
The World-Breaking Digital Pipeline (Main Execution)

This script implements the full pipeline:
1. DNA Construction & Quantum Optimization
2. Virtual Cell Environment Setup
3. Multiversal Hypothesis Testing (39 hypotheses)
4. Result Scoring and TCL Summarization
"""

import json
import os
from pathlib import Path
from src.bio_knowledge.biological_database import BiologicalKnowledgeBase
from src.bio_knowledge.dna_generator import DNAGenerator
from src.multiversal.virtual_cell import VirtualCellEnvironment
from src.bio_knowledge.cancer_hypothesis_generator import CancerHypothesisGenerator

def run_pipeline():
    print("🚀 Initializing World-Breaking Digital Pipeline...")
    
    # 1. Setup Knowledge Base
    bio_kb = BiologicalKnowledgeBase()
    
    # 2. Digitally Construct the DNA Strand
    print("🧬 Step 1: Digitally Constructing DNA Strand...")
    dna_gen = DNAGenerator()
    pik3ca_dna = dna_gen.get_gene_sequence("PIK3CA")
    
    if not pik3ca_dna:
        print("❌ Failed to get PIK3CA sequence")
        return

    print("⚛️ Enhancing DNA with Quantum H-bond Engine...")
    optimized_dna = dna_gen.quantum_optimize_dna(pik3ca_dna)
    fasta_path = dna_gen.export_fasta(optimized_dna)
    print(f"✅ Quantum-optimized DNA saved to: {fasta_path}")
    print(f"📊 Quantum stats: {optimized_dna.quantum_stats}")

    # 3. Build the Virtual Cell Environment
    print("🧫 Step 2: Building Virtual Cell Environment...")
    env = VirtualCellEnvironment(bio_kb)
    # Start with PIK3CA hotspot mutation
    mutations = ["PIK3CA:H1047R"] 
    cells = env.create_cell_line(optimized_dna.sequence, mutations, count=100)
    print(f"✅ Created 100 parallel virtual cells with PIK3CA:H1047R mutation")

    # 4. Digitally Test the Cancer Treatments
    print("🧪 Step 3: Digitally Testing 39 Hypotheses...")
    
    # Load hypotheses
    hypotheses_path = Path("cancer_artifacts/hypotheses/cancer_hypotheses_detailed.json")
    if not hypotheses_path.exists():
        print("⚠️ Hypotheses file not found. Generating fresh hypotheses...")
        generator = CancerHypothesisGenerator(bio_kb)
        summary = generator.generate_hypotheses(n_proteins=15)
        hypotheses = summary["hypotheses"]
    else:
        with open(hypotheses_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "hypotheses" in data:
                hypotheses = data["hypotheses"]
            else:
                hypotheses = data
            
    print(f"✅ Loaded {len(hypotheses)} hypotheses for testing")

    # Run at scale
    print("🌌 Step 4: Running Multiversal Branching Simulation...")
    simulation_results = env.run_multiversal_simulation(cells, hypotheses, steps=20)
    
    # 5. Output Results
    print("📝 Step 5: Generating Final Report...")
    report = env.generate_report(simulation_results)
    
    output_dir = Path("cancer_artifacts/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "simulation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        
    # Save detailed JSON results
    with open(output_dir / "simulation_results_detailed.json", "w") as f:
        json.dump(simulation_results, f, indent=2)

    print(f"✅ Pipeline complete! Final report saved to: {report_path}")
    
    # Print top candidate
    sorted_results = sorted(simulation_results.values(), key=lambda x: x["viability_score"], reverse=True)
    if sorted_results:
        top = sorted_results[0]
        h = top['hypothesis']
        drug_name = "Unknown"
        if "suggested_drug" in h and isinstance(h["suggested_drug"], dict):
            drug_name = h["suggested_drug"].get("name", "Unknown")
        elif "drug_name" in h:
            drug_name = h["drug_name"]
        elif "drug" in h:
            drug_name = h["drug"]
            
        print(f"\n🏆 TOP CANDIDATE: {drug_name}")
        print(f"📈 Cure Rate: {top['cured_percentage']:.1f}%")
        print(f"🧠 TCL Outcome: {top['tcl_outcome']}")

if __name__ == "__main__":
    run_pipeline()
