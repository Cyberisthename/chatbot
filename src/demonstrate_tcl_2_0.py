import sys
import os

# Import project modules

from chatbot.src.bio_knowledge.biological_database import BiologicalKnowledgeBase, PathwayType
from chatbot.src.bio_knowledge.tcl_quantum_integrator import TCLQuantumIntegrator
from chatbot.src.thought_compression.tcl_types import Evidence, TCLMetadata, HyperEdge
from chatbot.src.thought_compression.tcl_symbols import TCLSymbol, SymbolType

def run_demonstration():
    print("🚀 Starting TCL 2.0 Hypergraph Demonstration...")
    
    # 1. Initialize Biological Knowledge Base
    bio_kb = BiologicalKnowledgeBase()
    integrator = TCLQuantumIntegrator(bio_kb)
    
    session_id = integrator.session_id
    context = integrator.tcl_engine.sessions[session_id]
    
    print(f"✅ Session initialized: {session_id}")
    
    # 2. Define a multi-causal scientific link (Genomics + Quantum Physics + Clinical)
    
    # Genomics Symbol
    kras_mutation = TCLSymbol(
        id="kras_mut_g12d",
        name="KRAS_G12D",
        type=SymbolType.CONCEPT,
        definition="Specific KRAS mutation common in pancreatic cancer",
        relationships={"proliferation": 0.9},
        causal_links=[],
        compression_ratio=0.8,
        cognitive_weight=0.9,
        evidence=[Evidence("TCGA Database", 0.98, "2024-01-10")],
        metadata=TCLMetadata(spatial_scale="molecular", domain="genetics")
    )
    
    # Quantum Physics Symbol
    proton_tunneling = TCLSymbol(
        id="dna_proton_tunneling",
        name="DNA_Proton_Tunneling",
        type=SymbolType.PRIMITIVE,
        definition="Quantum tunneling of protons in DNA base pairs leading to tautomeric shifts",
        relationships={"mutation": 0.7},
        causal_links=[],
        compression_ratio=0.7,
        cognitive_weight=0.85,
        evidence=[Evidence("Quantum Biology Research (2023)", 0.8, "2023-11-15")],
        metadata=TCLMetadata(spatial_scale="quantum", domain="quantum_physics")
    )
    
    # Clinical Symbol
    survival_data = TCLSymbol(
        id="clinical_survival_low",
        name="Low_Survival_Rate",
        type=SymbolType.CONCEPT,
        definition="Reduced 5-year survival in patients with aggressive mutations",
        relationships={"cancer": 0.95},
        causal_links=[],
        compression_ratio=0.9,
        cognitive_weight=1.0,
        evidence=[Evidence("Clinical Oncology Journal", 0.95, "2024-02-20")],
        metadata=TCLMetadata(spatial_scale="organism", domain="clinical_oncology")
    )
    
    # 3. Add symbols to the graph
    context.symbols.add_symbol(kras_mutation)
    context.symbols.add_symbol(proton_tunneling)
    context.symbols.add_symbol(survival_data)
    
    # 4. Create a HyperEdge connecting all three
    # This represents a hypothesis: Quantum tunneling causes KRAS mutations which leads to low survival.
    quantum_clinical_link = HyperEdge(
        id="quantum_genomics_clinical_hyperlink",
        nodes=[proton_tunneling.id, kras_mutation.id, survival_data.id],
        weight=0.88,
        edge_type="causal_chain",
        evidence=[Evidence("Hypothesis: Quantum-driven Oncogenesis", 0.6, "2024-04-17")],
        metadata=TCLMetadata(domain="multidisciplinary", tags={"quantum_biology", "genomics", "clinical"})
    )
    context.symbols.add_hyperedge(quantum_clinical_link)
    
    print(f"✅ Added 3-node HyperEdge: {quantum_clinical_link.id}")
    
    # 5. Demonstrate Analogy Operator
    print("\n🔍 Testing Analogy Operator...")
    # Note: Analogy operator relies on structural similarity. 
    # Let's add some causal links to make it work.
    kras_mutation.causal_links = ["link1"]
    survival_data.causal_links = ["link2"]
    
    # Refresh metrics to include new hyperedges
    integrator.tcl_engine.refresh_metrics(session_id)
    
    reasoning_result = integrator.tcl_engine.enhance_reasoning(session_id, "How does KRAS_G12D affect survival?")
    
    print(f"   Found {len(reasoning_result.get('analogies', []))} analogies.")
    for analogy in reasoning_result.get('analogies', []):
        print(f"   - {analogy['description']}")
    
    # 6. Check Metrics
    print("\n📊 Checking TCL 2.0 Metrics...")
    status = integrator.tcl_engine.get_session_status(session_id)
    metrics = status['metrics']
    print(f"   Hypergraph Complexity: {metrics['hypergraph_complexity']:.4f}")
    print(f"   Conceptual Density: {metrics['conceptual_density']:.4f}")
    print(f"   Abstract Reasoning Score: {metrics['abstract_reasoning_score']:.4f}")
    
    # 7. Quantum Integration Report
    print("\n🔬 Generating Quantum Integration Report...")
    report = integrator.generate_summary_report()
    print(f"   Systemic Quantum Coherence: {report['systemic_quantum_coherence']:.4f}")
    
    print("\n✅ Demonstration Complete.")

if __name__ == "__main__":
    run_demonstration()
