import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.pipeline import IngestionPipeline
from src.thought_compression.tcl_engine import ThoughtCompressionEngine

def run_full_scale():
    pipeline = IngestionPipeline()
    
    # Target: 100+ papers
    queries = [
        # ArXiv queries
        {"query": "cat:cs.AI", "source": "arxiv", "max": 25},
        {"query": "cat:quant-ph", "source": "arxiv", "max": 25},
        {"query": "cat:cs.LG", "source": "arxiv", "max": 25},
        {"query": "cat:cs.NE", "source": "arxiv", "max": 25},
        # PubMed queries
        {"query": "Quantum Biology", "source": "pubmed", "max": 20},
        {"query": "Cancer Genomics KRAS", "source": "pubmed", "max": 20},
        {"query": "Artificial General Intelligence", "source": "pubmed", "max": 10},
    ]
    
    all_created = []
    
    print("--- PHASE 1: MASSIVE INGESTION ---")
    for q in queries:
        try:
            created = pipeline.run(q["query"], source=q["source"], max_results=q["max"])
            all_created.extend(created)
        except Exception as e:
            print(f"Error with query {q['query']}: {e}")
            
    print(f"\nTotal adapters created: {len(all_created)}")
    
    print("\n--- PHASE 2: STRUCTURAL ANALOGY DISCOVERY ---")
    # Use the session from the last run to find analogies
    # The pipeline uses a new session for every run, but they all share the global symbols
    # So we can just create a new session and use it to analyze the global state
    session_id = pipeline.tcl_engine.create_session("analogy_discovery")
    
    # We'll pick some key biological and quantum concepts and look for analogies
    targets = ["Cancer", "Quantum_Coherence", "DNA_Tunneling", "Neural_Network", "Backpropagation"]
    
    for target in targets:
        result = pipeline.tcl_engine.enhance_reasoning(session_id, f"Find analogies for {target}")
        if result.get("analogies"):
            print(f"\nAnalogies for {target}:")
            for analogy in result["analogies"]:
                print(f"  - {analogy['description']}")
    
    print("\n--- PHASE 3: METRICS VERIFICATION ---")
    pipeline.tcl_engine.refresh_metrics(session_id)
    status = pipeline.tcl_engine.get_session_status(session_id)
    print(f"Final Cognitive Metrics:")
    print(f"  Hypergraph Complexity: {status['metrics']['hypergraph_complexity']:.4f}")
    print(f"  Conceptual Density: {status['metrics']['conceptual_density']:.4f}")
    print(f"  Abstract Reasoning Score: {status['metrics']['abstract_reasoning_score']:.4f}")
    print(f"  Total Symbols: {status['symbol_count']}")
    
    # Save the adapter_graph.json explicitly if needed
    print("\nSaving final state...")
    if hasattr(pipeline.adapter_engine.adapter_graph, "_save_graph"):
        pipeline.adapter_engine.adapter_graph._save_graph()
    
    print("\n✅ Full-scale ingestion and discovery complete.")

if __name__ == "__main__":
    run_full_scale()
