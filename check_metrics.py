import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.pipeline import IngestionPipeline

def check_metrics():
    pipeline = IngestionPipeline()
    session_id = pipeline.tcl_engine.create_session("check_metrics")
    
    # Manually refresh metrics
    pipeline.tcl_engine.refresh_metrics(session_id)
    
    status = pipeline.tcl_engine.get_session_status(session_id)
    print(f"Verified Cognitive Metrics:")
    print(f"  Hypergraph Complexity: {status['metrics']['hypergraph_complexity']:.4f}")
    print(f"  Conceptual Density: {status['metrics']['conceptual_density']:.4f}")
    print(f"  Abstract Reasoning Score: {status['metrics']['abstract_reasoning_score']:.4f}")
    print(f"  Total Symbols: {status['symbol_count']}")
    
    # Check if hyperedges exist
    num_hyper = len(pipeline.tcl_engine.global_symbols.hyperedges)
    print(f"  Total HyperEdges: {num_hyper}")

if __name__ == "__main__":
    check_metrics()
