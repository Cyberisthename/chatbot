import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.pipeline import IngestionPipeline

def bulk_ingest():
    """
    Run bulk ingestion for high-priority scientific domains
    """
    pipeline = IngestionPipeline()
    
    topics = [
        # Quantum Information & AI
        {"query": "cat:quant-ph AND cat:cs.AI", "source": "arxiv", "max": 3},
        {"query": "Topological Quantum Computing", "source": "arxiv", "max": 3},
        
        # Quantum Biology & Genomics
        {"query": "Quantum Biology", "source": "pubmed", "max": 3},
        {"query": "Quantum Genomics", "source": "pubmed", "max": 2},
        
        # Cancer Research & Theory
        {"query": "Theoretical Oncology", "source": "pubmed", "max": 3},
        {"query": "Quantum Effects in Biological Systems", "source": "arxiv", "max": 2}
    ]
    
    print("🌊 Starting Bulk Ingestion Process...")
    total_created = 0
    
    for topic in topics:
        try:
            created = pipeline.run(
                query=topic["query"],
                source=topic["source"],
                max_results=topic["max"]
            )
            total_created += len(created)
        except Exception as e:
            print(f"❌ Error during ingestion for topic '{topic['query']}': {e}")
            
    print(f"\n✅ Bulk Ingestion Complete!")
    print(f"🚀 Total new knowledge adapters created: {total_created}")

if __name__ == "__main__":
    bulk_ingest()
