#!/usr/bin/env python3
"""
Test script for JARVIS-2v Knowledge Base System
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.knowledge_base import KnowledgeBase

def test_knowledge_base():
    """Test knowledge base functionality"""
    print("🧪 Testing JARVIS-2v Knowledge Base System")
    print("=" * 50)
    
    # Initialize knowledge base
    kb_config = {
        "data_path": "./data",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_dim": 384
    }
    
    print("📚 Initializing Knowledge Base...")
    kb = KnowledgeBase(kb_config)
    
    # Test file ingestion
    print("\n📖 Testing file ingestion...")
    
    # Ingest training data
    try:
        chunk_ids = kb.ingest_directory("./training-data")
        print(f"✓ Ingested {len(chunk_ids)} chunks from training-data")
    except Exception as e:
        print(f"✗ Error ingesting training data: {e}")
        return False
    
    # Ingest raw data
    try:
        chunk_ids = kb.ingest_directory("./data/raw")
        print(f"✓ Ingested {len(chunk_ids)} chunks from data/raw")
    except Exception as e:
        print(f"✗ Error ingesting raw data: {e}")
        return False
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    
    test_queries = [
        "programming functions python",
        "mathematics algebra equations",
        "object oriented programming",
        "geometry area volume"
    ]
    
    for query in test_queries:
        try:
            results = kb.search(query, top_k=3)
            print(f"✓ Query '{query}': Found {len(results)} results")
            
            # Show top result
            if results:
                top_chunk, score = results[0]
                print(f"  - Top result: {top_chunk.content[:100]}... (score: {score:.3f})")
        except Exception as e:
            print(f"✗ Search failed for query '{query}': {e}")
            return False
    
    # Test context generation
    print("\n📄 Testing context generation...")
    
    try:
        context = kb.get_context_for_query("Explain functions in programming")
        print(f"✓ Generated context ({len(context)} characters)")
        print(f"Context preview: {context[:200]}...")
    except Exception as e:
        print(f"✗ Context generation failed: {e}")
        return False
    
    # Get statistics
    print("\n📊 Knowledge Base Statistics:")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Knowledge Base test completed successfully!")
    return True

if __name__ == "__main__":
    test_knowledge_base()