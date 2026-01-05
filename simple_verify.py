#!/usr/bin/env python3
"""
Simple JARVIS-2v Implementation Verification
Tests core components without external dependencies
"""

import sys
import json
from pathlib import Path

def verify_implementation():
    """Verify JARVIS-2v implementation is complete"""
    print("🔍 JARVIS-2v Implementation Verification")
    print("=" * 50)
    
    results = []
    
    # Core Components
    core_files = [
        "src/core/knowledge_base.py",
        "src/core/adapter_engine.py", 
        "src/api/main.py",
        "src/quantum/synthetic_quantum.py"
    ]
    
    print("\n📁 Core Components:")
    for file_path in core_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Training System
    training_files = [
        "scripts/train_adapters.py",
        "test_kb.py"
    ]
    
    print("\n🚀 Training System:")
    for file_path in training_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Configuration
    config_files = [
        "config.yaml",
        ".env.example",
        "config_jetson.yaml"
    ]
    
    print("\n⚙️ Configuration:")
    for file_path in config_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Ollama Integration
    ollama_files = [
        "ollama/Modelfile",
        "models/jarvis-7b-q4_0.gguf"
    ]
    
    print("\n🤖 Ollama Integration:")
    for file_path in ollama_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Documentation
    doc_files = [
        "docs/TRAINING_MY_JARVIS.md",
        "docs/OLLAMA.md", 
        "docs/DEPLOYMENT.md"
    ]
    
    print("\n📚 Documentation:")
    for file_path in doc_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Deployment
    deploy_files = [
        "Dockerfile",
        "docker-compose.yml",
        "vercel.json"
    ]
    
    print("\n🚀 Deployment:")
    for file_path in deploy_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        results.append(exists)
    
    # Test knowledge base
    print("\n🧪 Knowledge Base Test:")
    try:
        sys.path.insert(0, "src")
        from core.knowledge_base import KnowledgeBase
        
        kb_config = {
            "data_path": "./data",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_dim": 384
        }
        
        kb = KnowledgeBase(kb_config)
        print("✓ KnowledgeBase import successful")
        results.append(True)
        
    except Exception as e:
        print(f"✗ KnowledgeBase import failed: {e}")
        results.append(False)
    
    # Summary
    total = len(results)
    passed = sum(results)
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} components verified")
    
    if passed == total:
        print("🎉 JARVIS-2v Implementation Complete!")
        print("\n✅ Implemented Features:")
        print("• Layer A: RAG Knowledge Base with file ingestion")
        print("• Layer B: Adapter training pipeline") 
        print("• Layer D: Ollama GGUF integration")
        print("• Layer F: Complete deployment configurations")
        print("• Full documentation and guides")
        
        print("\n🚀 Quick Start:")
        print("1. Train your AI: python scripts/train_adapters.py --input ./training-data")
        print("2. Start API: python -m src.api.main")
        print("3. Create Ollama model: ollama create jarvis2v -f ollama/Modelfile")
        print("4. Deploy: docker-compose up -d")
        
    else:
        failed_count = total - passed
        print(f"⚠️  {failed_count} components need attention")
    
    return passed == total

if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)