#!/usr/bin/env python3
"""
JARVIS-2v Complete Verification Checklist
Tests all implemented features: RAG, Adapters, API, and Ollama integration
"""

import sys
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:3001"
TEST_TIMEOUT = 30

class Jarvis2vVerifier:
    """Complete verification suite for JARVIS-2v"""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"    {message}")
        
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
    
    def test_knowledge_base_ingestion(self) -> bool:
        """Test KB file ingestion"""
        try:
            # Create test file
            test_file = Path("test_verification.txt")
            test_file.write_text("JARVIS-2v is an advanced AI assistant trained on custom documents.")
            
            # Test ingestion
            response = requests.post(
                f"{API_BASE_URL}/kb/ingest",
                json={"file_path": str(test_file)},
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "chunk_ids" in data and len(data["chunk_ids"]) > 0:
                    self.log_result(
                        "KB Ingestion", 
                        True, 
                        f"Ingested {len(data['chunk_ids'])} chunks"
                    )
                    # Cleanup
                    test_file.unlink()
                    return True
            
            self.log_result("KB Ingestion", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("KB Ingestion", False, str(e))
            return False
    
    def test_knowledge_base_search(self) -> bool:
        """Test KB search functionality"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/kb/search",
                json={"query": "JARVIS capabilities", "top_k": 3},
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    self.log_result(
                        "KB Search", 
                        True, 
                        f"Found {len(data['results'])} results"
                    )
                    return True
            
            self.log_result("KB Search", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("KB Search", False, str(e))
            return False
    
    def test_knowledge_base_stats(self) -> bool:
        """Test KB statistics"""
        try:
            response = requests.get(f"{API_BASE_URL}/kb/stats", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                stats = response.json()
                required_fields = ["total_chunks", "file_types", "vector_dim"]
                if all(field in stats for field in required_fields):
                    self.log_result(
                        "KB Statistics", 
                        True, 
                        f"Total chunks: {stats['total_chunks']}, Vector dim: {stats['vector_dim']}"
                    )
                    return True
            
            self.log_result("KB Statistics", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("KB Statistics", False, str(e))
            return False
    
    def test_adapter_creation(self) -> bool:
        """Test adapter creation"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/adapters",
                json={
                    "task_tags": ["test", "verification"],
                    "parameters": {"test": True}
                },
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "adapter_id" in data:
                    self.log_result(
                        "Adapter Creation", 
                        True, 
                        f"Created adapter: {data['adapter_id']}"
                    )
                    return True
            
            self.log_result("Adapter Creation", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("Adapter Creation", False, str(e))
            return False
    
    def test_adapter_listing(self) -> bool:
        """Test adapter listing"""
        try:
            response = requests.get(f"{API_BASE_URL}/adapters", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                if "adapters" in data and "total" in data:
                    self.log_result(
                        "Adapter Listing", 
                        True, 
                        f"Total adapters: {data['total']}"
                    )
                    return True
            
            self.log_result("Adapter Listing", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("Adapter Listing", False, str(e))
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test health endpoint"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            
            if response.status_code == 200:
                health = response.json()
                required_fields = ["status", "llm_ready", "version"]
                if all(field in health for field in required_fields):
                    self.log_result(
                        "Health Endpoint", 
                        True, 
                        f"Status: {health['status']}, Version: {health['version']}"
                    )
                    return True
            
            self.log_result("Health Endpoint", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("Health Endpoint", False, str(e))
            return False
    
    def test_chat_with_rag(self) -> bool:
        """Test chat endpoint with RAG integration"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "What is JARVIS and what are its capabilities?"}
                    ],
                    "options": {}
                },
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                required_fields = ["message", "usage", "adapters_used"]
                if all(field in chat_data for field in required_fields):
                    # Check if KB context was used
                    kb_used = chat_data.get("kb_context_used", False)
                    self.log_result(
                        "Chat with RAG", 
                        True, 
                        f"KB context used: {kb_used}, Adapters: {len(chat_data['adapters_used'])}"
                    )
                    return True
            
            self.log_result("Chat with RAG", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("Chat with RAG", False, str(e))
            return False
    
    def test_quantum_experiments(self) -> bool:
        """Test quantum experiment functionality"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/quantum/experiment",
                json={
                    "experiment_type": "interference_experiment",
                    "config": {"iterations": 100, "noise_level": 0.1}
                },
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                exp_data = response.json()
                required_fields = ["artifact_id", "experiment_type"]
                if all(field in exp_data for field in required_fields):
                    self.log_result(
                        "Quantum Experiments", 
                        True, 
                        f"Created artifact: {exp_data['artifact_id']}"
                    )
                    return True
            
            self.log_result("Quantum Experiments", False, f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_result("Quantum Experiments", False, str(e))
            return False
    
    def test_file_structure(self) -> bool:
        """Verify required files and directories exist"""
        required_files = [
            "src/core/knowledge_base.py",
            "src/core/adapter_engine.py", 
            "src/api/main.py",
            "scripts/train_adapters.py",
            "config.yaml",
            ".env.example",
            "docs/TRAINING_MY_JARVIS.md",
            "docs/OLLAMA.md",
            "docs/DEPLOYMENT.md",
            "ollama/Modelfile",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            self.log_result("File Structure", True, f"All {len(required_files)} required files present")
            return True
        else:
            self.log_result("File Structure", False, f"Missing: {', '.join(missing_files)}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration loading"""
        try:
            # Check if config.yaml exists and is valid
            config_path = Path("config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                required_sections = ["knowledge_base", "adapters", "api", "model"]
                if all(section in config for section in required_sections):
                    self.log_result("Configuration", True, "Valid YAML with required sections")
                    return True
            
            self.log_result("Configuration", False, "Invalid or missing config.yaml")
            return False
            
        except Exception as e:
            self.log_result("Configuration", False, str(e))
            return False
    
    def test_training_script(self) -> bool:
        """Test training script functionality"""
        try:
            # Check if training script exists and can be imported
            sys.path.insert(0, str(Path("scripts").parent))
            
            # This will test the import without actually running training
            from scripts.train_adapters import AdapterTrainer
            
            self.log_result("Training Script", True, "AdapterTrainer class importable")
            return True
            
        except Exception as e:
            self.log_result("Training Script", False, str(e))
            return False
    
    def test_ollama_modelfile(self) -> bool:
        """Test Ollama Modelfile"""
        try:
            modelfile_path = Path("ollama/Modelfile")
            if modelfile_path.exists():
                content = modelfile_path.read_text()
                
                required_elements = ["FROM", "SYSTEM", "PARAMETER"]
                if all(element in content for element in required_elements):
                    self.log_result("Ollama Modelfile", True, "Valid Modelfile with required elements")
                    return True
            
            self.log_result("Ollama Modelfile", False, "Invalid or missing Modelfile")
            return False
            
        except Exception as e:
            self.log_result("Ollama Modelfile", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run complete verification suite"""
        print("🔍 JARVIS-2v Complete Verification Checklist")
        print("=" * 60)
        
        # Test file structure and configuration first
        print("\n📁 Testing File Structure and Configuration:")
        self.test_file_structure()
        self.test_configuration()
        self.test_training_script()
        self.test_ollama_modelfile()
        
        # Test API endpoints
        print("\n🔗 Testing API Endpoints:")
        self.test_health_endpoint()
        
        # Only test API endpoints if server is running
        try:
            requests.get(f"{API_BASE_URL}/health", timeout=5)
            print("\n🧠 Testing Knowledge Base:")
            self.test_knowledge_base_ingestion()
            self.test_knowledge_base_search()
            self.test_knowledge_base_stats()
            
            print("\n🔌 Testing Adapter Engine:")
            self.test_adapter_creation()
            self.test_adapter_listing()
            
            print("\n💬 Testing Chat with RAG:")
            self.test_chat_with_rag()
            
            print("\n⚛️ Testing Quantum Module:")
            self.test_quantum_experiments()
            
        except requests.exceptions.RequestException:
            self.log_result("API Server", False, "Server not running on localhost:3001")
            print("\n⚠️  API server not running. Start it with: python -m src.api.main")
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.passed_tests}/{self.total_tests} passed")
        
        if self.passed_tests == self.total_tests:
            print("🎉 All tests passed! JARVIS-2v is ready for deployment.")
        else:
            failed_tests = [r["test"] for r in self.results if not r["passed"]]
            print(f"❌ Failed tests: {', '.join(failed_tests)}")
        
        # Provide next steps
        print("\n🚀 Next Steps:")
        if self.passed_tests >= self.total_tests - 2:  # Allow for server not running
            print("1. Start JARVIS-2v: python -m src.api.main")
            print("2. Train your AI: python scripts/train_adapters.py --input ./training-data")
            print("3. Create Ollama model: ollama create jarvis2v -f ollama/Modelfile")
            print("4. Deploy using Docker: docker-compose up -d")
        else:
            print("1. Fix failed tests first")
            print("2. Ensure all dependencies are installed")
            print("3. Check file permissions and paths")
        
        return self.passed_tests == self.total_tests

def main():
    """Main verification function"""
    verifier = Jarvis2vVerifier()
    
    # Check if running from correct directory
    if not Path("src").exists():
        print("❌ Error: Run this script from the JARVIS-2v root directory")
        print("   Expected to see 'src' directory in current path")
        sys.exit(1)
    
    # Run verification
    success = verifier.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()