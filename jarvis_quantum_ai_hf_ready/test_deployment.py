#!/usr/bin/env python3
"""
🚀 JARVIS Quantum AI Suite - Deployment Verification
=====================================================

Quick test to ensure all components work before Hugging Face deployment.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    # Test src modules
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test quantum LLM
        from quantum_llm import quantum_transformer
        print("✅ Quantum LLM: OK")
        
        # Test thought compression
        from thought_compression import tcl_engine
        print("✅ Thought Compression: OK")
        
        # Test core engines
        from core import adapter_engine
        print("✅ Core Engines: OK")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_demo_modules():
    """Test demo modules"""
    print("\n🎭 Testing demo modules...")
    
    try:
        # Test cancer demo
        import gradio_quantum_cancer_demo
        print("✅ Cancer Demo: OK")
        
        # Test Jarvis demo
        import jarvis_v1_gradio_space
        print("✅ Jarvis Demo: OK")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False
    
    return True

def test_main_app():
    """Test main app"""
    print("\n🚀 Testing main app...")
    
    try:
        # This will test if the main app can import everything
        import app
        print("✅ Main App: OK")
        
        # Test interface creation
        demo = app.create_interface()
        print("✅ Interface Creation: OK")
        
    except Exception as e:
        print(f"❌ App error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🌌 JARVIS Quantum AI Suite - Deployment Verification")
    print("=" * 55)
    
    tests = [
        test_imports,
        test_demo_modules, 
        test_main_app
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Ready for Hugging Face deployment!")
        return True
    else:
        print("❌ Some tests failed. Please fix before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)