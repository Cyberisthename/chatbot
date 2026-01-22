#!/usr/bin/env python3
"""
Jarvis Quantum LLM - Ollama Test Suite
Test the model after deployment to Ollama
"""

import json
import time
import sys
from typing import Dict, List

try:
    import requests
except ImportError:
    print("Installing requests...")
    import os
    os.system("pip install requests")
    import requests


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")


def test_ollama_running() -> bool:
    """Test if Ollama is running"""
    print_info("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print_success("Ollama is running")
            return True
        else:
            print_error("Ollama returned unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Ollama is not running!")
        print_info("Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print_error(f"Error connecting to Ollama: {e}")
        return False


def test_model_exists() -> bool:
    """Test if Jarvis model exists in Ollama"""
    print_info("Checking if Jarvis model exists...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if "jarvis:latest" in models or "jarvis" in models:
                print_success("Jarvis model found in Ollama")
                return True
            else:
                print_error("Jarvis model not found!")
                print_info("Available models: " + ", ".join(models))
                print_info("Create the model with: ollama create jarvis -f Modelfile")
                return False
        return False
    except Exception as e:
        print_error(f"Error checking models: {e}")
        return False


def test_generation(prompt: str) -> Dict:
    """Test text generation with a prompt"""
    print_info(f"Testing generation with prompt: '{prompt[:50]}...'")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "jarvis",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get("response", "")
            
            print_success(f"Generation complete in {duration:.2f}s")
            print(f"{Colors.BOLD}Response:{Colors.END}")
            print(f"{Colors.CYAN}{generated_text[:200]}{Colors.END}")
            if len(generated_text) > 200:
                print(f"{Colors.CYAN}... (truncated){Colors.END}")
            
            return {
                "success": True,
                "response": generated_text,
                "duration": duration,
                "tokens": len(generated_text.split())
            }
        else:
            print_error(f"Generation failed: {response.status_code}")
            return {"success": False, "error": response.text}
            
    except Exception as e:
        print_error(f"Error during generation: {e}")
        return {"success": False, "error": str(e)}


def test_streaming():
    """Test streaming generation"""
    print_info("Testing streaming generation...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "jarvis",
                "prompt": "Explain quantum mechanics in one sentence.",
                "stream": True
            },
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            tokens = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    tokens.append(token)
                    if data.get("done", False):
                        break
            
            full_response = "".join(tokens)
            print_success("Streaming generation works!")
            print(f"{Colors.CYAN}Response: {full_response}{Colors.END}")
            return True
        else:
            print_error(f"Streaming failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error during streaming: {e}")
        return False


def run_test_suite():
    """Run complete test suite"""
    print_header("🧪 JARVIS QUANTUM LLM - Ollama Test Suite")
    
    results = {
        "ollama_running": False,
        "model_exists": False,
        "generation_works": False,
        "streaming_works": False
    }
    
    # Test 1: Ollama running
    print_header("Test 1: Ollama Connection")
    results["ollama_running"] = test_ollama_running()
    
    if not results["ollama_running"]:
        print_error("\nCannot proceed - Ollama is not running!")
        return results
    
    # Test 2: Model exists
    print_header("Test 2: Model Availability")
    results["model_exists"] = test_model_exists()
    
    if not results["model_exists"]:
        print_error("\nCannot proceed - Jarvis model not found!")
        return results
    
    # Test 3: Generation
    print_header("Test 3: Text Generation")
    
    test_prompts = [
        "What is quantum mechanics?",
        "Explain neural networks",
        "Tell me about DNA"
    ]
    
    generation_results = []
    for prompt in test_prompts:
        result = test_generation(prompt)
        generation_results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    results["generation_works"] = all(r.get("success", False) for r in generation_results)
    
    if results["generation_works"]:
        avg_duration = sum(r["duration"] for r in generation_results) / len(generation_results)
        avg_tokens = sum(r["tokens"] for r in generation_results) / len(generation_results)
        print_success(f"All generation tests passed!")
        print_info(f"Average response time: {avg_duration:.2f}s")
        print_info(f"Average tokens: {avg_tokens:.1f}")
    
    # Test 4: Streaming
    print_header("Test 4: Streaming Generation")
    results["streaming_works"] = test_streaming()
    
    # Summary
    print_header("📊 Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.END}")
    
    if passed_tests == total_tests:
        print_success("\n🎉 All tests passed! Jarvis is ready to use!")
        print_info("\nTry it: ollama run jarvis")
    else:
        print_error("\n⚠️  Some tests failed. Check the output above.")
    
    return results


def interactive_test():
    """Interactive testing mode"""
    print_header("💬 Interactive Test Mode")
    print_info("Enter prompts to test Jarvis. Type 'exit' to quit.\n")
    
    while True:
        try:
            prompt = input(f"{Colors.BOLD}You:{Colors.END} ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                print_info("Goodbye!")
                break
            
            print()
            result = test_generation(prompt)
            print()
            
            if not result.get("success", False):
                print_error("Generation failed!")
                
        except KeyboardInterrupt:
            print("\n")
            print_info("Goodbye!")
            break
        except Exception as e:
            print_error(f"Error: {e}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        results = run_test_suite()
        
        # Exit code based on results
        if all(results.values()):
            return 0
        else:
            return 1


if __name__ == "__main__":
    exit(main())
