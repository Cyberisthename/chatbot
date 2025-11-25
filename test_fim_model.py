#!/usr/bin/env python3
"""
Test the trained FIM model with sample completions.
"""
import json
import subprocess
import sys
from pathlib import Path


def test_ollama_available():
    """Check if Ollama is available."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def test_model_installed(model_name='fim-code-completion'):
    """Check if FIM model is installed in Ollama."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return model_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def test_fim_completion(prefix, suffix, model_name='fim-code-completion'):
    """Test FIM completion with given prefix and suffix."""
    fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name, fim_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {e}"


def run_tests():
    """Run test suite for FIM model."""
    print("=" * 70)
    print("🧪 FIM Model Test Suite")
    print("=" * 70)
    print()
    
    # Test 1: Check Ollama availability
    print("Test 1: Checking Ollama availability...")
    if test_ollama_available():
        print("✅ Ollama is available\n")
    else:
        print("❌ Ollama is not available")
        print("Please install Ollama from https://ollama.ai\n")
        return False
    
    # Test 2: Check model installation
    print("Test 2: Checking FIM model installation...")
    if test_model_installed():
        print("✅ FIM model is installed\n")
    else:
        print("❌ FIM model is not installed")
        print("Please run: ollama create fim-code-completion -f Modelfile.fim\n")
        return False
    
    # Test 3: Python function completion
    print("Test 3: Python function completion...")
    prefix = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    """
    suffix = """
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
    
    print(f"Prefix: {prefix[:50]}...")
    print(f"Suffix: {suffix[:50]}...")
    
    completion = test_fim_completion(prefix, suffix)
    print(f"Completion: {completion}\n")
    
    # Test 4: JavaScript async function
    print("Test 4: JavaScript async function...")
    prefix = """async function fetchData(url) {
    try {
        """
    suffix = """
        return await response.json();
    } catch (error) {
        console.error('Fetch failed:', error);
        throw error;
    }
}"""
    
    completion = test_fim_completion(prefix, suffix)
    print(f"Completion: {completion}\n")
    
    # Test 5: Class method completion
    print("Test 5: Python class method...")
    prefix = """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """
    suffix = """
    
    result = []
    result.extend(inorder_traversal(root.left))
    result.append(root.val)
    result.extend(inorder_traversal(root.right))
    
    return result"""
    
    completion = test_fim_completion(prefix, suffix)
    print(f"Completion: {completion}\n")
    
    print("=" * 70)
    print("✅ Test suite complete!")
    print("=" * 70)
    
    return True


def interactive_test():
    """Interactive FIM testing."""
    print("\n" + "=" * 70)
    print("🎯 Interactive FIM Testing")
    print("=" * 70)
    print("Enter prefix and suffix to test completions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            print("\nPrefix (code before cursor):")
            prefix = input("> ").strip()
            
            if prefix.lower() == 'quit':
                break
            
            print("\nSuffix (code after cursor):")
            suffix = input("> ").strip()
            
            if suffix.lower() == 'quit':
                break
            
            print("\n🤖 Generating completion...")
            completion = test_fim_completion(prefix, suffix)
            
            print("\n" + "=" * 70)
            print("Completion:")
            print("=" * 70)
            print(completion)
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main test runner."""
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_test()
    else:
        success = run_tests()
        
        if success:
            print("\n💡 Want to try interactive testing?")
            print("Run: python test_fim_model.py --interactive\n")


if __name__ == "__main__":
    main()
