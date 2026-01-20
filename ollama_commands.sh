# JARVIS Ollama Deployment Commands

# Step 1: Create model in Ollama
ollama create jarvis -f Modelfile.jarvis

# Step 2: Verify model creation
ollama list | grep jarvis

# Step 3: Test basic functionality
ollama run jarvis "Hello JARVIS, are you ready?"

# Step 4: Test quantum knowledge
ollama run jarvis "Explain quantum entanglement and its implications"

# Step 5: Test scientific reasoning
ollama run jarvis "What is the significance of Bell's theorem?"

# Step 6: Test historical context
ollama run jarvis "Describe the development of quantum mechanics in the 1920s"

# Step 7: Test AI-quantum synergy
ollama run jarvis "How can quantum principles enhance artificial intelligence?"

# Advanced: Run with specific parameters
ollama run jarvis   --temperature 0.8   --top-p 0.95   "Generate a creative explanation of quantum superposition"

# Batch test
for query in   "Define quantum coherence"   "Explain the uncertainty principle"   "What is quantum teleportation?"   "How does quantum computing work?"   "What is wave-particle duality?"
do
  echo "=== Testing: $query ==="
  ollama run jarvis "$query"
  echo ""
done
