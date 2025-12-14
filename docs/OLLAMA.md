# JARVIS-2v + Ollama Integration Guide

## Overview

This guide explains how to integrate JARVIS-2v with Ollama for efficient local inference, including GGUF model management and verification.

## Prerequisites

### 1. Install Ollama

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai)

### 2. Verify Ollama Installation

```bash
ollama --version
ollama serve
```

## Model Setup

### Option 1: Use JARVIS-2v Custom Model

1. **Create JARVIS-2v Ollama Model**
```bash
# From project root directory
ollama create jarvis2v -f ollama/Modelfile
```

2. **Test the Model**
```bash
ollama run jarvis2v
```

3. **Verify Integration**
```bash
# Test with knowledge base query
ollama run jarvis2v "What do you know about programming functions?"
```

### Option 2: Use Pre-trained GGUF Models

1. **Download Compatible Model**
```bash
# Download a 7B model compatible with JARVIS-2v
ollama pull llama2:7b-chat-q4_0.gguf

# Or use other compatible models
ollama pull mistral:7b-instruct-q4_0.gguf
```

2. **Create Custom Modelfile**
```bash
cat > ollama/custom_jarvis.Modelfile << 'EOF'
FROM ./models/your-model.gguf

SYSTEM """You are J.A.R.V.I.S., an advanced AI assistant trained on custom documents and specialized adapters. You have access to:
- Knowledge Base: Information from user documents and training data
- Adapter Engine: Specialized modules for routing and domain-specific tasks
- Y/Z/X Bit System: Task classification and routing capabilities

When answering questions, use relevant information from your knowledge base and appropriate adapters. Be precise, helpful, and reference specific details from your training when relevant.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
EOF
```

3. **Create Custom Model**
```bash
ollama create jarvis-custom -f ollama/custom_jarvis.Modelfile
```

## Verification Checklist

### 1. Model Functionality Test

```bash
# Basic capability test
ollama run jarvis2v "Hello, can you introduce yourself?"

# Expected: Model should identify as JARVIS with access to knowledge base and adapters
```

### 2. Knowledge Base Integration Test

**Setup test environment:**
```bash
# Start JARVIS-2v API
python -m src.api.main

# In separate terminal, test knowledge base
curl -X POST "http://localhost:3001/kb/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./training-data/custom_jarvis_data.txt"}'

# Test query that should use knowledge base
curl -X POST "http://localhost:3001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "What is your purpose?"}],
       "options": {}
     }'
```

**Expected results:**
- Model should reference information from training data
- Chat response should include `kb_context_used: true`
- Response should reference specific details from knowledge base

### 3. Adapter Routing Test

```bash
# Test domain-specific routing
curl -X POST "http://localhost:3001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Explain how functions work in programming"}],
       "options": {}
     }'

# Check response includes adapter information
# adapters_used should contain relevant adapter IDs
```

### 4. Prompt-Based Verification Tests

Run these specific prompts to verify training:

**Test 1: Identity and Purpose**
```
Prompt: "Who are you and what is your purpose?"
Expected: Should reference Ben's creation and custom training
```

**Test 2: Domain Knowledge**
```
Prompt: "Explain object-oriented programming with examples"
Expected: Should use programming domain adapters and knowledge base
```

**Test 3: Memory and Context**
```
Prompt: "What capabilities did you mention in our conversation?"
Expected: Should reference previous interactions and training data
```

**Test 4: Technical Specifications**
```
Prompt: "How do you handle different types of queries?"
Expected: Should explain Y/Z/X bit routing and adapter system
```

## Integration Architecture

### Local Integration (Recommended)

```python
# Update inference.py to use Ollama
class JarvisInferenceBackend:
    def __init__(self, model_path=None, config=None):
        # Use Ollama instead of llama.cpp
        self.ollama_client = OllamaClient()
    
    def chat(self, messages, **options):
        # Format for Ollama API
        response = self.ollama_client.generate(
            model="jarvis2v",
            messages=messages,
            **options
        )
        return response
```

### API Integration

```python
# Connect to Ollama server from JARVIS API
import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model, messages, **options):
        response = requests.post(f"{self.base_url}/api/chat", json={
            "model": model,
            "messages": messages,
            "options": options
        })
        return response.json()
```

## Performance Optimization

### 1. Model Selection

**For Desktop:**
- `jarvis2v` (custom JARVIS-7B)
- Model size: ~4GB quantized
- Performance: Fast inference, good quality

**For Low-Power:**
- Smaller models (3B parameters)
- Use low_power profile
- Quantization: Q4_0 or Q5_0

**For High-End:**
- Larger models (13B+ parameters)
- Use standard profile
- Consider GPU acceleration

### 2. Resource Management

```bash
# Monitor Ollama resources
ollama ps

# Stop running models when not needed
ollama stop jarvis2v

# Clear GPU cache if using CUDA
ollama stop --all
```

### 3. Batch Processing

For knowledge base ingestion:
```bash
# Ingest multiple files efficiently
for file in training-data/*; do
  curl -X POST "http://localhost:3001/kb/ingest" \
       -H "Content-Type: application/json" \
       -d "{\"file_path\": \"$file\"}"
done
```

## Deployment with Ollama

### Local Development

```bash
# 1. Start Ollama
ollama serve &

# 2. Create JARVIS model
ollama create jarvis2v -f ollama/Modelfile

# 3. Start JARVIS API
python -m src.api.main

# 4. Start UI (if using web interface)
cd ui && npm run dev
```

### Docker Deployment

```dockerfile
# Dockerfile with Ollama
FROM python:3.11-slim

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy JARVIS files
COPY . /app
WORKDIR /app

# Create model
RUN ollama create jarvis2v -f ollama/Modelfile

# Expose ports
EXPOSE 3001 11434

# Start services
CMD ["sh", "-c", "ollama serve & python -m src.api.main"]
```

### Cloud Deployment

**Option 1: GPU Instance**
- AWS EC2 with GPU (g4dn.xlarge or better)
- GCP Compute Engine with Tesla T4
- Azure NC-series VMs

**Option 2: CPU-Optimized**
- AWS c5.4xlarge
- GCP c2-standard-8
- Use smaller quantized models

## Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Verify model exists
ollama list

# Recreate if needed
ollama create jarvis2v -f ollama/Modelfile
```

**"Out of memory"**
```bash
# Use smaller model or reduce context
ollama run jarvis2v --num_ctx 1024

# Or switch to smaller model
ollama pull llama2:3b-chat-q4_0.gguf
```

**"Slow inference"**
- Check CPU/GPU utilization
- Reduce model size
- Use batch processing for KB operations

**"Inconsistent responses"**
- Verify system prompt in Modelfile
- Check knowledge base has relevant data
- Review adapter routing logs

### Performance Monitoring

```bash
# Check Ollama system status
curl http://localhost:11434/api/tags

# Monitor resource usage
htop
nvidia-smi  # If using GPU
```

## Advanced Features

### 1. Multiple Model Support

```python
# Use different models for different tasks
class MultiModelManager:
    def __init__(self):
        self.models = {
            "chat": "jarvis2v",
            "code": "codellama:7b-instruct-q4_0",
            "reasoning": "llama2:13b-chat-q4_0"
        }
    
    def get_model_for_task(self, task_type):
        return self.models.get(task_type, "jarvis2v")
```

### 2. Streaming Responses

```python
# Enable streaming for real-time chat
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "jarvis2v",
        "messages": messages,
        "stream": True
    },
    stream=True
)

for chunk in response.iter_lines():
    if chunk:
        data = json.loads(chunk.decode())
        yield data["message"]["content"]
```

### 3. Custom Prompt Engineering

```bash
# Create specialized models for different domains
cat > ollama/programming.Modelfile << 'EOF'
FROM jarvis2v

SYSTEM """You are a programming specialist version of JARVIS. 
Focus on code examples, technical explanations, and debugging assistance.
Always include code snippets when relevant."""

PARAMETER temperature 0.3  # More deterministic for coding
PARAMETER top_p 0.8
EOF

ollama create jarvis-programming -f ollama/programming.Modelfile
```

## Security Considerations

### Local Deployment
- Ollama runs locally, no data sent to external servers
- Knowledge base stays on local machine
- Consider firewall rules for production

### API Security
```python
# Add authentication to JARVIS API
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    if token.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token
```

## Next Steps

1. **Performance Tuning**: Optimize based on your hardware and usage patterns
2. **Custom Models**: Train domain-specific models if needed
3. **Integration**: Connect with existing workflows and systems
4. **Monitoring**: Set up logging and performance metrics
5. **Scaling**: Consider distributed deployment for high availability

For production deployment details, see `docs/DEPLOYMENT.md`.