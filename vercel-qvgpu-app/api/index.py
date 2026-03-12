
import os
import sys
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# Add project root to path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_llm.quantum_transformer import QuantumTransformer, SimpleTokenizer

app = FastAPI()

# Load model
MODEL_PATH = Path(__file__).parent.parent / "jarvis_qvgpu_trained.npz"
model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None:
        model = QuantumTransformer.load(str(MODEL_PATH))
        # Create a basic tokenizer (in production, load the saved one)
        tokenizer = SimpleTokenizer(vocab_size=5000)
    return model, tokenizer

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20

@app.get("/api/health")
def health():
    return {"status": "Quantum Swarm Engine Online", "model": "Jarvis-QvGPU-12M"}

@app.post("/api/generate")
def generate(request: GenerateRequest):
    m, t = get_model()
    # Simplified generation for API
    input_ids = t.encode(request.prompt)
    input_ids = np.array(input_ids).reshape(1, -1)
    
    generated = input_ids[0].tolist()
    for _ in range(request.max_tokens):
        context = generated[-64:]
        logits, _ = m.forward(np.array(context).reshape(1, -1))
        next_token = np.argmax(logits[0, -1, :])
        generated.append(int(next_token))
        if next_token == 1: # EOS
            break
            
    response = t.decode(generated)
    return {"generated_text": response}

@app.get("/api/summary")
def summary():
    summary_path = Path(__file__).parent.parent / "training_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path, "r") as f:
            return json.load(f)
    return {"error": "Summary not found"}
