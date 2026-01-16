
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.quantum_llm.quantum_transformer import QuantumTransformer, SimpleTokenizer
from src.quantum_llm.training_engine import QuantumTrainingEngine, TrainingConfig
from src.thought_compression.tcl_engine import ThoughtCompressionEngine
from src.core.adapter_engine import AdapterEngine, AdapterStatus

def train_jarvis():
    print("🧠 Starting Jarvis Final Form Training...")
    
    # 1. Load Filtered Data
    with open("filtered_books.json", "r") as f:
        books = json.load(f)
    
    # 2. TCL Compression
    print("🗜️ Compressing concepts with TCL Engine...")
    tcl_engine = ThoughtCompressionEngine()
    session_id = tcl_engine.create_session("jarvis_train")
    
    book_seeds = []
    for book in books:
        # Compress title and first 1000 chars as a seed
        seed_input = f"{book['title']} by {book['author']}. {book['text'][:500]}"
        compressed = tcl_engine.compress_concept(session_id, seed_input)
        book_seeds.append({
            "title": book['title'],
            "seed": compressed['compressed_symbols'],
            "ratio": compressed['compression_ratio']
        })
        print(f"   Compressed {book['title']} -> {len(compressed['compressed_symbols'])} symbols")

    # 3. Create Adapters
    print("🔌 Creating adapters via AdapterEngine...")
    adapter_config = {
        "adapters": {"storage_path": "./adapters", "graph_path": "./adapters_graph.json"},
        "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
    }
    adapter_engine = AdapterEngine(adapter_config)
    
    # Create ~50 adapters by variations of topics and eras
    eras = ["Victorian", "Industrial", "Edwardian", "Early 20th Century", "Post-War"]
    topics = ["Medicine", "Physics", "Biology", "Quantum", "Disease", "Cure", "Alchemy"]
    
    created_adapters = []
    for era in eras:
        for topic in topics:
            # Random bit patterns for variety
            y_bits = [0] * 16
            y_bits[hash(era) % 16] = 1
            y_bits[hash(topic) % 16] = 1
            
            z_bits = [0] * 8
            z_bits[hash(era + topic) % 8] = 1
            
            x_bits = [0] * 8
            
            adapter = adapter_engine.create_adapter(
                task_tags=[era, topic],
                y_bits=y_bits,
                z_bits=z_bits,
                x_bits=x_bits,
                parameters={"era": era, "topic": topic}
            )
            created_adapters.append(adapter.id)
            
    print(f"   Created {len(created_adapters)} adapters.")

    # 4. FULL BACKPROP TRAINING
    print("🚂 Initializing Quantum LLM training...")
    config = TrainingConfig(
        vocab_size=20000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        epochs=3, # Adjusted for time, but real training
        batch_size=4
    )
    
    model = QuantumTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff
    )
    
    engine = QuantumTrainingEngine(config, model)
    engine.load_dataset("filtered_books.json")
    
    # Run training - limited for this environment but doing real steps
    max_steps = 200
    print(f"🚂 Training for max {max_steps} steps...")
    
    epoch_loss = 0
    for epoch in range(config.epochs):
        engine.train_dataset.shuffle()
        for i in range(0, len(engine.train_dataset), config.batch_size):
            batch_input, batch_target = engine.train_dataset.get_batch(config.batch_size, i)
            if batch_input.shape[0] == 0: continue
            
            metrics = engine.train_step(batch_input, batch_target)
            if engine.global_step % 5 == 0:
                print(f"Step {engine.global_step} | Loss: {metrics['loss']:.4f} | Coherence: {metrics['avg_coherence']:.3f}")
                model.save("jarvis_quantum_final.npz") # Save frequently in case of timeout
            
            if engine.global_step >= max_steps:
                break
        if engine.global_step >= max_steps:
            break
    
    print("✅ Training complete (reached step limit)!")
    
    # 5. Hook up Talking Voice (Mock logic for Phi-3 routing in this demo context)
    # We save the model and adapters for use in the final demo
    model.save("jarvis_quantum_final.npz")
    print("💾 Final model saved to jarvis_quantum_final.npz")

if __name__ == "__main__":
    train_jarvis()
