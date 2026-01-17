"""
Quantum LLM Training Engine
Real training implementation from scratch with actual datasets and real backprop
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

import numpy as np

from .quantum_transformer import QuantumTransformer, SimpleTokenizer


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model architecture (scaled as requested)
    vocab_size: int = 20000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 0.0005
    epochs: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Data configuration
    dataset_path: str = "filtered_books.json"
    
    # Checkpointing
    checkpoint_interval: int = 50
    save_path: str = "./quantum_llm_checkpoints"
    
    # Logging
    log_interval: int = 5
    metrics_path: str = "./quantum_llm_metrics"


class Dataset:
    """
    Dataset handler for real training data
    """
    
    def __init__(self, texts: List[str], max_seq_len: int, tokenizer):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
        # Pre-compute tokenized sequences
        self.tokenized = []
        for text in texts:
            # Split long text into chunks
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens), max_seq_len):
                chunk = tokens[i:i + max_seq_len]
                if len(chunk) > 10:
                    self.tokenized.append(chunk)
        
        print(f"📚 Created dataset with {len(self.tokenized)} chunks")
    
    def get_batch(self, batch_size: int, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_input = []
        batch_target = []
        
        for i in range(index, min(index + batch_size, len(self.tokenized))):
            tokens = self.tokenized[i]
            
            # Pad if needed
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [0] * (self.max_seq_len - len(tokens))
            
            # Input: all tokens except last
            input_ids = tokens[:-1]
            # Target: all tokens except first
            target_ids = tokens[1:]
            
            batch_input.append(input_ids)
            batch_target.append(target_ids)
        
        return np.array(batch_input), np.array(batch_target)
    
    def __len__(self):
        return len(self.tokenized)

    def shuffle(self):
        random.shuffle(self.tokenized)


class QuantumTrainingEngine:
    """
    Training engine for Quantum LLM with real backpropagation
    """
    
    def __init__(self, config: TrainingConfig, model: QuantumTransformer):
        self.config = config
        self.model = model
        self.tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        self.train_dataset = None
        self.global_step = 0
        self.current_epoch = 0
        
        self.optimizer_state = self._initialize_optimizer_state()
        
        self.train_losses = []
        self.quantum_metrics_history = []
        
        Path(config.save_path).mkdir(parents=True, exist_ok=True)
        Path(config.metrics_path).mkdir(parents=True, exist_ok=True)
        
        print("🔧 Initialized QuantumTrainingEngine with Real Backprop")

    def _initialize_optimizer_state(self) -> Dict[str, Any]:
        state = {}
        for name, param in self._get_model_params().items():
            state[name] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
                "t": 0
            }
        return state
    
    def _get_model_params(self) -> Dict[str, np.ndarray]:
        params = {
            "embedding": self.model.embedding,
            "output_projection": self.model.output_projection,
        }
        for i, layer in enumerate(self.model.layers):
            params[f"layer_{i}_query"] = layer.query_proj
            params[f"layer_{i}_key"] = layer.key_proj
            params[f"layer_{i}_value"] = layer.value_proj
            params[f"layer_{i}_ffn1"] = layer.ffn1
            params[f"layer_{i}_ffn2"] = layer.ffn2
            params[f"layer_{i}_gamma1"] = layer.gamma1
            params[f"layer_{i}_beta1"] = layer.beta1
            params[f"layer_{i}_gamma2"] = layer.gamma2
            params[f"layer_{i}_beta2"] = layer.beta2
        return params

    def load_dataset(self, data_path: str):
        print(f"\n📚 Loading data from {data_path}...")
        
        # Check if dataset exists, if not - load from string
        if isinstance(data_path, str) and data_path.startswith("["):
            # Direct data passed as JSON string
            data = json.loads(data_path)
        elif Path(data_path).exists():
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        texts = [item["text"] for item in data]
        self.train_dataset = Dataset(texts, self.config.max_seq_len, self.tokenizer)
        print(f"✅ Loaded {len(self.train_dataset)} training chunks")

    def compute_loss(self, logits: np.ndarray, target_ids: np.ndarray) -> Tuple[float, np.ndarray]:
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target_ids.reshape(-1)
        
        # Softmax
        probs = self._softmax(logits_flat)
        
        # Loss
        target_probs = probs[np.arange(len(target_flat)), target_flat]
        loss = -np.log(target_probs + 1e-10)
        avg_loss = np.mean(loss)
        
        # Gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(len(target_flat)), target_flat] -= 1.0
        grad_logits = grad_logits / (batch_size * seq_len)
        grad_logits = grad_logits.reshape(batch_size, seq_len, vocab_size)
        
        return avg_loss, grad_logits

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def train_step(self, batch_input: np.ndarray, batch_target: np.ndarray) -> Dict[str, float]:
        # 1. Forward pass
        logits, quantum_metrics = self.model.forward(batch_input)
        
        # 2. Compute loss
        loss, grad_logits = self.compute_loss(logits, batch_target)
        
        # 3. Backward pass (REAL BACKPROP)
        gradients = self.model.backward(grad_logits)
        
        # 4. Optimizer step
        self.optimizer_step(gradients)
        
        self.global_step += 1
        return {"loss": loss, **quantum_metrics}

    def optimizer_step(self, gradients: Dict[str, np.ndarray]):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = self._get_learning_rate()
        
        params = self._get_model_params()
        
        for name, grad in gradients.items():
            if name not in self.optimizer_state:
                continue
            
            param = params[name]
            state = self.optimizer_state[name]
            
            # Clip grad
            if self.config.gradient_clip > 0:
                grad = np.clip(grad, -self.config.gradient_clip, self.config.gradient_clip)
            
            # Adam update
            state["t"] += 1
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
            
            m_hat = state["m"] / (1 - beta1 ** state["t"])
            v_hat = state["v"] / (1 - beta2 ** state["t"])
            
            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Weight decay
            if self.config.weight_decay > 0:
                param -= lr * self.config.weight_decay * param

    def _get_learning_rate(self) -> float:
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * (self.global_step / self.config.warmup_steps)
        return self.config.learning_rate

    def train(self):
        print(f"\n🚀 Starting training for {self.config.epochs} epochs...")
        start_time = time.time()
        
        # Add validation tracking
        self.best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self.train_dataset.shuffle()
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.train_dataset), self.config.batch_size):
                batch_input, batch_target = self.train_dataset.get_batch(self.config.batch_size, i)
                if batch_input.shape[0] == 0: continue
                
                metrics = self.train_step(batch_input, batch_target)
                epoch_loss += metrics["loss"]
                num_batches += 1
                
                # Store quantum metrics
                self.quantum_metrics_history.append(metrics)
                
                if self.global_step % self.config.log_interval == 0:
                    print(f"Epoch {epoch} | Step {self.global_step} | Loss: {metrics['loss']:.4f} | Coherence: {metrics.get('avg_coherence', 0):.3f}")
                
                if self.global_step % self.config.checkpoint_interval == 0:
                    checkpoint_path = Path(self.config.save_path) / f"checkpoint_{self.global_step}.npz"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(checkpoint_path))
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.train_losses.append(avg_epoch_loss)
            
            # Update best validation loss
            if avg_epoch_loss < self.best_val_loss:
                self.best_val_loss = avg_epoch_loss
            
            print(f"✅ Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
        print(f"🏁 Training finished in {time.time() - start_time:.2f}s")
        
        final_path = Path(self.config.save_path) / "final_model.npz"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(final_path))
