"""
Quantum Transformer Architecture
Neural network from scratch using quantum-inspired operations with real backpropagation
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .quantum_attention import QuantumAttention, QuantumSuperposition


@dataclass
class QuantumState:
    """
    Represents the state of the quantum transformer
    Includes all learnable parameters
    """
    embedding_matrix: np.ndarray  # Token embeddings
    position_embeddings: np.ndarray
    layer_weights: List[Dict[str, np.ndarray]]  # Weights for each layer
    layer_norms: List[Dict[str, np.ndarray]]  # Layer normalization params
    output_weights: np.ndarray  # Final projection to vocab
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            "embedding_matrix": self.embedding_matrix.tolist(),
            "position_embeddings": self.position_embeddings.tolist(),
            "layer_weights": [
                {
                    "query": lw["query"].tolist(),
                    "key": lw["key"].tolist(),
                    "value": lw["value"].tolist(),
                    "ffn1": lw["ffn1"].tolist(),
                    "ffn2": lw["ffn2"].tolist(),
                }
                for lw in self.layer_weights
            ],
            "layer_norms": [
                {
                    "gamma1": ln["gamma1"].tolist(),
                    "beta1": ln["beta1"].tolist(),
                    "gamma2": ln["gamma2"].tolist(),
                    "beta2": ln["beta2"].tolist(),
                }
                for ln in self.layer_norms
            ],
            "output_weights": self.output_weights.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantumState":
        """Deserialize state from dictionary"""
        return cls(
            embedding_matrix=np.array(data["embedding_matrix"]),
            position_embeddings=np.array(data["position_embeddings"]),
            layer_weights=[
                {
                    "query": np.array(lw["query"]),
                    "key": np.array(lw["key"]),
                    "value": np.array(lw["value"]),
                    "ffn1": np.array(lw["ffn1"]),
                    "ffn2": np.array(lw["ffn2"]),
                }
                for lw in data["layer_weights"]
            ],
            layer_norms=[
                {
                    "gamma1": np.array(ln["gamma1"]),
                    "beta1": np.array(ln["beta1"]),
                    "gamma2": np.array(ln["gamma2"]),
                    "beta2": np.array(ln["beta2"]),
                }
                for ln in data["layer_norms"]
            ],
            output_weights=np.array(data["output_weights"]),
        )


class QuantumLayer:
    """
    Single transformer layer with quantum attention and real backprop
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize quantum transformer layer
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (d_model + d_model))
        self.query_proj = np.random.uniform(-limit, limit, (d_model, d_model))
        self.key_proj = np.random.uniform(-limit, limit, (d_model, d_model))
        self.value_proj = np.random.uniform(-limit, limit, (d_model, d_model))
        
        # Initialize feed-forward weights
        limit_ff = np.sqrt(6.0 / (d_model + d_ff))
        self.ffn1 = np.random.uniform(-limit_ff, limit_ff, (d_model, d_ff))
        self.ffn2 = np.random.uniform(-limit_ff, limit_ff, (d_ff, d_model))
        
        # Initialize layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        # Initialize quantum attention
        self.quantum_attention = QuantumAttention(d_model, n_heads)
        
        # Cache for backprop
        self.cache = {}
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass with caching
        """
        # 1. Layer norm 1
        x_norm1, norm1_cache = self._layer_norm_forward(x, self.gamma1, self.beta1)
        
        # 2. Attention projections
        q = x_norm1 @ self.query_proj
        k = x_norm1 @ self.key_proj
        v = x_norm1 @ self.value_proj
        
        # 3. Quantum attention
        attn_out, attn_weights, metrics = self.quantum_attention.compute_quantum_attention(q, k, v, mask)
        
        # 4. Residual 1
        x_res1 = x + attn_out
        
        # 5. Layer norm 2
        x_norm2, norm2_cache = self._layer_norm_forward(x_res1, self.gamma2, self.beta2)
        
        # 6. FFN
        # FFN1
        h1 = x_norm2 @ self.ffn1
        # GELU
        h_gelu = self._gelu_forward(h1)
        # FFN2
        h2 = h_gelu @ self.ffn2
        
        # 7. Residual 2
        out = x_res1 + h2
        
        # Store in cache
        self.cache = {
            "x": x,
            "x_norm1": x_norm1,
            "norm1_cache": norm1_cache,
            "q": q, "k": k, "v": v,
            "attn_out": attn_out,
            "x_res1": x_res1,
            "x_norm2": x_norm2,
            "norm2_cache": norm2_cache,
            "h1": h1,
            "h_gelu": h_gelu,
            "h2": h2,
            "mask": mask
        }
        
        return out, metrics

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Real backpropagation through the layer
        """
        grads = {}
        
        # Backprop through Residual 2
        grad_h2 = grad_out
        grad_x_res1 = grad_out # Residual connection
        
        # Backprop through FFN
        # grad_h2 = dL/dh2 [batch, seq, d_model]
        # h2 = h_gelu @ ffn2
        grads["ffn2"] = self.cache["h_gelu"].transpose(0, 2, 1) @ grad_h2
        grads["ffn2"] = np.sum(grads["ffn2"], axis=0)
        grad_h_gelu = grad_h2 @ self.ffn2.T
        
        # GELU backward
        grad_h1 = self._gelu_backward(self.cache["h1"], grad_h_gelu)
        
        # FFN1 backward
        grads["ffn1"] = self.cache["x_norm2"].transpose(0, 2, 1) @ grad_h1
        grads["ffn1"] = np.sum(grads["ffn1"], axis=0)
        grad_x_norm2 = grad_h1 @ self.ffn1.T
        
        # Layer norm 2 backward
        grad_x_res1_from_norm, grads["gamma2"], grads["beta2"] = self._layer_norm_backward(
            grad_x_norm2, self.cache["norm2_cache"], self.gamma2
        )
        grad_x_res1 += grad_x_res1_from_norm
        
        # Backprop through Attention
        grad_attn_out = grad_x_res1
        grad_x_res1_attn = grad_x_res1 # Residual connection for first part
        
        # Quantum Attention backward
        grad_q, grad_k, grad_v = self.quantum_attention.backward(grad_attn_out)
        
        # Attention Projections backward
        grads["query"] = self.cache["x_norm1"].transpose(0, 2, 1) @ grad_q
        grads["query"] = np.sum(grads["query"], axis=0)
        grads["key"] = self.cache["x_norm1"].transpose(0, 2, 1) @ grad_k
        grads["key"] = np.sum(grads["key"], axis=0)
        grads["value"] = self.cache["x_norm1"].transpose(0, 2, 1) @ grad_v
        grads["value"] = np.sum(grads["value"], axis=0)
        
        grad_x_norm1 = (grad_q @ self.query_proj.T + 
                        grad_k @ self.key_proj.T + 
                        grad_v @ self.value_proj.T)
        
        # Layer norm 1 backward
        grad_x_from_norm1, grads["gamma1"], grads["beta1"] = self._layer_norm_backward(
            grad_x_norm1, self.cache["norm1_cache"], self.gamma1
        )
        
        grad_input = grad_x_res1_attn + grad_x_from_norm1
        
        return grad_input, grads

    def _layer_norm_forward(self, x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_centered = x - mean
        std = np.sqrt(var + 1e-10)
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        cache = (x, x_norm, x_centered, std, gamma)
        return out, cache

    def _layer_norm_backward(self, grad_out, cache, gamma):
        x, x_norm, x_centered, std, gamma = cache
        batch, seq, d = grad_out.shape
        
        grad_gamma = np.sum(grad_out * x_norm, axis=(0, 1))
        grad_beta = np.sum(grad_out, axis=(0, 1))
        
        grad_x_norm = grad_out * gamma
        grad_std = np.sum(grad_x_norm * x_centered * (-1.0 / (std**2)), axis=-1, keepdims=True)
        grad_var = grad_std * 0.5 * (1.0 / std)
        
        grad_x_centered = grad_x_norm / std + (2.0 / d) * x_centered * grad_var
        grad_mean = np.sum(grad_x_centered * -1.0, axis=-1, keepdims=True)
        
        grad_x = grad_x_centered + grad_mean / d
        return grad_x, grad_gamma, grad_beta

    def _gelu_forward(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_backward(self, x, grad_out):
        # Approximate derivative of GELU
        s = np.sqrt(2.0 / np.pi)
        term1 = 0.5 * (1.0 + np.tanh(s * (x + 0.044715 * x**3)))
        term2 = 0.5 * x * (1.0 - np.tanh(s * (x + 0.044715 * x**3))**2) * s * (1.0 + 3 * 0.044715 * x**2)
        return grad_out * (term1 + term2)


class QuantumTransformer:
    """
    Complete quantum transformer model built from scratch with real backprop
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Token embeddings
        limit = np.sqrt(6.0 / (vocab_size + d_model))
        self.embedding = np.random.uniform(-limit, limit, (vocab_size, d_model))
        
        # Position embeddings
        self.pos_embedding = self._create_position_embeddings()
        
        # Transformer layers
        self.layers = [
            QuantumLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ]
        
        # Final projection
        limit_out = np.sqrt(6.0 / (d_model + vocab_size))
        self.output_projection = np.random.uniform(-limit_out, limit_out, (d_model, vocab_size))
        
        self.cache = {}
        
        print(f"✨ Initialized QuantumTransformer with Real Backprop")
        print(f"   Config: {n_layers}L, {n_heads}H, {d_model}D, {d_ff}FF")

    def _create_position_embeddings(self) -> np.ndarray:
        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_seq_len, self.d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return pe

    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding
        x_emb = self.embedding[input_ids]
        
        # 2. Positional Embedding
        x = x_emb + self.pos_embedding[:seq_len]
        
        # 3. Layers
        layer_metrics = []
        layer_inputs = [x]
        for layer in self.layers:
            x, metrics = layer.forward(x, mask)
            layer_inputs.append(x)
            layer_metrics.append(metrics)
            
        # 4. Final Projection
        logits = x @ self.output_projection
        
        # Store cache
        self.cache = {
            "input_ids": input_ids,
            "layer_inputs": layer_inputs,
            "logits": logits,
            "mask": mask
        }
        
        # Aggregate metrics
        avg_metrics = {
            "avg_coherence": np.mean([m.get("coherence", 0) for m in layer_metrics]),
            "avg_entanglement": np.mean([m.get("entanglement", 0) for m in layer_metrics]),
            "avg_interference": np.mean([m.get("interference", 0) for m in layer_metrics]),
            "avg_fidelity": np.mean([m.get("quantum_fidelity", 0) for m in layer_metrics]),
        }
        
        return logits, avg_metrics

    def backward(self, grad_logits: np.ndarray) -> Dict[str, Any]:
        """
        Full backward pass through the entire transformer
        """
        all_grads = {}
        
        # 1. Output projection backward
        # logits = last_layer_x @ output_projection
        last_x = self.cache["layer_inputs"][-1]
        all_grads["output_projection"] = last_x.transpose(0, 2, 1) @ grad_logits
        all_grads["output_projection"] = np.sum(all_grads["output_projection"], axis=0)
        
        grad_x = grad_logits @ self.output_projection.T
        
        # 2. Layers backward
        for i in reversed(range(self.n_layers)):
            grad_x, layer_grads = self.layers[i].backward(grad_x)
            # Prefix layer grads
            for name, g in layer_grads.items():
                all_grads[f"layer_{i}_{name}"] = g
                
        # 3. Embedding backward
        # grad_x is grad w.r.t input to first layer (emb + pos)
        # pos is fixed, so grad w.r.t emb is grad_x
        input_ids = self.cache["input_ids"]
        grad_emb = np.zeros_like(self.embedding)
        for b in range(input_ids.shape[0]):
            for s in range(input_ids.shape[1]):
                grad_emb[input_ids[b, s]] += grad_x[b, s]
        all_grads["embedding"] = grad_emb
        
        return all_grads

    def generate(self, prompt: str, tokenizer, max_tokens: int = 50, temperature: float = 0.7, top_k: int = 50):
        """
        Generate text from prompt
        
        Returns:
            Tuple of (generated_text, metrics)
        """
        input_ids = tokenizer.encode(prompt)
        input_ids = np.array(input_ids).reshape(1, -1)
        generated = input_ids[0].tolist()
        
        # Track quantum metrics during generation
        quantum_metrics_list = []
        
        for _ in range(max_tokens):
            # Limit context to max_seq_len
            context = generated[-self.max_seq_len:]
            logits, metrics = self.forward(np.array(context).reshape(1, -1))
            
            quantum_metrics_list.append(metrics)
            
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                top_k_logits = next_token_logits[top_k_indices]
                top_k_probs = self._softmax(top_k_logits)
                next_token = top_k_indices[np.random.choice(len(top_k_probs), p=top_k_probs)]
            else:
                probs = self._softmax(next_token_logits)
                next_token = np.random.choice(len(probs), p=probs)
            
            generated.append(int(next_token))
            
            # Check for EOS token
            if hasattr(tokenizer, 'eos_token'):
                if tokenizer.decode([next_token]) == tokenizer.eos_token:
                    break
        
        generated_text = tokenizer.decode(generated)
        
        # Aggregate quantum metrics
        avg_metrics = {
            "quantum_metrics": {
                "avg_coherence": float(np.mean([m.get("avg_coherence", 0) for m in quantum_metrics_list])),
                "avg_entanglement": float(np.mean([m.get("avg_entanglement", 0) for m in quantum_metrics_list])),
                "avg_interference": float(np.mean([m.get("avg_interference", 0) for m in quantum_metrics_list])),
                "avg_fidelity": float(np.mean([m.get("avg_fidelity", 0) for m in quantum_metrics_list])),
            },
            "generated_tokens": len(generated) - len(input_ids[0])
        }
        
        return generated_text, avg_metrics

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def save(self, path: str):
        # Save using numpy binary format for efficiency
        params = {
            "embedding": self.embedding,
            "output_projection": self.output_projection,
            "vocab_size": np.array(self.vocab_size),
            "d_model": np.array(self.d_model),
            "n_layers": np.array(self.n_layers),
            "n_heads": np.array(self.n_heads),
            "d_ff": np.array(self.d_ff)
        }
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_q"] = layer.query_proj
            params[f"layer_{i}_k"] = layer.key_proj
            params[f"layer_{i}_v"] = layer.value_proj
            params[f"layer_{i}_ffn1"] = layer.ffn1
            params[f"layer_{i}_ffn2"] = layer.ffn2
            params[f"layer_{i}_g1"] = layer.gamma1
            params[f"layer_{i}_b1"] = layer.beta1
            params[f"layer_{i}_g2"] = layer.gamma2
            params[f"layer_{i}_b2"] = layer.beta2
            for j, rot in enumerate(layer.quantum_attention.rotation_matrices):
                params[f"layer_{i}_rot_{j}"] = rot
        
        np.savez(path, **params)

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        model = cls(
            int(data["vocab_size"]), 
            int(data["d_model"]), 
            int(data["n_layers"]), 
            int(data["n_heads"]), 
            int(data["d_ff"])
        )
        model.embedding = data["embedding"]
        model.output_projection = data["output_projection"]
        for i in range(model.n_layers):
            model.layers[i].query_proj = data[f"layer_{i}_q"]
            model.layers[i].key_proj = data[f"layer_{i}_k"]
            model.layers[i].value_proj = data[f"layer_{i}_v"]
            model.layers[i].ffn1 = data[f"layer_{i}_ffn1"]
            model.layers[i].ffn2 = data[f"layer_{i}_ffn2"]
            model.layers[i].gamma1 = data[f"layer_{i}_g1"]
            model.layers[i].beta1 = data[f"layer_{i}_b1"]
            model.layers[i].gamma2 = data[f"layer_{i}_g2"]
            model.layers[i].beta2 = data[f"layer_{i}_b2"]
            model.layers[i].quantum_attention.rotation_matrices = [
                data[f"layer_{i}_rot_{j}"] for j in range(model.n_heads)
            ]
        return model

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.word_to_id = {self.pad_token: 0, self.eos_token: 1}
        self.id_to_word = {0: self.pad_token, 1: self.eos_token}
        self.next_id = 2

    def encode(self, text):
        words = text.lower().split()
        ids = []
        for w in words:
            if w not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[w] = self.next_id
                    self.id_to_word[self.next_id] = w
                    self.next_id += 1
                else:
                    ids.append(0) # pad/unk
                    continue
            ids.append(self.word_to_id[w])
        return ids + [1] # add eos

    def decode(self, ids):
        return " ".join([self.id_to_word.get(i, "<?>") for i in ids if i > 0])
    
    def save(self, path: str):
        """Save tokenizer to JSON"""
        import json
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "word_to_id": self.word_to_id,
                "id_to_word": {int(k): v for k, v in self.id_to_word.items()},
                "next_id": self.next_id
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from JSON"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.word_to_id = data["word_to_id"]
        tokenizer.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        tokenizer.next_id = data["next_id"]
        return tokenizer
