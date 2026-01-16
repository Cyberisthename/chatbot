"""
Quantum Transformer Architecture
Neural network from scratch using quantum-inspired operations
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
                    "gamma": ln["gamma"].tolist(),
                    "beta": ln["beta"].tolist(),
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
                    "gamma": np.array(ln["gamma"]),
                    "beta": np.array(ln["beta"]),
                }
                for ln in data["layer_norms"]
            ],
            output_weights=np.array(data["output_weights"]),
        )


class QuantumLayer:
    """
    Single transformer layer with quantum attention
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize quantum transformer layer
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Initialize attention weights (Q, K, V projections)
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
        
        # Store metrics
        self.quantum_metrics = {}
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through quantum layer
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, metrics)
        """
        # Layer norm before attention
        x_norm = self._layer_norm(x, self.gamma1, self.beta1)
        
        # Compute Q, K, V
        query = x_norm @ self.query_proj
        key = x_norm @ self.key_proj
        value = x_norm @ self.value_proj
        
        # Quantum attention
        attn_output, attn_weights, quantum_metrics = self.quantum_attention.compute_quantum_attention(
            query, key, value, mask
        )
        
        # Store quantum metrics
        self.quantum_metrics = quantum_metrics
        
        # Residual connection
        x = x + self._dropout(attn_output)
        
        # Layer norm before FFN
        x_norm = self._layer_norm(x, self.gamma2, self.beta2)
        
        # Feed-forward network
        ffn_output = self._feed_forward(x_norm)
        
        # Residual connection
        x = x + self._dropout(ffn_output)
        
        return x, quantum_metrics
    
    def backward(self, grad_output: np.ndarray, lr: float) -> Dict[str, np.ndarray]:
        """
        Backward pass - compute gradients and update weights
        
        Args:
            grad_output: Gradient from next layer
            lr: Learning rate
            
        Returns:
            Dictionary of gradients
        """
        # Simplified backpropagation (in full implementation, would be more complex)
        # This is a placeholder for the actual gradient computation
        
        gradients = {
            "query": np.zeros_like(self.query_proj),
            "key": np.zeros_like(self.key_proj),
            "value": np.zeros_like(self.value_proj),
            "ffn1": np.zeros_like(self.ffn1),
            "ffn2": np.zeros_like(self.ffn2),
        }
        
        # Update weights with small random noise for now
        # In real implementation, would compute actual gradients
        noise_scale = lr * 0.01
        self.query_proj += np.random.randn(*self.query_proj.shape) * noise_scale
        self.key_proj += np.random.randn(*self.key_proj.shape) * noise_scale
        self.value_proj += np.random.randn(*self.value_proj.shape) * noise_scale
        self.ffn1 += np.random.randn(*self.ffn1.shape) * noise_scale
        self.ffn2 += np.random.randn(*self.ffn2.shape) * noise_scale
        
        return gradients
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-10)
        return gamma * normalized + beta
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network with GELU activation"""
        hidden = x @ self.ffn1
        # GELU activation
        hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
        output = hidden @ self.ffn2
        return output
    
    def _dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Dropout regularization"""
        if not training or self.dropout == 0:
            return x
        mask = (np.random.random(x.shape) > self.dropout).astype(np.float32)
        return x * mask / (1.0 - self.dropout)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all trainable weights"""
        return {
            "query": self.query_proj,
            "key": self.key_proj,
            "value": self.value_proj,
            "ffn1": self.ffn1,
            "ffn2": self.ffn2,
        }


class QuantumTransformer:
    """
    Complete quantum transformer model built from scratch
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
        """
        Initialize quantum transformer
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Initialize token embeddings
        limit = np.sqrt(6.0 / (vocab_size + d_model))
        self.embedding = np.random.uniform(-limit, limit, (vocab_size, d_model))
        
        # Initialize position embeddings
        self.pos_embedding = self._create_position_embeddings()
        
        # Initialize transformer layers
        self.layers = [
            QuantumLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ]
        
        # Initialize final projection
        limit_out = np.sqrt(6.0 / (d_model + vocab_size))
        self.output_projection = np.random.uniform(-limit_out, limit_out, (d_model, vocab_size))
        
        # Training state
        self.training_step = 0
        self.loss_history = []
        self.quantum_metrics_history = []
        
        print(f"✨ Initialized QuantumTransformer from scratch")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model dimension: {d_model}")
        print(f"   Layers: {n_layers}")
        print(f"   Attention heads: {n_heads}")
        print(f"   Parameters: {self._count_parameters():,}")
    
    def _create_position_embeddings(self) -> np.ndarray:
        """Create sinusoidal position embeddings"""
        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_seq_len, self.d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        
        return pe
    
    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        params = (
            np.prod(self.embedding.shape) +
            np.prod(self.pos_embedding.shape) +
            np.prod(self.output_projection.shape)
        )
        
        for layer in self.layers:
            params += (
                np.prod(layer.query_proj.shape) +
                np.prod(layer.key_proj.shape) +
                np.prod(layer.value_proj.shape) +
                np.prod(layer.ffn1.shape) +
                np.prod(layer.ffn2.shape) +
                2 * len(layer.gamma1)  # layer norm params
            )
        
        return params
    
    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through model
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Optional attention mask
            
        Returns:
            Tuple of (logits, metrics)
        """
        batch_size, seq_len = input_ids.shape
        
        # Lookup token embeddings
        x = self.embedding[input_ids]
        
        # Add position embeddings
        x = x + self.pos_embedding[:seq_len]
        
        # Apply dropout
        if self.dropout > 0:
            dropout_mask = (np.random.random(x.shape) > self.dropout).astype(np.float32)
            x = x * dropout_mask / (1.0 - self.dropout)
        
        # Pass through transformer layers
        all_quantum_metrics = []
        for i, layer in enumerate(self.layers):
            x, quantum_metrics = layer.forward(x, mask)
            all_quantum_metrics.append(quantum_metrics)
        
        # Final projection to vocabulary
        logits = x @ self.output_projection
        
        # Aggregate quantum metrics
        metrics = {
            "layer_metrics": all_quantum_metrics,
            "avg_coherence": np.mean([m.get("coherence", 0) for m in all_quantum_metrics]),
            "avg_entanglement": np.mean([m.get("entanglement", 0) for m in all_quantum_metrics]),
            "avg_interference": np.mean([m.get("interference", 0) for m in all_quantum_metrics]),
            "avg_fidelity": np.mean([m.get("quantum_fidelity", 0) for m in all_quantum_metrics]),
        }
        
        return logits, metrics
    
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt text
            tokenizer: Tokenizer for encoding/decoding
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Tuple of (generated_text, generation_metrics)
        """
        start_time = time.time()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_ids = np.array(input_ids).reshape(1, -1)
        
        generated_ids = input_ids[0].copy()
        
        for i in range(max_tokens):
            # Forward pass
            logits, metrics = self.forward(generated_ids.reshape(1, -1))
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply softmax
            probs = self._softmax(next_token_logits)
            
            # Top-k sampling
            if top_k is not None:
                top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                mask = np.zeros_like(probs)
                mask[top_k_indices] = 1
                probs = probs * mask
                probs = probs / (np.sum(probs) + 1e-10)
            
            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated_ids = np.append(generated_ids, next_token)
            
            # Check for end token
            if tokenizer.decode([next_token]) == tokenizer.eos_token:
                break
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_ids.tolist())
        
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_ids) / generation_time
        
        generation_metrics = {
            "generated_tokens": len(generated_ids) - len(input_ids[0]),
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "quantum_metrics": metrics,
        }
        
        return generated_text, generation_metrics
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def get_state(self) -> QuantumState:
        """Get model state for saving"""
        layer_weights = []
        layer_norms = []
        
        for layer in self.layers:
            layer_weights.append({
                "query": layer.query_proj,
                "key": layer.key_proj,
                "value": layer.value_proj,
                "ffn1": layer.ffn1,
                "ffn2": layer.ffn2,
            })
            layer_norms.append({
                "gamma": layer.gamma1,
                "beta": layer.beta1,
            })
        
        return QuantumState(
            embedding_matrix=self.embedding,
            position_embeddings=self.pos_embedding,
            layer_weights=layer_weights,
            layer_norms=layer_norms,
            output_weights=self.output_projection,
        )
    
    def set_state(self, state: QuantumState):
        """Set model state from saved checkpoint"""
        self.embedding = state.embedding_matrix
        self.pos_embedding = state.position_embeddings
        self.output_projection = state.output_weights
        
        for i, (layer, lw, ln) in enumerate(zip(self.layers, state.layer_weights, state.layer_norms)):
            layer.query_proj = lw["query"]
            layer.key_proj = lw["key"]
            layer.value_proj = lw["value"]
            layer.ffn1 = lw["ffn1"]
            layer.ffn2 = lw["ffn2"]
            layer.gamma1 = ln["gamma"]
            layer.beta1 = ln["beta"]
    
    def save(self, path: str):
        """Save model state to disk"""
        state = self.get_state()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        print(f"✅ Saved QuantumTransformer to {path}")
    
    def load(self, path: str):
        """Load model state from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        state = QuantumState.from_dict(data)
        self.set_state(state)
        
        print(f"✅ Loaded QuantumTransformer from {path}")


class SimpleTokenizer:
    """
    Simple tokenizer for testing (word-level)
    In production, would use proper tokenizer
    """
    
    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            vocab = ["<pad>", "<unk>", "<eos>", "<bos>"] + list("abcdefghijklmnopqrstuvwxyz ")
        
        self.vocab = vocab
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        for char in text.lower():
            if char in self.word_to_id:
                tokens.append(self.word_to_id[char])
            else:
                tokens.append(self.word_to_id[self.unk_token])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return "".join([self.id_to_word.get(tid, self.unk_token) for tid in token_ids])


__all__ = [
    "QuantumState",
    "QuantumLayer",
    "QuantumTransformer",
    "SimpleTokenizer",
]
