"""
Quantum Attention Mechanism
Implements quantum superposition, entanglement, and interference for attention
"""

import cmath
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QuantumSuperposition:
    """
    Represents a quantum superposition state
    Each basis state has a complex amplitude
    """
    amplitudes: np.ndarray  # Complex amplitudes for each basis state
    basis_labels: List[str]  # Labels for basis states (e.g., tokens)
    
    def __post_init__(self):
        # Normalize to ensure sum of probabilities = 1
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    def probabilities(self) -> np.ndarray:
        """Get probability distribution from amplitudes"""
        return np.abs(self.amplitudes) ** 2
    
    def measure(self) -> Tuple[str, float]:
        """
        Collapse superposition to a single basis state (quantum measurement)
        Returns: (basis_label, probability)
        """
        probs = self.probabilities()
        idx = np.random.choice(len(self.amplitudes), p=probs)
        return self.basis_labels[idx], probs[idx]
    
    def entangle_with(self, other: 'QuantumSuperposition') -> 'QuantumSuperposition':
        """
        Create entangled superposition with another state
        Uses tensor product of amplitude vectors
        """
        # Tensor product of amplitude vectors
        new_amplitudes = np.kron(self.amplitudes, other.amplitudes)
        
        # Create combined basis labels
        new_labels = []
        for lbl1 in self.basis_labels:
            for lbl2 in other.basis_labels:
                new_labels.append(f"{lbl1}⊗{lbl2}")
        
        return QuantumSuperposition(new_amplitudes, new_labels)
    
    def interfere(self, other: 'QuantumSuperposition', alpha: float = 0.5) -> 'QuantumSuperposition':
        """
        Interfere this superposition with another
        Simulates quantum interference patterns
        """
        # Ensure same dimension
        if len(self.amplitudes) != len(other.amplitudes):
            raise ValueError("Cannot interfere states of different dimensions")
        
        # Quantum interference: superposition with phase
        new_amplitudes = alpha * self.amplitudes + (1 - alpha) * other.amplitudes
        
        return QuantumSuperposition(new_amplitudes, self.basis_labels.copy())
    
    def apply_phase_shift(self, phi: float, basis_idx: Optional[int] = None):
        """Apply phase shift to amplitudes"""
        if basis_idx is None:
            # Apply to all amplitudes
            self.amplitudes *= cmath.exp(1j * phi)
        else:
            # Apply to specific basis
            if 0 <= basis_idx < len(self.amplitudes):
                self.amplitudes[basis_idx] *= cmath.exp(1j * phi)
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes = self.amplitudes / norm


class QuantumAttention:
    """
    Quantum-inspired attention mechanism using superposition and interference
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        """
        Initialize quantum attention
        
        Args:
            d_model: Dimension of model
            n_heads: Number of attention heads
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Initialize quantum rotation matrices for each head
        self.rotation_matrices = []
        for _ in range(n_heads):
            # Random unitary matrix (quantum gate)
            Q = np.random.randn(self.head_dim, self.head_dim)
            # Make it unitary via QR decomposition
            Q, _ = np.linalg.qr(Q)
            self.rotation_matrices.append(Q)
        
        # Initialize scaling parameters
        self.temperature = np.sqrt(self.head_dim)
    
    def compute_quantum_attention(
        self, 
        query: np.ndarray, 
        key: np.ndarray, 
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute quantum attention scores
        
        Args:
            query: Query vectors [batch, seq_len, d_model]
            key: Key vectors [batch, seq_len, d_model]
            value: Value vectors [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights, quantum_metrics)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Reshape for multi-head attention
        query_reshaped = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        key_reshaped = key.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        value_reshaped = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for matrix operations
        q = query_reshaped.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_len, head_dim]
        k = key_reshaped.transpose(0, 2, 1, 3)
        v = value_reshaped.transpose(0, 2, 1, 3)
        
        # Apply quantum rotation gates
        rotated_query = np.zeros_like(q)
        rotated_key = np.zeros_like(k)
        
        for head in range(self.n_heads):
            rotated_query[:, head] = q[:, head] @ self.rotation_matrices[head].T
            rotated_key[:, head] = k[:, head] @ self.rotation_matrices[head].T
        
        # Compute quantum attention scores using complex-valued inner product
        # Convert to complex domain for quantum interference
        query_complex = rotated_query.astype(np.complex128)
        key_complex = rotated_key.astype(np.complex128)
        
        # Quantum attention: complex inner product
        attention_scores = np.zeros((batch_size, self.n_heads, seq_len, seq_len), dtype=np.complex128)
        
        for b in range(batch_size):
            for h in range(self.n_heads):
                # Matrix multiplication in complex domain
                attention_scores[b, h] = query_complex[b, h] @ key_complex[b, h].conj().T
        
        # Scale scores
        attention_scores = attention_scores / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            mask_reshaped = mask.reshape(batch_size, 1, 1, seq_len)
            attention_scores = np.where(mask_reshaped == 0, -1e10, attention_scores)
        
        # Compute quantum probability amplitudes
        # Use complex exponential for quantum interference
        attention_weights_complex = np.exp(attention_scores)
        
        # Normalize to get valid probabilities
        attention_weights_real = np.abs(attention_weights_complex)
        attention_weights_real = attention_weights_real / (np.sum(attention_weights_real, axis=-1, keepdims=True) + 1e-10)
        
        # Apply attention to values
        output_complex = np.zeros_like(v, dtype=np.complex128)
        for b in range(batch_size):
            for h in range(self.n_heads):
                output_complex[b, h] = attention_weights_real[b, h] @ v[b, h].astype(np.complex128)
        
        # Take real part for final output
        out = np.real(output_complex)
        
        # Combine heads
        out = out.transpose(0, 2, 1, 3)  # [batch, seq_len, n_heads, head_dim]
        out = out.reshape(batch_size, seq_len, d_model)
        
        # Compute quantum metrics
        quantum_metrics = {
            "coherence": self._compute_coherence(attention_weights_real),
            "entanglement": self._compute_entanglement(attention_weights_real),
            "interference": self._compute_interference(attention_weights_complex),
            "quantum_fidelity": self._compute_fidelity(attention_weights_real)
        }

        # Store cache for backward pass
        self.cache = {
            "q": q, "k": k, "v": v,
            "rotated_query": rotated_query,
            "rotated_key": rotated_key,
            "attention_weights_real": attention_weights_real,
            "output_complex": output_complex,
            "mask": mask
        }
        
        return out, attention_weights_real, quantum_metrics

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Real backpropagation for quantum attention
        """
        q, k, v = self.cache["q"], self.cache["k"], self.cache["v"]
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # grad_output is [batch, seq_len, d_model]
        grad_output = grad_output.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        grad_output = grad_output.transpose(0, 2, 1, 3) # [batch, n_heads, seq_len, head_dim]
        
        weights = self.cache["attention_weights_real"]
        
        # dL/dv = weights.T @ grad_output
        grad_v = np.zeros_like(v)
        for b in range(batch_size):
            for h in range(n_heads):
                grad_v[b, h] = weights[b, h].T @ grad_output[b, h]
        
        # dL/dweights = grad_output @ v.T
        grad_weights = np.zeros_like(weights)
        for b in range(batch_size):
            for h in range(n_heads):
                grad_weights[b, h] = grad_output[b, h] @ v[b, h].T
                
        # dL/dscores (simplified for real part of exp)
        # S = softmax(scores) -> dL/dscores = S * (grad_weights - sum(S * grad_weights))
        grad_scores = weights * (grad_weights - np.sum(weights * grad_weights, axis=-1, keepdims=True))
        grad_scores /= self.temperature
        
        # dL/dq = grad_scores @ k
        # dL/dk = grad_scores.T @ q
        grad_q_rotated = np.zeros_like(q)
        grad_k_rotated = np.zeros_like(k)
        for b in range(batch_size):
            for h in range(n_heads):
                grad_q_rotated[b, h] = grad_scores[b, h] @ self.cache["rotated_key"][b, h]
                grad_k_rotated[b, h] = grad_scores[b, h].T @ self.cache["rotated_query"][b, h]
        
        # Backprop through rotations
        grad_q = np.zeros_like(q)
        grad_k = np.zeros_like(k)
        for h in range(n_heads):
            grad_q[:, h] = grad_q_rotated[:, h] @ self.rotation_matrices[h]
            grad_k[:, h] = grad_k_rotated[:, h] @ self.rotation_matrices[h]
            
        # Reshape back
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return grad_q, grad_k, grad_v
    
    def _compute_coherence(self, weights: np.ndarray) -> float:
        """
        Compute quantum coherence of attention weights
        Coherence measures how quantum-like the distribution is
        """
        batch_size, n_heads, seq_len, _ = weights.shape
        coherences = []
        
        for b in range(batch_size):
            for h in range(n_heads):
                # Use Von Neumann entropy
                probs = weights[b, h]
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                max_entropy = np.log(seq_len)
                coherence = 1.0 - (entropy / max_entropy)
                coherences.append(np.mean(coherence))
        
        return float(np.mean(coherences))
    
    def _compute_entanglement(self, weights: np.ndarray) -> float:
        """
        Compute entanglement entropy between attention heads
        """
        batch_size, n_heads, seq_len, _ = weights.shape
        
        if n_heads < 2:
            return 0.0
        
        # Treat each head as a subsystem
        head_probs = np.mean(weights, axis=(0, 2, 3))  # Probability for each head
        
        # Compute entropy
        entropy = -np.sum(head_probs * np.log(head_probs + 1e-10))
        max_entropy = np.log(n_heads)
        
        entanglement = entropy / max_entropy
        return float(entanglement)
    
    def _compute_interference(self, weights_complex: np.ndarray) -> float:
        """
        Compute quantum interference from complex attention weights
        """
        # Measure phase coherence
        phases = np.angle(weights_complex)
        
        # Compute circular variance
        mean_cos = np.mean(np.cos(phases))
        mean_sin = np.mean(np.sin(phases))
        
        circular_variance = 1.0 - np.sqrt(mean_cos**2 + mean_sin**2)
        interference = 1.0 - circular_variance
        
        return float(interference)
    
    def _compute_fidelity(self, weights: np.ndarray) -> float:
        """
        Compute quantum fidelity (closeness to pure quantum state)
        """
        # Trace of squared density matrix
        batch_size, n_heads, seq_len, _ = weights.shape
        
        fidelities = []
        for b in range(batch_size):
            for h in range(n_heads):
                # Purity of quantum state
                rho = weights[b, h]
                purity = np.sum(rho**2, axis=-1)
                fidelity = np.mean(np.sqrt(purity))
                fidelities.append(fidelity)
        
        return float(np.mean(fidelities))


class QuantumSuperpositionAttention:
    """
    Attention using explicit quantum superposition states
    Each token is represented as a superposition over semantic basis states
    """
    
    def __init__(self, n_basis_states: int = 64, vocab_size: int = 50000):
        """
        Initialize superposition attention
        
        Args:
            n_basis_states: Number of semantic basis states
            vocab_size: Size of vocabulary
        """
        self.n_basis_states = n_basis_states
        self.vocab_size = vocab_size
        
        # Initialize token embeddings as superposition parameters
        # Each token maps to amplitude distribution over basis states
        self.token_amplitudes = np.random.randn(vocab_size, n_basis_states)
        # Normalize to valid quantum states
        norms = np.linalg.norm(self.token_amplitudes, axis=1, keepdims=True)
        self.token_amplitudes = self.token_amplitudes / (norms + 1e-10)
        
        # Initialize basis state representations
        self.basis_embeddings = np.random.randn(n_basis_states, 128)
    
    def tokens_to_superpositions(self, token_ids: np.ndarray) -> List[QuantumSuperposition]:
        """
        Convert token IDs to quantum superposition states
        
        Args:
            token_ids: Array of token IDs
            
        Returns:
            List of QuantumSuperposition objects
        """
        superpositions = []
        
        for token_id in token_ids:
            if token_id < self.vocab_size:
                amplitudes = self.token_amplitudes[token_id].astype(np.complex128)
                labels = [f"basis_{i}" for i in range(self.n_basis_states)]
                superpositions.append(QuantumSuperposition(amplitudes, labels))
        
        return superpositions
    
    def superposition_attention(
        self, 
        query_superpositions: List[QuantumSuperposition],
        key_superpositions: List[QuantumSuperposition],
        value_superpositions: List[QuantumSuperposition]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Compute attention using quantum superposition
        
        Args:
            query_superpositions: List of query superpositions
            key_superpositions: List of key superpositions
            value_superpositions: List of value superpositions
            
        Returns:
            Tuple of (attention_output, quantum_metrics)
        """
        attention_weights = []
        quantum_metrics = []
        
        for i, query_sp in enumerate(query_superpositions):
            weights = []
            
            for j, key_sp in enumerate(key_superpositions):
                # Compute quantum overlap (inner product of amplitude vectors)
                overlap = np.vdot(query_sp.amplitudes, key_sp.amplitudes)
                weight = np.abs(overlap) ** 2  # Probability
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / (np.sum(weights) + 1e-10)
            attention_weights.append(weights)
            
            # Compute quantum metrics
            metrics = {
                "overlap_entropy": self._compute_entropy(weights),
                "superposition_coherence": self._compute_superposition_coherence(query_sp),
                "interference_pattern": self._detect_interference_pattern(weights)
            }
            quantum_metrics.append(metrics)
        
        attention_weights = np.array(attention_weights)
        
        # Weighted combination of value superpositions
        output = np.zeros((len(query_superpositions), self.n_basis_states))
        for i in range(len(query_superpositions)):
            for j in range(len(value_superpositions)):
                output[i] += attention_weights[i, j] * value_superpositions[j].amplitudes.real
        
        return output, quantum_metrics
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute Shannon entropy of attention weights"""
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        return float(entropy)
    
    def _compute_superposition_coherence(self, sp: QuantumSuperposition) -> float:
        """Compute coherence of superposition state"""
        probs = sp.probabilities()
        purity = np.sum(probs**2)
        return float(purity)
    
    def _detect_interference_pattern(self, weights: np.ndarray) -> Dict[str, Any]:
        """Detect interference patterns in attention weights"""
        # Look for constructive/destructive interference patterns
        if len(weights) < 2:
            return {"has_interference": False}
        
        # Check for oscillating pattern (constructive/destructive)
        diffs = np.diff(weights)
        oscillations = np.sum(np.sign(diffs[:-1]) != np.sign(diffs[1:]))
        
        return {
            "has_interference": oscillations > 1,
            "oscillation_count": int(oscillations),
            "pattern_type": "oscillatory" if oscillations > 1 else "monotonic"
        }


__all__ = [
    "QuantumSuperposition",
    "QuantumAttention",
    "QuantumSuperpositionAttention",
]
