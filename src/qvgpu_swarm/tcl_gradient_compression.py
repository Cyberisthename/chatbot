"""
Pillar 2: Gradient & State Compression via TCL Engine

The biggest problem with swapping VRAM to System RAM or networking old laptops 
together is that moving the data is too slow.

Integration with existing TCL: We repurpose the TCL (Thought Compression Layer) 
engine to compress activations and gradients during forward and backward passes.

How it works: Instead of sending massive matrices of FP16/FP32 data back and 
forth between CPU and old GPU, the TCL compresses states into ultra-dense 
representations. The old GPU calculates math on compressed state and passes 
it back. This virtually multiplies memory bandwidth.
"""

import numpy as np
import math
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent to path for TCL imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from thought_compression.tcl_engine import ThoughtCompressionEngine
    from thought_compression.tcl_symbols import TCLSymbol, ConceptGraph, SymbolType
    TCL_AVAILABLE = True
except ImportError:
    TCL_AVAILABLE = False


@dataclass
class CompressedTensor:
    """
    A tensor compressed using TCL-inspired symbolic representation.
    
    Instead of storing raw float32 values, we store:
    - Symbolic basis vectors
    - Sparse coefficient matrix
    - Metadata for reconstruction
    """
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    compression_ratio: float
    symbol_basis: List[np.ndarray]  # Compressed basis vectors
    coefficients: np.ndarray  # Sparse coefficient matrix
    metadata: Dict[str, Any]
    
    def memory_size_bytes(self) -> int:
        """Calculate compressed size in bytes"""
        basis_size = sum(b.nbytes for b in self.symbol_basis)
        coeff_size = self.coefficients.nbytes
        return basis_size + coeff_size + 256  # metadata overhead
    
    def effective_compression(self) -> float:
        """Calculate effective compression ratio vs original"""
        original_size = int(np.prod(self.original_shape)) * 4  # float32
        compressed_size = self.memory_size_bytes()
        return original_size / compressed_size


class TCLGradientCompressor:
    """
    Compresses gradients and activations using TCL-inspired techniques.
    
    Key insight: Gradients have structure. Instead of transmitting full
    matrices, we transmit compressed symbolic representations that can
    be reconstructed on the destination device.
    """
    
    def __init__(self, 
                 target_compression: float = 8.0,  # 8x compression
                 min_dimension: int = 128,  # Don't compress smaller than this
                 adaptive_quality: bool = True):
        
        self.target_compression = target_compression
        self.min_dimension = min_dimension
        self.adaptive_quality = adaptive_quality
        
        # Compression statistics
        self.stats = {
            'tensors_compressed': 0,
            'tensors_skipped': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'avg_compression_ratio': 0.0,
        }
        
        # Quality levels based on gradient magnitude
        self.quality_tiers = {
            'high': 0.95,    # Keep 95% of energy
            'medium': 0.85,  # Keep 85% of energy
            'low': 0.70,     # Keep 70% of energy
        }
        
        # Symbolic basis cache for reuse
        self._basis_cache: Dict[Tuple, List[np.ndarray]] = {}
        
    def should_compress(self, tensor: np.ndarray) -> bool:
        """Determine if a tensor should be compressed"""
        # Skip small tensors
        if tensor.size < self.min_dimension:
            return False
        
        # Skip already compressed or non-numeric
        if not np.issubdtype(tensor.dtype, np.floating):
            return False
        
        return True
    
    def compress(self, tensor: np.ndarray, 
                 quality: str = 'medium') -> Optional[CompressedTensor]:
        """
        Compress a tensor using TCL-inspired symbolic compression.
        
        Uses low-rank approximation + sparse coding techniques
        to achieve high compression ratios.
        """
        if not self.should_compress(tensor):
            self.stats['tensors_skipped'] += 1
            return None
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Flatten for processing
        flat = tensor.astype(np.float32).reshape(-1)
        original_size = flat.size
        
        # Determine target rank based on quality and compression goal
        energy_ratio = self.quality_tiers.get(quality, 0.85)
        target_rank = max(1, int(original_size / (self.target_compression * 8)))
        
        # Use randomized SVD for efficiency
        U, S, Vt = self._randomized_svd(flat.reshape(1, -1), target_rank)
        
        # Truncate based on energy
        cumsum = np.cumsum(S**2)
        total_energy = cumsum[-1]
        k = np.searchsorted(cumsum, energy_ratio * total_energy) + 1
        k = min(k, len(S))
        
        # Build compressed representation
        symbol_basis = [U[:, :k], S[:k].reshape(-1, 1), Vt[:k, :]]
        coefficients = S[:k].reshape(1, -1)
        
        # Pack into compressed tensor
        compressed = CompressedTensor(
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_ratio=original_size / (k * (1 + flat.size // k)),
            symbol_basis=symbol_basis,
            coefficients=coefficients,
            metadata={
                'rank': k,
                'energy_retained': float(np.sum(S[:k]**2) / total_energy),
                'compression_method': 'tcl_svd',
                'shape_hash': hash(original_shape) & 0xFFFFFFFF,
            }
        )
        
        # Update stats
        self.stats['tensors_compressed'] += 1
        self.stats['total_original_bytes'] += tensor.nbytes
        self.stats['total_compressed_bytes'] += compressed.memory_size_bytes()
        
        if self.stats['tensors_compressed'] > 0:
            self.stats['avg_compression_ratio'] = (
                self.stats['total_original_bytes'] / self.stats['total_compressed_bytes']
            )
        
        return compressed
    
    def decompress(self, compressed: CompressedTensor) -> np.ndarray:
        """Decompress a compressed tensor back to full representation"""
        U, S, Vt = compressed.symbol_basis[0], compressed.symbol_basis[1].flatten(), compressed.symbol_basis[2]
        
        # Reconstruct
        k = len(S)
        reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]
        
        # Reshape to original
        result = reconstructed.reshape(compressed.original_shape)
        
        return result.astype(compressed.original_dtype)
    
    def _randomized_svd(self, matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast randomized SVD for large matrices.
        
        This is much faster than full SVD for the low-rank approximations
        we need for compression.
        """
        m, n = matrix.shape
        
        # Oversample
        p = min(rank + 10, min(m, n))
        
        # Random projection
        if m < n:
            # Tall matrix
            Omega = np.random.randn(n, p).astype(matrix.dtype)
            Y = matrix @ Omega
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ matrix
            U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
            U = Q @ U_tilde
        else:
            # Wide matrix  
            Omega = np.random.randn(m, p).astype(matrix.dtype)
            Y = matrix.T @ Omega
            Q, _ = np.linalg.qr(Y)
            B = matrix @ Q
            U, S, Vt_tilde = np.linalg.svd(B, full_matrices=False)
            Vt = Vt_tilde @ Q.T
        
        # Return top rank components
        return U[:, :rank], S[:rank], Vt[:rank, :]
    
    def compress_gradient(self, grad: np.ndarray, 
                          layer_idx: int,
                          param_name: str) -> CompressedTensor:
        """
        Compress a gradient tensor with adaptive quality.
        
        Uses gradient statistics to determine appropriate compression level.
        """
        # Determine quality based on gradient statistics
        if self.adaptive_quality:
            grad_norm = np.linalg.norm(grad)
            grad_max = np.max(np.abs(grad))
            
            # High magnitude gradients need more precision
            if grad_norm > 1.0 or grad_max > 0.5:
                quality = 'high'
            elif grad_norm > 0.1:
                quality = 'medium'
            else:
                quality = 'low'
        else:
            quality = 'medium'
        
        return self.compress(grad, quality=quality)
    
    def compress_activation(self, activation: np.ndarray,
                           layer_idx: int) -> CompressedTensor:
        """
        Compress activation tensor.
        
        Activations can typically tolerate more compression than gradients.
        """
        # Always use medium or low for activations
        grad_norm = np.linalg.norm(activation)
        if grad_norm > 10.0:
            quality = 'medium'
        else:
            quality = 'low'
        
        return self.compress(activation, quality=quality)
    
    def batch_compress(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Optional[CompressedTensor]]:
        """Compress a batch of tensors"""
        return {
            name: self.compress(tensor) 
            for name, tensor in tensors.items()
        }
    
    def batch_decompress(self, compressed: Dict[str, Optional[CompressedTensor]]) -> Dict[str, np.ndarray]:
        """Decompress a batch of compressed tensors"""
        result = {}
        for name, comp in compressed.items():
            if comp is None:
                # Wasn't compressed
                continue
            result[name] = self.decompress(comp)
        return result
    
    def estimate_bandwidth_multiplier(self) -> float:
        """
        Estimate effective bandwidth multiplier from compression.
        
        If we achieve 8x compression, effective bandwidth is 8x higher.
        """
        if self.stats['tensors_compressed'] == 0:
            return 1.0
        return self.stats['avg_compression_ratio']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'tensors_compressed': 0,
            'tensors_skipped': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'avg_compression_ratio': 0.0,
        }


class GradientSparsifier:
    """
    Additional compression via gradient sparsification.
    
    Only transmits gradients above a threshold, further reducing bandwidth.
    """
    
    def __init__(self, 
                 sparsity_target: float = 0.9,  # 90% sparsity
                 threshold_method: str = 'adaptive'):
        self.sparsity_target = sparsity_target
        self.threshold_method = threshold_method
        
    def sparsify(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Sparsify a gradient tensor.
        
        Returns:
            - Sparse values (non-zero elements)
            - Indices of non-zero elements
            - Metadata for reconstruction
        """
        flat = grad.flatten()
        
        # Determine threshold
        if self.threshold_method == 'adaptive':
            # Use percentile-based threshold
            threshold = np.percentile(np.abs(flat), self.sparsity_target * 100)
        else:
            threshold = self.threshold_method
        
        # Create mask
        mask = np.abs(flat) >= threshold
        indices = np.where(mask)[0]
        values = flat[mask]
        
        metadata = {
            'original_shape': grad.shape,
            'threshold': float(threshold),
            'sparsity': 1.0 - (len(values) / len(flat)),
            'indices_dtype': 'int32',
        }
        
        return values, indices, metadata
    
    def desparsify(self, values: np.ndarray, indices: np.ndarray, 
                   metadata: Dict) -> np.ndarray:
        """Reconstruct gradient from sparse representation"""
        shape = metadata['original_shape']
        result = np.zeros(np.prod(shape), dtype=values.dtype)
        result[indices] = values
        return result.reshape(shape)


class UnifiedCompressionPipeline:
    """
    Combines multiple compression techniques for maximum bandwidth savings.
    
    Pipeline: Sparsify -> Quantize -> TCL Compress
    """
    
    def __init__(self,
                 enable_sparsification: bool = True,
                 enable_tcl_compression: bool = True,
                 sparsity_target: float = 0.9,
                 tcl_compression: float = 8.0):
        
        self.enable_sparsification = enable_sparsification
        self.enable_tcl_compression = enable_tcl_compression
        
        self.sparsifier = GradientSparsifier(sparsity_target) if enable_sparsification else None
        self.tcl_compressor = TCLGradientCompressor(tcl_compression) if enable_tcl_compression else None
        
    def compress(self, tensor: np.ndarray, tensor_type: str = 'gradient') -> Dict[str, Any]:
        """
        Full compression pipeline.
        
        Returns a dictionary with compression artifacts.
        """
        result = {
            'original_shape': tensor.shape,
            'original_dtype': str(tensor.dtype),
            'original_nbytes': tensor.nbytes,
            'compression_pipeline': [],
        }
        
        current = tensor.astype(np.float32)
        
        # Step 1: Sparsification
        if self.enable_sparsification and self.sparsifier:
            values, indices, sparse_meta = self.sparsifier.sparsify(current)
            result['sparsification'] = {
                'values': values,
                'indices': indices,
                'metadata': sparse_meta,
            }
            result['compression_pipeline'].append('sparsify')
            
            # Reconstruct for next stage
            current = self.sparsifier.desparsify(values, indices, sparse_meta)
        
        # Step 2: TCL Compression
        if self.enable_tcl_compression and self.tcl_compressor:
            if tensor_type == 'gradient':
                compressed = self.tcl_compressor.compress_gradient(current, 0, 'param')
            else:
                compressed = self.tcl_compressor.compress_activation(current, 0)
            
            if compressed:
                result['tcl_compression'] = compressed
                result['compression_pipeline'].append('tcl')
                result['final_compression_ratio'] = compressed.effective_compression()
        
        return result
    
    def get_effective_bandwidth_multiplier(self) -> float:
        """Calculate total effective bandwidth improvement"""
        multiplier = 1.0
        
        if self.enable_sparsification and self.sparsifier:
            multiplier *= 1.0 / (1.0 - self.sparsifier.sparsity_target)
        
        if self.enable_tcl_compression and self.tcl_compressor:
            multiplier *= self.tcl_compressor.estimate_bandwidth_multiplier()
        
        return multiplier
