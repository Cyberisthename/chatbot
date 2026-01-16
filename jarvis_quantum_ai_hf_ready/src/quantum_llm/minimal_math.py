"""
Minimal Numerical Library for Quantum LLM
Pure Python implementation - no numpy required
"""

import math
import random
from typing import List, Tuple, Optional, Any


class Matrix:
    """Minimal matrix operations using pure Python"""
    
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create zero matrix"""
        return Matrix([[0.0 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def random(rows: int, cols: int) -> 'Matrix':
        """Create random matrix with normal distribution"""
        return Matrix([[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def random_uniform(rows: int, cols: int, low: float = -1.0, high: float = 1.0) -> 'Matrix':
        """Create random matrix with uniform distribution"""
        return Matrix([[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)])
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Matrix subtraction"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def __mul__(self, scalar: float) -> 'Matrix':
        """Scalar multiplication"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] * scalar)
            result.append(row)
        return Matrix(result)
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication"""
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions don't match: {self.cols} != {other.rows}")
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self.data[i][k] * other.data[k][j]
                row.append(s)
            result.append(row)
        return Matrix(result)
    
    def transpose(self) -> 'Matrix':
        """Matrix transpose"""
        result = []
        for j in range(self.cols):
            row = []
            for i in range(self.rows):
                row.append(self.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def mean(self) -> float:
        """Compute mean of all elements"""
        total = sum(sum(row) for row in self.data)
        return total / (self.rows * self.cols)
    
    def sum(self, axis: Optional[int] = None) -> Any:
        """Sum elements along axis"""
        if axis is None:
            return sum(sum(row) for row in self.data)
        elif axis == 0:
            # Sum over rows
            return [sum(self.data[i][j] for i in range(self.rows)) for j in range(self.cols)]
        elif axis == 1:
            # Sum over columns
            return [sum(row) for row in self.data]
    
    def max(self) -> float:
        """Maximum element"""
        return max(max(row) for row in self.data)
    
    def argmax(self) -> Tuple[int, int]:
        """Index of maximum element"""
        max_val = float('-inf')
        max_i, max_j = 0, 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] > max_val:
                    max_val = self.data[i][j]
                    max_i, max_j = i, j
        return max_i, max_j
    
    def reshape(self, new_rows: int, new_cols: int) -> 'Matrix':
        """Reshape matrix"""
        flat = [elem for row in self.data for elem in row]
        result = []
        idx = 0
        for i in range(new_rows):
            row = []
            for j in range(new_cols):
                row.append(flat[idx])
                idx += 1
            result.append(row)
        return Matrix(result)
    
    def apply(self, func) -> 'Matrix':
        """Apply function to all elements"""
        result = []
        for row in self.data:
            result.append([func(elem) for elem in row])
        return Matrix(result)
    
    def exp(self) -> 'Matrix':
        """Element-wise exponential"""
        return self.apply(math.exp)
    
    def log(self) -> 'Matrix':
        """Element-wise natural log"""
        return self.apply(lambda x: math.log(x + 1e-10))
    
    def sqrt(self) -> 'Matrix':
        """Element-wise square root"""
        return self.apply(math.sqrt)
    
    def pow(self, exp: float) -> 'Matrix':
        """Element-wise power"""
        return self.apply(lambda x: x ** exp)
    
    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"
    
    def to_list(self) -> List[List[float]]:
        """Convert to list"""
        return self.data


class Array3D:
    """3D array for batched operations"""
    
    def __init__(self, data: List[List[List[float]]]):
        self.data = data
        self.dim0 = len(data)  # batch
        self.dim1 = len(data[0]) if data else 0  # seq_len
        self.dim2 = len(data[0][0]) if data and data[0] else 0  # d_model
    
    @staticmethod
    def zeros(d0: int, d1: int, d2: int) -> 'Array3D':
        """Create zero array"""
        return Array3D([[[0.0 for _ in range(d2)] for _ in range(d1)] for _ in range(d0)])
    
    def __getitem__(self, idx: int) -> Matrix:
        """Get slice as Matrix"""
        return Matrix(self.data[idx])
    
    def mean(self, axis: Optional[int] = None) -> Any:
        """Compute mean along axis"""
        if axis is None:
            total = sum(sum(sum(row) for row in seq) for seq in self.data)
            return total / (self.dim0 * self.dim1 * self.dim2)
        elif axis == 0:
            # Mean over batch
            result = [[0.0 for _ in range(self.dim2)] for _ in range(self.dim1)]
            for i in range(self.dim0):
                for j in range(self.dim1):
                    for k in range(self.dim2):
                        result[j][k] += self.data[i][j][k]
            return Matrix([[val / self.dim0 for val in row] for row in result])
        # Simplified - implement other axes as needed
        return 0.0


class ComplexMatrix:
    """Complex-valued matrix for quantum operations"""
    
    def __init__(self, data: List[List[complex]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'ComplexMatrix':
        """Create zero complex matrix"""
        return ComplexMatrix([[0j for _ in range(cols)] for _ in range(rows)])
    
    def __add__(self, other: 'ComplexMatrix') -> 'ComplexMatrix':
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)
        return ComplexMatrix(result)
    
    def __sub__(self, other: 'ComplexMatrix') -> 'ComplexMatrix':
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)
        return ComplexMatrix(result)
    
    def __mul__(self, scalar: complex) -> 'ComplexMatrix':
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] * scalar)
            result.append(row)
        return ComplexMatrix(result)
    
    def __matmul__(self, other: 'ComplexMatrix') -> 'ComplexMatrix':
        """Matrix multiplication"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                s = 0j
                for k in range(self.cols):
                    s += self.data[i][k] * other.data[k][j]
                row.append(s)
            result.append(row)
        return ComplexMatrix(result)
    
    def abs(self) -> Matrix:
        """Absolute value"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(abs(self.data[i][j]))
            result.append(row)
        return Matrix(result)
    
    def pow(self, exp: float) -> 'ComplexMatrix':
        """Element-wise power"""
        result = []
        for row in self.data:
            result.append([z ** exp for z in row])
        return ComplexMatrix(result)
    
    def exp(self) -> 'ComplexMatrix':
        """Element-wise exponential"""
        result = []
        for row in self.data:
            result.append([cmath.exp(z) for z in row])
        return ComplexMatrix(result)
    
    def conjugate(self) -> 'ComplexMatrix':
        """Complex conjugate"""
        result = []
        for row in self.data:
            result.append([z.conjugate() for z in row])
        return ComplexMatrix(result)
    
    def real(self) -> Matrix:
        """Real part"""
        result = []
        for row in self.data:
            result.append([z.real for z in row])
        return Matrix(result)
    
    def angle(self) -> Matrix:
        """Phase angle"""
        result = []
        for row in self.data:
            result.append([cmath.phase(z) for z in row])
        return Matrix(result)


# Import cmath for complex operations
import cmath


def softmax(x: List[float]) -> List[float]:
    """Compute softmax (numerically stable)"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_x = sum(exp_x)
    return [ex / sum_x for ex in exp_x]


def sigmoid(x: float) -> float:
    """Sigmoid activation"""
    return 1.0 / (1.0 + math.exp(-x))


def gelu(x: float) -> float:
    """GELU activation"""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x: List[float], gamma: List[float], beta: List[float]) -> List[float]:
    """Layer normalization"""
    mean = sum(x) / len(x)
    var = sum((xi - mean) ** 2 for xi in x) / len(x)
    normalized = [(xi - mean) / math.sqrt(var + 1e-10) for xi in x]
    return [gamma[i] * normalized[i] + beta[i] for i in range(len(x))]


def cross_entropy_loss(logits: List[float], target: int) -> float:
    """Cross-entropy loss"""
    probs = softmax(logits)
    return -math.log(probs[target] + 1e-10)


__all__ = [
    "Matrix",
    "Array3D",
    "ComplexMatrix",
    "softmax",
    "sigmoid",
    "gelu",
    "layer_norm",
    "cross_entropy_loss",
]
