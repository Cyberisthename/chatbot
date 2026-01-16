"""
TCL Types - Shared types and dataclasses for Thought-Compression Language

This module contains shared types and dataclasses used across the TCL system
to avoid circular import dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

@dataclass
class CognitiveMetrics:
    """Metrics tracking cognitive enhancement levels"""
    compression_ratio: float = 0.0
    conceptual_density: float = 0.0
    causality_depth: int = 0
    cognitive_load: float = 0.0
    thinking_speed: float = 0.0
    abstract_reasoning_score: float = 0.0

@dataclass 
class TCLExecutionContext:
    """Context for TCL execution"""
    symbols: Any = field(default=None)  # Will be set to ConceptGraph
    causality: Any = field(default=None)  # Will be set to CausalityMap
    metrics: CognitiveMetrics = field(default_factory=CognitiveMetrics)
    session_id: str = ""
    user_cognitive_level: float = 0.5  # 0.0 = novice, 1.0 = master
    processing_thread: Optional[Any] = None
    active: bool = False

    def __post_init__(self):
        # Import here to avoid circular imports
        from .tcl_symbols import ConceptGraph, CausalityMap
        
        if self.symbols is None:
            self.symbols = ConceptGraph()
        if self.causality is None:
            self.causality = CausalityMap()

class RuntimeState(Enum):
    """Runtime execution states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    ERROR = "error"

@dataclass
class ExecutionStack:
    """Stack for TCL runtime execution"""
    stack: List[Any] = field(default_factory=list)
    max_size: int = 1000
    
    def push(self, value: Any):
        if len(self.stack) >= self.max_size:
            raise RuntimeError("Stack overflow")
        self.stack.append(value)
    
    def pop(self) -> Any:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()
    
    def peek(self) -> Any:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack[-1]
    
    def size(self) -> int:
        return len(self.stack)
    
    def clear(self):
        self.stack.clear()

@dataclass
class SymbolStack:
    """Stack for symbol operations"""
    symbols: List[Any] = field(default_factory=list)  # Will be TCLSymbol
    
    def push(self, symbol: Any):
        self.symbols.append(symbol)
    
    def pop(self) -> Any:
        if not self.symbols:
            raise RuntimeError("Symbol stack underflow")
        return self.symbols.pop()
    
    def peek(self) -> Any:
        if not self.symbols:
            raise RuntimeError("Symbol stack underflow")
        return self.symbols[-1]
    
    def size(self) -> int:
        return len(self.symbols)
    
    def clear(self):
        self.symbols.clear()