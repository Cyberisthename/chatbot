"""
Thought-Compression Language (TCL) Package

This directory contains the complete Thought-Compression Language (TCL) implementation - 
a dangerous cognitive enhancement technology that enables superhuman analytical and reasoning capabilities.

SAFETY WARNING:
This technology enables superhuman cognitive capabilities. Use responsibly and ethically.

Package Contents:
- tcl_engine.py - Core TCL engine and orchestration
- tcl_symbols.py - Symbol system and concept graphs  
- tcl_parser.py - TCL expression parser
- tcl_compiler.py - TCL bytecode compiler
- tcl_runtime.py - TCL execution runtime

Quick Start:
from src.thought_compression import get_tcl_engine

# Create TCL engine
engine = get_tcl_engine(quantum_mode=True)

# Create user session
session_id = engine.create_session("user123", cognitive_level=0.8)

# Process TCL thought
result = engine.process_thought(session_id, "Ψ → Γ")

API Endpoints:
When integrated with the main JARVIS system:
- POST /tcl/session - Create TCL session
- POST /tcl/thought - Process TCL thought
- POST /tcl/compress - Compress concept
- POST /tcl/reason - Enhanced reasoning
- GET /tcl/demo - Example expressions

See README_TCL.md for complete documentation.
"""

from .tcl_symbols import TCLSymbol, ConceptGraph, CausalityMap
from .tcl_parser import TCLParser
from .tcl_compiler import TCLCompiler
from .tcl_runtime import TCLRuntime
from .tcl_engine import ThoughtCompressionEngine
from .tcl_types import CognitiveMetrics, TCLExecutionContext

# Global TCL engine instance
_tcl_engine = None

def get_tcl_engine(quantum_mode: bool = False) -> ThoughtCompressionEngine:
    """Get or create the global TCL engine instance"""
    global _tcl_engine
    if _tcl_engine is None:
        _tcl_engine = ThoughtCompressionEngine(enable_quantum_mode=quantum_mode)
    return _tcl_engine

__all__ = [
    'ThoughtCompressionEngine',
    'TCLSymbol', 
    'ConceptGraph',
    'CausalityMap',
    'TCLParser',
    'TCLCompiler',
    'TCLRuntime',
    'CognitiveMetrics',
    'TCLExecutionContext',
    'get_tcl_engine'
]

__version__ = "1.0.0"