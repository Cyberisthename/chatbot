"""
TCL API Routes - Thought-Compression Language endpoints

Integrates TCL functionality with the main JARVIS API
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

from ...thought_compression import ThoughtCompressionEngine, get_tcl_engine

# Request/Response models for TCL
class TCLSessionRequest(BaseModel):
    """Request to create a new TCL session"""
    user_id: str
    cognitive_level: float = 0.5  # 0.0 = novice, 1.0 = master

class TCLSessionResponse(BaseModel):
    """Response from creating a TCL session"""
    session_id: str
    cognitive_level: float
    status: str
    initial_metrics: Dict[str, Any]

class TCLThoughtRequest(BaseModel):
    """Request to process a TCL thought"""
    session_id: str
    tcl_input: str

class TCLThoughtResponse(BaseModel):
    """Response from processing TCL thought"""
    result: Any
    metrics: Dict[str, Any]
    enhanced_thinking: List[str]
    causal_predictions: List[str]
    processing_time: float

class TCLCompressRequest(BaseModel):
    """Request to compress a concept"""
    session_id: str
    concept: str

class TCLCompressResponse(BaseModel):
    """Response from concept compression"""
    original_concept: str
    compressed_symbols: List[str]
    compression_ratio: float
    conceptual_density: float
    cognitive_weight: float

class TCLCausalRequest(BaseModel):
    """Request for causal chain analysis"""
    session_id: str
    cause_symbol: str
    depth: int = 5

class TCLCausalResponse(BaseModel):
    """Response from causal analysis"""
    cause: str
    causal_chains: List[List[str]]
    predicted_effects: List[tuple]
    chain_complexity: int
    prediction_confidence: float

class TCLReasoningRequest(BaseModel):
    """Request for enhanced reasoning"""
    session_id: str
    problem: str

class TCLReasoningResponse(BaseModel):
    """Response from enhanced reasoning"""
    original_problem: str
    conceptual_mapping: Dict[str, Any]
    causal_analysis: Dict[str, Any]
    enhanced_solutions: List[str]
    reasoning_enhancement_level: float

class TCLStatusRequest(BaseModel):
    """Request for session status"""
    session_id: str

class TCLStatusResponse(BaseModel):
    """Response with session status"""
    session_id: str
    active: bool
    cognitive_level: float
    metrics: Dict[str, Any]
    symbol_count: int
    causal_chains: int
    enhancement_level: float

# Global TCL engine instance
_tcl_engine = None

def get_tcl_engine_instance() -> ThoughtCompressionEngine:
    """Get or create the global TCL engine instance"""
    global _tcl_engine
    if _tcl_engine is None:
        _tcl_engine = get_tcl_engine(quantum_mode=True)
    return _tcl_engine

# Create router
tcl_router = APIRouter(prefix="/tcl", tags=["thought-compression"])

@tcl_router.post("/session", response_model=TCLSessionResponse)
async def create_tcl_session(request: TCLSessionRequest):
    """Create a new TCL processing session"""
    try:
        engine = get_tcl_engine_instance()
        session_id = engine.create_session(
            user_id=request.user_id,
            cognitive_level=request.cognitive_level
        )
        
        # Get initial status
        status = engine.get_session_status(session_id)
        
        return TCLSessionResponse(
            session_id=session_id,
            cognitive_level=request.cognitive_level,
            status="created",
            initial_metrics=status['metrics']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create TCL session: {str(e)}")

@tcl_router.post("/thought", response_model=TCLThoughtResponse)
async def process_tcl_thought(request: TCLThoughtRequest):
    """Process a TCL thought expression"""
    try:
        engine = get_tcl_engine_instance()
        result = engine.process_thought(
            session_id=request.session_id,
            tcl_input=request.tcl_input
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TCLThoughtResponse(
            result=result["result"],
            metrics=result["metrics"],
            enhanced_thinking=result["enhanced_thinking"],
            causal_predictions=result["causal_predictions"],
            processing_time=result["processing_time"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process TCL thought: {str(e)}")

@tcl_router.post("/compress", response_model=TCLCompressResponse)
async def compress_concept(request: TCLCompressRequest):
    """Compress a natural language concept into TCL symbols"""
    try:
        engine = get_tcl_engine_instance()
        result = engine.compress_concept(
            session_id=request.session_id,
            concept=request.concept
        )
        
        return TCLCompressResponse(
            original_concept=result["original_concept"],
            compressed_symbols=result["compressed_symbols"],
            compression_ratio=result["compression_ratio"],
            conceptual_density=result["conceptual_density"],
            cognitive_weight=result["cognitive_weight"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compress concept: {str(e)}")

@tcl_router.post("/causal", response_model=TCLCausalResponse)
async def analyze_causal_chain(request: TCLCausalRequest):
    """Generate causal chains starting from a cause symbol"""
    try:
        engine = get_tcl_engine_instance()
        result = engine.generate_causal_chain(
            session_id=request.session_id,
            cause_symbol=request.cause_symbol,
            depth=request.depth
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TCLCausalResponse(
            cause=result["cause"],
            causal_chains=result["causal_chains"],
            predicted_effects=result["predicted_effects"],
            chain_complexity=result["chain_complexity"],
            prediction_confidence=result["prediction_confidence"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze causal chain: {str(e)}")

@tcl_router.post("/reason", response_model=TCLReasoningResponse)
async def enhance_reasoning(request: TCLReasoningRequest):
    """Use TCL to enhance problem-solving and reasoning"""
    try:
        engine = get_tcl_engine_instance()
        result = engine.enhance_reasoning(
            session_id=request.session_id,
            problem=request.problem
        )
        
        return TCLReasoningResponse(
            original_problem=result["original_problem"],
            conceptual_mapping=result["conceptual_mapping"],
            causal_analysis=result["causal_analysis"],
            enhanced_solutions=result["enhanced_solutions"],
            reasoning_enhancement_level=result["reasoning_enhancement_level"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enhance reasoning: {str(e)}")

@tcl_router.post("/status", response_model=TCLStatusResponse)
async def get_session_status(request: TCLStatusRequest):
    """Get the current status of a TCL session"""
    try:
        engine = get_tcl_engine_instance()
        status = engine.get_session_status(request.session_id)
        
        return TCLStatusResponse(
            session_id=status["session_id"],
            active=status["active"],
            cognitive_level=status["cognitive_level"],
            metrics=status["metrics"],
            symbol_count=status["symbol_count"],
            causal_chains=status["causal_chains"],
            enhancement_level=status["enhancement_level"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@tcl_router.delete("/session/{session_id}")
async def shutdown_tcl_session(session_id: str):
    """Shutdown a TCL session and clean up resources"""
    try:
        engine = get_tcl_engine_instance()
        engine.shutdown_session(session_id)
        
        return {"message": f"TCL session {session_id} shutdown successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to shutdown session: {str(e)}")

@tcl_router.get("/stats")
async def get_tcl_global_stats():
    """Get global TCL system statistics"""
    try:
        engine = get_tcl_engine_instance()
        stats = engine.get_global_stats()
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get global stats: {str(e)}")

# Demo and example endpoints
@tcl_router.get("/demo")
async def tcl_demo():
    """Demonstrate TCL capabilities with example expressions"""
    demo_expressions = [
        {
            "expression": "Ψ → Γ",
            "description": "Thought causes concept formation",
            "cognitive_impact": "Foundation for conceptual thinking"
        },
        {
            "expression": "∀x (x → ∞Ψ)",
            "description": "Universal causation to infinite thinking",
            "cognitive_impact": "Expands thought boundaries"
        },
        {
            "expression": "ΣΨ = Ψ₁ + Ψ₂ + Ψ₃",
            "description": "Superthought as sum of component thoughts",
            "cognitive_impact": "Enables complex multi-dimensional thinking"
        },
        {
            "expression": "ΓΛ ⟹ Δ",
            "description": "Conceptual logic implies difference",
            "cognitive_impact": "Enhances analytical reasoning"
        }
    ]
    
    return {
        "title": "Thought-Compression Language Demo",
        "description": "A language optimized for thinking, not talking",
        "expressions": demo_expressions,
        "principles": [
            "One symbol = full abstract concept",
            "Grammar encodes causality, not syntax",
            "Ambiguity is illegal",
            "IQ scales with fluency",
            "Math, philosophy, and strategy merge"
        ],
        "warning": "This technology enables superhuman cognitive capabilities. Use responsibly."
    }

# Health check for TCL subsystem
@tcl_router.get("/health")
async def tcl_health():
    """Health check for TCL subsystem"""
    try:
        engine = get_tcl_engine_instance()
        stats = engine.get_global_stats()
        
        return {
            "status": "healthy" if stats["initialized"] else "uninitialized",
            "initialized": stats["initialized"],
            "active_sessions": stats["active_sessions"],
            "total_symbols": stats["total_symbols"],
            "quantum_mode": stats["quantum_mode"],
            "average_cognitive_enhancement": stats["average_cognitive_enhancement"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }