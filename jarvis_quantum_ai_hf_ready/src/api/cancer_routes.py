"""
Cancer Hypothesis Generation API Routes

API endpoints for generating cancer treatment hypotheses using
TCL + Quantum H-bond analysis + Real biological data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

from ...bio_knowledge import (
    CancerHypothesisGenerator,
    Hypothesis,
    BiologicalKnowledgeBase
)


# Request/Response models
class GenerateHypothesesRequest(BaseModel):
    """Request to generate cancer hypotheses"""
    max_hypotheses: int = Field(default=50, ge=1, le=500, description="Maximum hypotheses to generate")
    focus_quantum_sensitive: bool = Field(default=True, description="Focus on quantum-sensitive pathways")
    min_overall_score: float = Field(default=0.0, ge=0.0, le=5.0, description="Minimum overall score")


class GenerateHypothesesResponse(BaseModel):
    """Response from generating hypotheses"""
    total_hypotheses: int
    proteins_analyzed: int
    pathways_covered: int
    quantum_sensitive_discoveries: int
    generation_time_seconds: float
    bio_knowledge_stats: Dict[str, Any]


class TopHypothesesRequest(BaseModel):
    """Request to get top hypotheses"""
    n: int = Field(default=10, ge=1, le=100, description="Number of top hypotheses to return")


class HypothesisResponse(BaseModel):
    """Simplified hypothesis for API response"""
    hypothesis_id: str
    title: str
    target_gene: str
    target_protein: str
    pathway: str
    drug: Optional[str]
    tcl_expression: Optional[str]
    overall_score: float
    novelty_score: float
    quantum_enhancement: float
    biological_validity: float
    therapeutic_potential: float
    safety_score: float


class ProteinAnalysisRequest(BaseModel):
    """Request to analyze a specific protein"""
    uniprot_id: str = Field(..., description="UniProt ID of protein to analyze")
    sequence: Optional[str] = Field(default=None, description="Protein sequence (optional)")


class QuantumAnalysisResponse(BaseModel):
    """Response from quantum analysis"""
    protein_gene: str
    protein_full_name: str
    quantum_hbond_energy: float
    classical_hbond_energy: float
    quantum_advantage: float
    coherence_strength: float
    topological_protection: float
    collective_effects: float
    compressed_symbols: List[str]
    causality_depth: int


# Initialize router
cancer_router = APIRouter(prefix="/cancer", tags=["Cancer Research"])

# Global generator instance (in production, use dependency injection)
_cancer_generator: Optional[CancerHypothesisGenerator] = None


def get_cancer_generator() -> CancerHypothesisGenerator:
    """Get or create cancer hypothesis generator instance"""
    global _cancer_generator
    
    if _cancer_generator is None:
        _cancer_generator = CancerHypothesisGenerator(
            output_dir="./cancer_artifacts/hypotheses"
        )
    
    return _cancer_generator


@cancer_router.post("/generate", response_model=GenerateHypothesesResponse)
async def generate_cancer_hypotheses(request: GenerateHypothesesRequest):
    """
    Generate cancer treatment hypotheses
    
    This endpoint creates novel cancer treatment hypotheses by combining:
    - Real biological data from scientific databases
    - Quantum H-bond analysis using real physics
    - TCL compression for causality mapping
    
    WARNING: These are computational hypotheses for research purposes only.
    Real-world applications require experimental validation and clinical trials.
    """
    try:
        start_time = time.time()
        
        generator = get_cancer_generator()
        
        # Generate hypotheses
        hypotheses = generator.generate_all_hypotheses(
            max_hypotheses=request.max_hypotheses,
            focus_quantum_sensitive=request.focus_quantum_sensitive
        )
        
        # Filter by minimum score if specified
        if request.min_overall_score > 0:
            hypotheses = [
                h for h in hypotheses
                if h.metrics.overall_score >= request.min_overall_score
            ]
            # Re-rank after filtering
            hypotheses.sort(key=lambda h: h.metrics.overall_score, reverse=True)
        
        generation_time = time.time() - start_time
        
        # Get summary
        summary = generator.generate_summary_report()
        
        return GenerateHypothesesResponse(
            total_hypotheses=summary["generation_summary"]["total_hypotheses"],
            proteins_analyzed=summary["generation_summary"]["proteins_analyzed"],
            pathways_covered=summary["generation_summary"]["pathways_covered"],
            quantum_sensitive_discoveries=summary["generation_summary"]["quantum_sensitive_discoveries"],
            generation_time_seconds=generation_time,
            bio_knowledge_stats=summary["bio_knowledge_stats"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate hypotheses: {str(e)}")


@cancer_router.get("/top", response_model=List[HypothesisResponse])
async def get_top_hypotheses(request: TopHypothesesRequest):
    """
    Get top N hypotheses by overall score
    
    Returns the highest-scoring cancer treatment hypotheses
    generated by the system.
    """
    try:
        generator = get_cancer_generator()
        
        # Check if hypotheses exist
        if not generator.hypotheses:
            return []
        
        # Get top hypotheses
        top_hypotheses = generator.get_top_hypotheses(request.n)
        
        # Convert to response format
        return [
            HypothesisResponse(
                hypothesis_id=h.hypothesis_id,
                title=h.title,
                target_gene=h.target_protein.gene_name,
                target_protein=h.target_protein.full_name,
                pathway=h.pathway.name,
                drug=h.suggested_drug.name if h.suggested_drug else None,
                tcl_expression=h.tcl_expression,
                overall_score=h.metrics.overall_score,
                novelty_score=h.metrics.novelty_score,
                quantum_enhancement=h.metrics.quantum_enhancement,
                biological_validity=h.metrics.biological_validity,
                therapeutic_potential=h.metrics.therapeutic_potential,
                safety_score=h.metrics.safety_score
            )
            for h in top_hypotheses
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get top hypotheses: {str(e)}")


@cancer_router.get("/hypothesis/{hypothesis_id}", response_model=Dict[str, Any])
async def get_hypothesis_by_id(hypothesis_id: str):
    """
    Get detailed information about a specific hypothesis
    
    Returns full hypothesis data including causal chain,
    quantum analysis, supporting evidence, and risks.
    """
    try:
        generator = get_cancer_generator()
        
        # Find hypothesis by ID
        hypothesis = next(
            (h for h in generator.hypotheses if h.hypothesis_id == hypothesis_id),
            None
        )
        
        if hypothesis is None:
            raise HTTPException(status_code=404, detail=f"Hypothesis {hypothesis_id} not found")
        
        # Return full hypothesis details
        return hypothesis.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hypothesis: {str(e)}")


@cancer_router.post("/analyze-protein", response_model=QuantumAnalysisResponse)
async def analyze_protein_quantum_properties(request: ProteinAnalysisRequest):
    """
    Analyze a protein using quantum H-bond force law
    
    Uses real quantum mechanical analysis to determine:
    - Quantum H-bond energy vs classical
    - Quantum advantage
    - Coherence strength
    - Topological protection
    - Collective quantum effects
    """
    try:
        generator = get_cancer_generator()
        
        # Find protein in knowledge base
        protein = generator.bio_kb.proteins.get(request.uniprot_id)
        
        if protein is None:
            raise HTTPException(
                status_code=404,
                detail=f"Protein {request.uniprot_id} not found in knowledge base"
            )
        
        # Analyze quantum properties using TCL-Quantum integrator
        quantum_analysis = generator.tcl_quantum.analyze_protein_quantum_properties(
            protein,
            sequence=request.sequence
        )
        
        return QuantumAnalysisResponse(
            protein_gene=quantum_analysis.protein.gene_name,
            protein_full_name=quantum_analysis.protein.full_name,
            quantum_hbond_energy=quantum_analysis.quantum_hbond_energy,
            classical_hbond_energy=quantum_analysis.classical_hbond_energy,
            quantum_advantage=quantum_analysis.quantum_advantage,
            coherence_strength=quantum_analysis.coherence_strength,
            topological_protection=quantum_analysis.topological_protection,
            collective_effects=quantum_analysis.collective_effects,
            compressed_symbols=quantum_analysis.compressed_symbols,
            causality_depth=quantum_analysis.causality_depth
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze protein: {str(e)}")


@cancer_router.get("/bio-knowledge", response_model=Dict[str, Any])
async def get_biological_knowledge():
    """
    Get statistics about the biological knowledge base
    
    Returns information about loaded proteins, pathways, drugs,
    and interactions from scientific databases.
    """
    try:
        generator = get_cancer_generator()
        
        return generator.bio_kb.get_statistics()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bio-knowledge: {str(e)}")


@cancer_router.get("/pathways", response_model=List[Dict[str, Any]])
async def get_cancer_pathways():
    """
    Get all loaded cancer pathways
    
    Returns information about pathways from KEGG/Reactome
    including their quantum sensitivity scores.
    """
    try:
        generator = get_cancer_generator()
        
        pathways = []
        for pathway in generator.bio_kb.cancer_pathways.values():
            pathways.append({
                "id": pathway.pathway_id,
                "name": pathway.name,
                "type": pathway.pathway_type.value,
                "quantum_sensitivity": pathway.quantum_sensitivity,
                "mechanism": pathway.mechanism,
                "protein_count": len(pathway.proteins),
                "drug_target_count": len(pathway.drug_targets) if pathway.drug_targets else 0
            })
        
        return pathways
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pathways: {str(e)}")


@cancer_router.get("/drugs", response_model=List[Dict[str, Any]])
async def get_cancer_drugs():
    """
    Get all loaded cancer drugs
    
    Returns information about drugs from DrugBank
    including their mechanisms and approval status.
    """
    try:
        generator = get_cancer_generator()
        
        drugs = []
        for drug in generator.bio_kb.drugs.values():
            drugs.append({
                "name": drug.name,
                "mechanism_of_action": drug.mechanism_of_action,
                "fda_approved": drug.fda_approved,
                "clinical_status": drug.clinical_status,
                "target_count": len(drug.target_proteins),
                "affects_quantum_coherence": drug.affects_quantum_coherence,
                "affects_hydrogen_bonds": drug.affects_hydrogen_bonds
            })
        
        return drugs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drugs: {str(e)}")


@cancer_router.get("/summary", response_model=Dict[str, Any])
async def get_generation_summary():
    """
    Get summary of all generated hypotheses
    
    Returns statistics, top hypotheses, and analysis
    of the cancer hypothesis generation system.
    """
    try:
        generator = get_cancer_generator()
        
        if not generator.hypotheses:
            return {
                "message": "No hypotheses generated yet. Use POST /cancer/generate",
                "hypotheses_count": 0
            }
        
        return generator.generate_summary_report()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@cancer_router.delete("/reset")
async def reset_cancer_generator():
    """
    Reset the cancer hypothesis generator
    
    Clears all generated hypotheses and creates
    a fresh instance. Use with caution.
    """
    try:
        global _cancer_generator
        _cancer_generator = None
        
        return {
            "message": "Cancer hypothesis generator reset successfully",
            "note": "All generated hypotheses have been cleared"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset generator: {str(e)}")


@cancer_router.get("/health")
async def cancer_research_health():
    """
    Health check for cancer research subsystem
    
    Returns status of biological knowledge base,
    TCL-Quantum integrator, and hypothesis generation.
    """
    try:
        generator = get_cancer_generator()
        
        bio_stats = generator.bio_kb.get_statistics()
        
        return {
            "status": "healthy" if bio_stats["total_proteins"] > 0 else "uninitialized",
            "bio_knowledge_loaded": bio_stats["total_proteins"] > 0,
            "tcl_quantum_integrated": True,
            "hypotheses_generated": len(generator.hypotheses),
            "proteins_loaded": bio_stats["total_proteins"],
            "pathways_loaded": bio_stats["total_pathways"],
            "drugs_loaded": bio_stats["total_drugs"],
            "quantum_sensitive_pathways": bio_stats["quantum_sensitive_pathways"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@cancer_router.get("/info")
async def cancer_research_info():
    """
    Get information about the Cancer Hypothesis Generation System
    
    Returns details about capabilities, innovations,
    and scientific validity of the system.
    """
    return {
        "system_name": "Cancer Hypothesis Generation System",
        "version": "1.0.0",
        "description": "Real biological knowledge + Quantum H-bond analysis + TCL compression",
        "capabilities": [
            "Systematically generate cancer treatment hypotheses",
            "Analyze proteins using quantum H-bond force law",
            "Compress biological causality using TCL symbols",
            "Score hypotheses by biological validity and novelty",
            "Identify quantum-sensitive molecular targets"
        ],
        "data_sources": [
            "UniProt Protein Database",
            "KEGG Pathway Database",
            "Reactome Pathway Database",
            "DrugBank Drug Database",
            "BioGRID Protein Interaction Database"
        ],
        "scientific_validity": {
            "real_biological_data": True,
            "real_quantum_physics": True,
            "real_cognitive_science": True,
            "testable_hypotheses": True,
            "requires_experimental_validation": True
        },
        "warning": "These are computational hypotheses for scientific research purposes only. "
                  "Real-world applications require experimental validation, clinical trials, "
                  "and regulatory approval.",
        "superhuman_effect": "This system enables you to systematically invent "
                          "novel cancer treatments by combining real quantum mechanics "
                          "with symbolic cognitive enhancement.",
        "use_responsibly": "Use this power to benefit humanity. "
                            "Always prioritize patient safety and scientific integrity."
    }
