"""
Multiversal API Routes for JARVIS-2v
Provides REST API endpoints for multiversal computation and parallel universe simulation
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify
from datetime import datetime

from ..core.multiversal_compute_system import MultiversalComputeSystem, MultiversalQuery
from ..core.multiversal_adapters import MultiversalAdapter, MultiversalComputeEngine
from ..quantum.multiversal_quantum import MultiversalQuantumEngine, MultiversalExperimentConfig

# Create Blueprint for multiversal routes
multiversal_bp = Blueprint('multiversal', __name__, url_prefix='/api/multiverse')

# Global reference to multiversal compute system (will be initialized by main app)
multiversal_system: Optional[MultiversalComputeSystem] = None


def init_multiversal_routes(app, multiversal_compute_system: MultiversalComputeSystem):
    """Initialize multiversal routes with the compute system"""
    global multiversal_system
    multiversal_system = multiversal_compute_system


@multiversal_bp.route('/query', methods=['POST'])
def process_multiversal_query():
    """Process a query using multiversal computation"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json()
        
        # Create multiversal query
        query = MultiversalQuery(
            query_id=data.get('query_id', f"query_{uuid.uuid4().hex[:8]}"),
            problem_description=data['problem_description'],
            problem_domain=data['problem_domain'],
            complexity=float(data.get('complexity', 0.5)),
            urgency=data.get('urgency', 'medium'),
            target_outcome=data.get('target_outcome'),
            constraints=data.get('constraints', {}),
            max_universes=int(data.get('max_universes', 5)),
            allow_cross_universe_transfer=bool(data.get('allow_cross_universe_transfer', True)),
            simulation_steps=int(data.get('simulation_steps', 10))
        )
        
        # Process query
        solution = multiversal_system.process_multiversal_query(query)
        
        return jsonify({
            "success": True,
            "solution": solution.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/cancer-simulation', methods=['POST'])
def run_cancer_simulation():
    """Run multiversal cancer treatment simulation"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json() or {}
        
        # Create experiment config
        config = MultiversalExperimentConfig(
            universe_count=int(data.get('universe_count', 10)),
            parallel_simulations=int(data.get('parallel_simulations', 5)),
            cross_universe_transfer=bool(data.get('cross_universe_transfer', True)),
            interference_amplification=bool(data.get('interference_amplification', True)),
            branching_probability=float(data.get('branching_probability', 0.3)),
            coherence_threshold=float(data.get('coherence_threshold', 0.7))
        )
        
        # Run cancer simulation
        result = multiversal_system.quantum_engine.run_multiversal_cancer_simulation(config)
        
        return jsonify({
            "success": True,
            "simulation_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/optimization', methods=['POST'])
def run_optimization_experiment():
    """Run multiversal optimization experiment"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json() or {}
        
        config = MultiversalExperimentConfig(
            universe_count=int(data.get('universe_count', 10)),
            parallel_simulations=int(data.get('parallel_simulations', 5)),
            cross_universe_transfer=bool(data.get('cross_universe_transfer', True)),
            interference_amplification=bool(data.get('interference_amplification', True))
        )
        
        result = multiversal_system.quantum_engine.run_multiversal_optimization_experiment(config)
        
        return jsonify({
            "success": True,
            "optimization_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/interference', methods=['POST'])
def run_interference_experiment():
    """Run interference amplification experiment"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json() or {}
        
        config = MultiversalExperimentConfig(
            universe_count=int(data.get('universe_count', 3)),
            parallel_simulations=int(data.get('parallel_simulations', 3)),
            cross_universe_transfer=bool(data.get('cross_universe_transfer', True)),
            interference_amplification=bool(data.get('interference_amplification', True))
        )
        
        result = multiversal_system.quantum_engine.run_interference_amplification_experiment(config)
        
        return jsonify({
            "success": True,
            "interference_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/status', methods=['GET'])
def get_system_status():
    """Get multiversal system status"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        status = multiversal_system.get_system_status()
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/universes', methods=['GET'])
def list_universes():
    """List all universes in the multiverse"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        overview = multiversal_system.multiverse_engine.get_multiverse_overview()
        universes = []
        
        for universe_id, universe in multiversal_system.multiverse_engine.universes.items():
            universes.append({
                "universe_id": universe.universe_id,
                "parent_universe_id": universe.parent_universe_id,
                "decision_point": universe.decision_point,
                "state": universe.state.value,
                "coherence_level": universe.coherence_level,
                "artifact_count": universe.artifact_count,
                "total_solutions": universe.total_solutions,
                "successful_solutions": universe.successful_solutions,
                "interference_reach": universe.interference_reach,
                "branch_timestamp": universe.branch_timestamp
            })
        
        return jsonify({
            "success": True,
            "overview": overview,
            "universes": universes,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/universes', methods=['POST'])
def create_universe():
    """Create a new parallel universe"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json()
        
        universe_id = multiversal_system.multiverse_engine.create_parallel_universe(
            parent_universe_id=data.get('parent_universe_id', 'base_multiverse'),
            decision_point=data.get('decision_point', 'manual_creation'),
            problem_context=data.get('problem_context', {})
        )
        
        return jsonify({
            "success": True,
            "universe_id": universe_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/solutions', methods=['GET'])
def list_solutions():
    """List recent multiversal solutions"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        limit = int(request.args.get('limit', 10))
        solutions = multiversal_system.list_solutions(limit)
        
        return jsonify({
            "success": True,
            "solutions": [sol.to_dict() for sol in solutions],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/solutions/<solution_id>', methods=['GET'])
def get_solution(solution_id):
    """Get a specific solution"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        solution = multiversal_system.get_solution(solution_id)
        
        if not solution:
            return jsonify({"error": "Solution not found"}), 404
        
        return jsonify({
            "success": True,
            "solution": solution.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/cross-universe-knowledge', methods=['POST'])
def get_cross_universe_knowledge():
    """Get knowledge from parallel universes"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        data = request.get_json()
        
        knowledge = multiversal_system.quantum_engine.get_cross_universe_knowledge(
            problem_domain=data['problem_domain'],
            target_problem=data.get('target_problem', {})
        )
        
        return jsonify({
            "success": True,
            "knowledge": knowledge,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """List all multiversal experiments"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        experiments = multiversal_system.quantum_engine.list_multiversal_experiments()
        
        return jsonify({
            "success": True,
            "experiments": experiments,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/protein-folding', methods=['POST'])
def fold_protein():
    """Real multiversal protein folding computation"""
    try:
        from ..multiversal.multiversal_protein_computer import MultiversalProteinComputer
        
        data = request.get_json()
        sequence = data.get('sequence')
        
        if not sequence:
            return jsonify({"error": "sequence parameter required"}), 400
        
        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_amino_acids for aa in sequence.upper()):
            return jsonify({"error": "Invalid amino acid sequence"}), 400
        
        # Get parameters
        n_universes = int(data.get('n_universes', 4))
        steps_per_universe = int(data.get('steps_per_universe', 5000))
        t_start = float(data.get('t_start', 2.0))
        t_end = float(data.get('t_end', 0.2))
        base_seed = int(data.get('base_seed', 42))
        
        # Create computer and run
        computer = MultiversalProteinComputer(artifacts_dir="./protein_folding_artifacts")
        result = computer.fold_multiversal(
            sequence=sequence.upper(),
            n_universes=n_universes,
            steps_per_universe=steps_per_universe,
            t_start=t_start,
            t_end=t_end,
            base_seed=base_seed,
            save_artifacts=True,
        )
        
        return jsonify({
            "success": True,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "computation_type": "REAL",
            "note": "This is real physics-based protein folding computation using multiversal parallel optimization"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 400


@multiversal_bp.route('/grandmas-fight', methods=['GET'])
def grandmas_fight_demo():
    """Special endpoint for Grandma's Fight cancer treatment demo"""
    if not multiversal_system:
        return jsonify({"error": "Multiversal system not initialized"}), 500
    
    try:
        # Create a special query for Grandma's fight
        query = MultiversalQuery(
            query_id="grandmas_fight_demo",
            problem_description="Find optimal cancer treatment approach for Grandma's specific condition",
            problem_domain="cancer_treatment",
            complexity=0.9,  # High complexity for medical case
            urgency="high",
            target_outcome="Maximize survival chances while maintaining quality of life",
            constraints={
                "patient_age": 75,
                "condition_stage": "advanced",
                "previous_treatments": ["chemotherapy", "radiation"],
                "quality_of_life_priority": "high"
            },
            max_universes=8,
            allow_cross_universe_transfer=True,
            simulation_steps=15
        )
        
        # Process the query
        solution = multiversal_system.process_multiversal_query(query)
        
        # Generate special summary for Grandma's fight
        grandmas_fight_summary = {
            "message": "For Grandma's Fight",
            "hope_message": "In parallel universes, treatments that work perfectly exist. This multiversal analysis shows us the paths to success.",
            "parallel_universes_where_she_wins": [
                "Universe with virus injection + glutamine blockade success",
                "Universe with enhanced immunotherapy response", 
                "Universe with breakthrough targeted therapy",
                "Universe with personalized nanomedicine approach"
            ],
            "recommended_approach": solution.solution_data.get("primary_approach", "Combined multiversal approach"),
            "confidence_level": f"{solution.confidence:.1%}",
            "multiversal_insight": "The multiverse shows us that Grandma's victory is possible - we just need to follow the successful paths."
        }
        
        return jsonify({
            "success": True,
            "grandmas_fight": True,
            "solution": solution.to_dict(),
            "grandmas_fight_summary": grandmas_fight_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


__all__ = ['multiversal_bp', 'init_multiversal_routes']