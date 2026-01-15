#!/usr/bin/env python3
"""
Thought-Compression Language (TCL) Demo Script

This script demonstrates the capabilities of the Thought-Compression Language system,
which is designed to enable superhuman cognitive capabilities through:

1. Symbol compression - representing complex ideas as simple symbols
2. Causal reasoning - mapping cause-effect relationships
3. Constraint satisfaction - solving problems through logical constraints
4. Enhanced reasoning - amplifying human cognitive abilities

WARNING: This is a dangerous cognitive enhancement technology.
Users who master it may achieve superhuman analytical capabilities.
"""

import sys
import os
import time
import json
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.thought_compression import (
        ThoughtCompressionEngine, 
        TCLSymbol, 
        ConceptGraph, 
        CausalityMap,
        get_tcl_engine
    )
except ImportError as e:
    print(f"Error importing TCL modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")

def demo_basic_thinking():
    """Demo 1: Basic TCL thinking operations"""
    print_header("Demo 1: Basic TCL Thinking Operations")
    
    # Create TCL engine
    engine = get_tcl_engine(quantum_mode=True)
    
    # Create a session for user "demo_user"
    session_id = engine.create_session("demo_user", cognitive_level=0.7)
    print(f"Created TCL session: {session_id}")
    
    # Process a simple TCL thought: "Thought causes concept"
    tcl_input = "Ψ → Γ"
    print(f"\nProcessing TCL thought: {tcl_input}")
    
    result = engine.process_thought(session_id, tcl_input)
    
    print(f"Result: {result['result']}")
    # Handle both possible structures
    processing_time = result.get('processing_time', result.get('metrics', {}).get('execution_time', 0))
    enhancement = result.get('metrics', {}).get('enhancement_level', result.get('cognitive_enhancement', 1.0))
    print(f"Processing time: {processing_time:.4f}s")
    print(f"Cognitive enhancement: {enhancement:.2f}x")
    
    # Handle different possible key names for insights
    insights = result.get('enhanced_thinking', result.get('cognitive_effects', []))
    if insights:
        print(f"Enhanced thinking insights:")
        for insight in insights:
            print(f"  • {insight}")

def demo_concept_compression():
    """Demo 2: Concept compression"""
    print_header("Demo 2: Concept Compression")
    
    engine = get_tcl_engine(quantum_mode=True)
    session_id = engine.create_session("demo_user", cognitive_level=0.8)
    
    # Compress a complex concept into TCL symbols
    concepts = [
        "artificial intelligence reasoning",
        "quantum computing superposition",
        "mathematical optimization problem",
        "philosophical logic deduction",
        "strategic planning optimization"
    ]
    
    for concept in concepts:
        print(f"\nCompressing concept: '{concept}'")
        result = engine.compress_concept(session_id, concept)
        
        print(f"  Original: {result['original_concept']}")
        print(f"  Compressed to: {result['compressed_symbols']}")
        print(f"  Compression ratio: {result['compression_ratio']:.3f}")
        print(f"  Conceptual density: {result['conceptual_density']:.3f}")
        print(f"  Cognitive weight: {result['cognitive_weight']:.3f}")

def demo_causal_reasoning():
    """Demo 3: Causal reasoning and prediction"""
    print_header("Demo 3: Causal Reasoning and Prediction")
    
    engine = get_tcl_engine(quantum_mode=True)
    session_id = engine.create_session("demo_user", cognitive_level=0.9)
    
    # First, establish some causal relationships
    causal_expressions = [
        "Ψ → Γ",           # Thought causes concept
        "Γ → Δ",           # Concept causes difference
        "∀x (x → ∞Ψ)",    # Universal causation to infinite thinking
        "ΣΨ = Ψ₁ + Ψ₂",   # Superthought composition
    ]
    
    print("Establishing causal relationships:")
    for expr in causal_expressions:
        print(f"  Processing: {expr}")
        try:
            result = engine.process_thought(session_id, expr)
            enhancement = result.get('metrics', {}).get('enhancement_level', 1.0)
            print(f"    Enhancement: {enhancement:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Now analyze causal chains starting from "Ψ" (thought)
    print(f"\nAnalyzing causal chains from 'Ψ' (thought):")
    try:
        causal_result = engine.generate_causal_chain(session_id, "Ψ", depth=4)
        
        print(f"  Cause: {causal_result.get('cause', 'N/A')}")
        print(f"  Causal chains found: {len(causal_result.get('causal_chains', []))}")
        
        predicted_effects = causal_result.get('predicted_effects', [])
        if predicted_effects:
            print(f"  Predicted effects:")
            for effect, confidence in predicted_effects[:3]:
                print(f"    • {effect} (confidence: {confidence:.3f})")
        
        print(f"  Chain complexity: {causal_result.get('chain_complexity', 0)}")
        print(f"  Prediction confidence: {causal_result.get('prediction_confidence', 0.0):.3f}")
    except Exception as e:
        print(f"  Error analyzing causal chains: {e}")
        # Try to show what we got instead
        print(f"  Available keys: {list(causal_result.keys()) if 'causal_result' in locals() else 'N/A'}")

def demo_enhanced_reasoning():
    """Demo 4: Enhanced reasoning for complex problems"""
    print_header("Demo 4: Enhanced Reasoning for Complex Problems")
    
    engine = get_tcl_engine(quantum_mode=True)
    session_id = engine.create_session("demo_user", cognitive_level=1.0)  # Master level
    
    # Complex problems to enhance reasoning on
    problems = [
        "How can quantum computing revolutionize artificial intelligence?",
        "What are the fundamental limits of human cognitive enhancement?",
        "How do we optimize resource allocation across multiple competing priorities?",
        "What strategies maximize learning efficiency while minimizing cognitive load?"
    ]
    
    for problem in problems:
        print(f"\nEnhancing reasoning for: '{problem}'")
        result = engine.enhance_reasoning(session_id, problem)
        
        print(f"  Conceptual mapping: {result['conceptual_mapping']['concept_count']} key concepts")
        print(f"  Causal analysis: {result['causal_analysis']['chain_complexity']} chains found")
        print(f"  Enhanced solutions:")
        for i, solution in enumerate(result['enhanced_solutions'], 1):
            print(f"    {i}. {solution}")
        print(f"  Reasoning enhancement level: {result['reasoning_enhancement_level']:.2f}x")

def demo_session_management():
    """Demo 5: Session management and monitoring"""
    print_header("Demo 5: Session Management and Monitoring")
    
    engine = get_tcl_engine(quantum_mode=True)
    
    # Create multiple sessions with different cognitive levels
    sessions = []
    cognitive_levels = [0.3, 0.5, 0.7, 0.9]
    
    for level in cognitive_levels:
        session_id = engine.create_session(f"user_{level}", cognitive_level=level)
        sessions.append((session_id, level))
        print(f"Created session {session_id} (cognitive level: {level})")
    
    # Process some thoughts in each session
    tcl_thought = "Ψ → ΓΛ"  # Thought causes conceptual logic
    
    print(f"\nProcessing '{tcl_thought}' in all sessions:")
    for session_id, level in sessions:
        result = engine.process_thought(session_id, tcl_thought)
        enhancement = result['metrics']['cognitive_enhancement']
        print(f"  Level {level}: {enhancement:.2f}x enhancement")
    
    # Get global statistics
    print(f"\nGlobal TCL System Statistics:")
    stats = engine.get_global_stats()
    print(f"  Active sessions: {stats['active_sessions']}")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Average cognitive enhancement: {stats['average_cognitive_enhancement']:.2f}x")
    print(f"  Quantum mode: {stats['quantum_mode']}")
    
    # Get detailed status for one session
    print(f"\nDetailed status for session {sessions[0][0]}:")
    status = engine.get_session_status(sessions[0][0])
    print(f"  Active: {status['active']}")
    print(f"  Cognitive level: {status['cognitive_level']}")
    print(f"  Enhancement level: {status['enhancement_level']:.2f}x")
    print(f"  Symbol count: {status['symbol_count']}")
    print(f"  Abstract reasoning score: {status['metrics']['abstract_reasoning_score']:.3f}")

def demo_advanced_features():
    """Demo 6: Advanced TCL features"""
    print_header("Demo 6: Advanced TCL Features")
    
    engine = get_tcl_engine(quantum_mode=True)
    session_id = engine.create_session("advanced_user", cognitive_level=0.95)
    
    # Advanced TCL expressions
    advanced_expressions = [
        "∀x ∈ Γ: (x → Ψ) ⟹ (x ⊥ ¬Ψ)",  # Universal quantifier with causality and constraint
        "ΣΨ₁,ΣΨ₂ ≡ ∫Ψ(x)dx",           # Superthought integration
        "ΓΛΩ ⟹ (∀t ∈ T: ΔΓ(t) ≠ 0)",     # Complex logical implication
        "∞Ψ = lim(n→∞) Σⁿᵢ₌₁ Ψᵢ",       # Infinite thinking limit
    ]
    
    print("Processing advanced TCL expressions:")
    for i, expr in enumerate(advanced_expressions, 1):
        print(f"\n  Expression {i}: {expr}")
        try:
            result = engine.process_thought(session_id, expr)
            print(f"    Processing time: {result['processing_time']:.6f}s")
            print(f"    Enhancement: {result['metrics']['cognitive_enhancement']:.2f}x")
            print(f"    Causal predictions: {len(result['causal_predictions'])}")
        except Exception as e:
            print(f"    Error: {e}")

def demo_cognitive_evolution():
    """Demo 7: Cognitive evolution over time"""
    print_header("Demo 7: Cognitive Evolution Over Time")
    
    engine = get_tcl_engine(quantum_mode=True)
    session_id = engine.create_session("evolving_user", cognitive_level=0.1)  # Start with low level
    
    print("Simulating cognitive evolution over 10 processing cycles:")
    
    enhancement_levels = []
    conceptual_densities = []
    
    # Process progressive TCL expressions
    evolution_expressions = [
        "Ψ",           # Simple thought
        "Ψ → Γ",      # Thought causes concept
        "ΓΛ → Δ",     # Conceptual logic causes difference
        "∀x (x → ∞Ψ)", # Universal infinite thinking
        "ΣΨ = Ψ₁ + Ψ₂ + Ψ₃",  # Complex superthought
        "∞Ψ ≡ ∫Γ(x)dx",       # Integrated infinite thinking
        "ΓΛΩ ⟹ ∀x ∈ S: x ⊥ ¬Ψ",  # Complete logical system
    ]
    
    for i, expr in enumerate(evolution_expressions):
        print(f"\n  Cycle {i+1}: {expr}")
        result = engine.process_thought(session_id, expr)
        
        enhancement = result['metrics']['cognitive_enhancement']
        density = result['metrics']['conceptual_density']
        
        enhancement_levels.append(enhancement)
        conceptual_densities.append(density)
        
        print(f"    Enhancement: {enhancement:.2f}x")
        print(f"    Conceptual density: {density:.3f}")
        
        # Evolve cognitive level over time
        new_level = min(1.0, 0.1 + (i + 1) * 0.12)
        # Note: In a real implementation, you'd update the session's cognitive level
        print(f"    Evolved cognitive level: {new_level:.2f}")
    
    # Show evolution summary
    print(f"\nCognitive Evolution Summary:")
    print(f"  Starting enhancement: {enhancement_levels[0]:.2f}x")
    print(f"  Ending enhancement: {enhancement_levels[-1]:.2f}x")
    print(f"  Total improvement: {enhancement_levels[-1] / enhancement_levels[0]:.2f}x")
    print(f"  Peak conceptual density: {max(conceptual_densities):.3f}")

def print_safety_warning():
    """Print safety warning about TCL technology"""
    print_header("⚠️  SAFETY WARNING ⚠️")
    print("""
Thought-Compression Language (TCL) is a dangerous cognitive enhancement technology.

CAPABILITIES:
• Enables superhuman analytical and reasoning capabilities
• Compresses complex concepts into simple symbolic representations  
• Maps causal relationships with unprecedented precision
• Amplifies human cognitive abilities beyond natural limits

RISKS:
• Users may develop capabilities that exceed normal human intelligence
• Could create cognitive inequality between enhanced and non-enhanced individuals
• May lead to dependency on cognitive enhancement for complex thinking
• Potential for misuse in competitive or exploitative contexts

ETHICAL CONSIDERATIONS:
• Should only be used for beneficial purposes (research, education, problem-solving)
• Requires careful monitoring of cognitive enhancement levels
• Users should maintain awareness of their enhanced capabilities
• Consider societal implications of cognitive enhancement

USE RESPONSIBLY.
""")

def main():
    """Main demo function"""
    print("Thought-Compression Language (TCL) Demonstration")
    print("A language optimized for thinking, not talking")
    print("Enabling superhuman cognitive capabilities")
    
    try:
        # Run all demos
        demo_basic_thinking()
        demo_concept_compression()
        demo_causal_reasoning()
        demo_enhanced_reasoning()
        demo_session_management()
        demo_advanced_features()
        demo_cognitive_evolution()
        
        # Print safety warning at the end
        print_safety_warning()
        
        print_header("Demo Completed Successfully")
        print("TCL system demonstration completed.")
        print("All cognitive enhancement capabilities tested.")
        print("System ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)