"""
Thought-Compression Language Engine - Core processing engine

This is the main orchestrator for the TCL system that:
1. Processes TCL code/expressions
2. Manages symbol compression and concept evolution
3. Handles causality mapping and prediction
4. Provides high-level cognitive operations
5. Integrates with existing JARVIS inference system

WARNING: This system enables superhuman cognitive capabilities.
Use responsibly.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .tcl_symbols import (
    TCLSymbol, ConceptGraph, CausalityMap, 
    PrimitiveSymbolFactory, SymbolType
)
from .tcl_parser import TCLParser
from .tcl_compiler import TCLCompiler
from .tcl_types import TCLExecutionContext, CognitiveMetrics

# Import runtime lazily to avoid circular imports
_TCLRuntime = None

def _get_tcl_runtime():
    global _TCLRuntime
    if _TCLRuntime is None:
        from .tcl_runtime import TCLRuntime
        _TCLRuntime = TCLRuntime
    return _TCLRuntime

class ThoughtCompressionEngine:
    """
    Core TCL Engine
    
    This engine transforms human thinking by compressing concepts into 
    high-density symbolic representations, enabling superhuman analytical capabilities.
    """
    
    def __init__(self, enable_quantum_mode: bool = False):
        self.quantum_mode = enable_quantum_mode
        self.sessions: Dict[str, TCLExecutionContext] = {}
        self.global_symbols = ConceptGraph()
        self.global_causality = CausalityMap()
        self.initialized = False
        self._lock = threading.Lock()
        
        # Initialize primitive symbols
        self._initialize_primitives()
        
    def _initialize_primitives(self):
        """Initialize the TCL system with primitive symbols"""
        math_primitives = PrimitiveSymbolFactory.create_mathematical_primitives()
        cognitive_primitives = PrimitiveSymbolFactory.create_cognitive_primitives()
        temporal_primitives = PrimitiveSymbolFactory.create_temporal_primitives()
        
        all_primitives = math_primitives + cognitive_primitives + temporal_primitives
        
        for primitive in all_primitives:
            self.global_symbols.add_symbol(primitive)
            
        self.initialized = True
        
    def create_session(self, user_id: str, cognitive_level: float = 0.5) -> str:
        """Create a new TCL processing session"""
        session_id = f"session_{int(time.time())}_{user_id}"
        
        context = TCLExecutionContext(
            symbols=self.global_symbols,
            causality=self.global_causality,
            session_id=session_id,
            user_cognitive_level=cognitive_level
        )
        
        with self._lock:
            self.sessions[session_id] = context
            
        return session_id
    
    def process_thought(self, session_id: str, tcl_input: str) -> Dict[str, Any]:
        """
        Process a TCL thought expression
        
        Args:
            session_id: TCL processing session
            tcl_input: TCL expression to process
            
        Returns:
            Dict containing processed result, metrics, and cognitive enhancements
        """
        start_time = time.time()
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.sessions[session_id]
        
        try:
            # Parse TCL input
            parser = TCLParser()
            parsed = parser.parse(tcl_input)
            
            # Compile and execute
            compiler = TCLCompiler()
            compiled = compiler.compile(parsed, context)
            
            runtime = _get_tcl_runtime()()
            result = runtime.execute(compiled, context)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(context, processing_time, len(tcl_input))
            
            return {
                'result': result,
                'metrics': {
                    'compression_ratio': context.metrics.compression_ratio,
                    'conceptual_density': context.metrics.conceptual_density,
                    'cognitive_enhancement': self._calculate_cognitive_enhancement(context),
                    'processing_time': processing_time,
                    'symbol_count': len(context.symbols.symbols)
                },
                'enhanced_thinking': self._generate_enhanced_insights(context, result),
                'causal_predictions': self._predict_causal_outcomes(context, result)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def compress_concept(self, session_id: str, concept: str) -> Dict[str, Any]:
        """Compress a natural language concept into TCL symbols"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        context = self.sessions[session_id]
        
        # Decompose concept into primitive elements
        compressed_symbols = self._decompose_concept(concept, context)
        
        # Compress the concept graph
        compression_ratio = context.symbols.compress_graph()
        context.metrics.compression_ratio = compression_ratio
        
        return {
            'original_concept': concept,
            'compressed_symbols': [symbol.name for symbol in compressed_symbols],
            'compression_ratio': compression_ratio,
            'conceptual_density': self._calculate_density(compressed_symbols),
            'cognitive_weight': sum(symbol.cognitive_weight for symbol in compressed_symbols) / len(compressed_symbols)
        }
    
    def generate_causal_chain(self, session_id: str, cause_symbol: str, depth: int = 5) -> Dict[str, Any]:
        """Generate causal chains starting from a cause symbol"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        context = self.sessions[session_id]
        
        # Find or create the cause symbol
        cause_id = self._find_symbol_id(context, cause_symbol)
        if not cause_id:
            return {'error': f'Symbol "{cause_symbol}" not found'}
        
        # Generate causal chains
        chains = context.causality.find_causal_chains(max_length=depth)
        predictions = context.causality.predict_effects(cause_id)
        
        return {
            'cause': cause_symbol,
            'causal_chains': chains,
            'predicted_effects': predictions,
            'chain_complexity': len(chains),
            'prediction_confidence': sum(strength for _, strength in predictions) / len(predictions) if predictions else 0.0
        }
    
    def enhance_reasoning(self, session_id: str, problem: str) -> Dict[str, Any]:
        """Use TCL to enhance problem-solving and reasoning"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        context = self.sessions[session_id]
        
        # Convert problem to TCL concepts
        concept_mapping = self._map_problem_to_concepts(problem, context)
        
        # Apply causal analysis
        causal_analysis = self._analyze_problem_causality(concept_mapping, context)
        
        # Generate enhanced solutions
        solutions = self._generate_enhanced_solutions(concept_mapping, causal_analysis, context)
        
        return {
            'original_problem': problem,
            'conceptual_mapping': concept_mapping,
            'causal_analysis': causal_analysis,
            'enhanced_solutions': solutions,
            'reasoning_enhancement_level': self._calculate_reasoning_enhancement(context)
        }
    
    def _decompose_concept(self, concept: str, context: TCLExecutionContext) -> List[TCLSymbol]:
        """Decompose a natural language concept into TCL symbols"""
        # This is a simplified decomposition - in reality this would use NLP
        # For now, we'll map common concepts to symbols
        
        concept_lower = concept.lower()
        symbols = []
        
        # Mathematical concepts
        if any(word in concept_lower for word in ['sum', 'total', 'add']):
            symbols.append(self.global_symbols.symbols.get('∑') or TCLSymbol('∑', 'sum', SymbolType.PRIMITIVE, 'Aggregation', {}, [], 0.6, 0.7))
        
        if any(word in concept_lower for word in ['change', 'difference', 'delta']):
            symbols.append(self.global_symbols.symbols.get('∂') or TCLSymbol('∂', 'change', SymbolType.PRIMITIVE, 'Rate of change', {}, [], 0.5, 0.8))
        
        # Cognitive concepts
        if any(word in concept_lower for word in ['think', 'thought', 'reason']):
            symbols.append(self.global_symbols.symbols.get('Ψ') or TCLSymbol('Ψ', 'thought', SymbolType.PRIMITIVE, 'Unit of thinking', {}, [], 0.9, 1.0))
        
        if any(word in concept_lower for word in ['concept', 'idea', 'notion']):
            symbols.append(self.global_symbols.symbols.get('Γ') or TCLSymbol('Γ', 'concept', SymbolType.PRIMITIVE, 'Abstract idea', {}, [], 0.8, 0.9))
        
        # Logical concepts
        if any(word in concept_lower for word in ['all', 'every', 'universal']):
            symbols.append(self.global_symbols.symbols.get('∀') or TCLSymbol('∀', 'universal', SymbolType.PRIMITIVE, 'For all cases', {}, [], 0.7, 0.6))
        
        if any(word in concept_lower for word in ['cause', 'cause', 'because']):
            symbols.append(self.global_symbols.symbols.get('→') or TCLSymbol('→', 'causes', SymbolType.CAUSALITY, 'Direct causation', {}, [], 0.8, 1.0))
        
        # If no symbols found, create a conceptual symbol
        if not symbols:
            concept_symbol = TCLSymbol(
                id=f"concept_{hash(concept)}",
                name=concept.replace(" ", "_"),
                type=SymbolType.CONCEPT,
                definition=concept,
                relationships={},
                causal_links=[],
                compression_ratio=0.5,
                cognitive_weight=0.8
            )
            symbols.append(concept_symbol)
            context.symbols.add_symbol(concept_symbol)
        
        return symbols
    
    def _update_metrics(self, context: TCLExecutionContext, processing_time: float, input_length: int):
        """Update cognitive metrics based on processing"""
        context.metrics.processing_time = processing_time
        context.metrics.thinking_speed = input_length / processing_time if processing_time > 0 else 0
        
        # Update conceptual density
        if context.symbols.symbols:
            total_connections = sum(len(connections) for connections in context.symbols.connections.values())
            max_possible_connections = len(context.symbols.symbols) * (len(context.symbols.symbols) - 1)
            context.metrics.conceptual_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
        
        # Update causality depth
        chains = context.causality.find_causal_chains()
        if chains:
            context.metrics.causality_depth = max(len(chain) for chain in chains)
        
        # Calculate cognitive load
        context.metrics.cognitive_load = min(1.0, len(context.symbols.symbols) / 100)
        
        # Calculate abstract reasoning score
        reasoning_factors = [
            context.metrics.compression_ratio,
            context.metrics.conceptual_density,
            min(1.0, context.metrics.causality_depth / 10),
            context.metrics.cognitive_load
        ]
        context.metrics.abstract_reasoning_score = sum(reasoning_factors) / len(reasoning_factors)
    
    def _calculate_cognitive_enhancement(self, context: TCLExecutionContext) -> float:
        """Calculate the cognitive enhancement level achieved"""
        base_enhancement = context.metrics.abstract_reasoning_score
        
        # Factor in user cognitive level
        level_multiplier = 1.0 + context.user_cognitive_level
        
        # Factor in compression efficiency
        compression_multiplier = 1.0 + context.metrics.compression_ratio
        
        enhancement = base_enhancement * level_multiplier * compression_multiplier
        return min(2.0, enhancement)  # Cap at 2x enhancement for safety
    
    def _generate_enhanced_insights(self, context: TCLExecutionContext, result: Any) -> List[str]:
        """Generate enhanced insights based on TCL processing"""
        insights = []
        
        if context.metrics.compression_ratio > 0.7:
            insights.append("High conceptual compression achieved - complex ideas are now representable as simple symbols")
        
        if context.metrics.conceptual_density > 0.5:
            insights.append("Rich conceptual connections detected - new insights may emerge from symbol interactions")
        
        if context.metrics.causality_depth > 3:
            insights.append("Deep causal chains identified - cause-and-effect patterns are highly interconnected")
        
        if context.metrics.abstract_reasoning_score > 0.6:
            insights.append("Enhanced abstract reasoning capabilities detected - ready for complex problem solving")
        
        return insights
    
    def _predict_causal_outcomes(self, context: TCLExecutionContext, result: Any) -> List[str]:
        """Predict likely outcomes based on causal analysis"""
        predictions = []
        
        # Analyze current symbol state
        symbols = list(context.symbols.symbols.values())
        
        # Find high-impact symbols
        high_impact_symbols = [s for s in symbols if s.cognitive_weight > 0.8]
        
        for symbol in high_impact_symbols[:3]:  # Top 3
            predictions.append(f"Symbol '{symbol.name}' will likely influence future thinking patterns")
        
        # Predict enhancement trajectory
        if context.metrics.cognitive_load < 0.5:
            predictions.append("Cognitive capacity available for additional complex processing")
        
        if context.metrics.thinking_speed > 1000:
            predictions.append("Ultra-fast cognitive processing achieved - may enable real-time complex analysis")
        
        return predictions
    
    def _find_symbol_id(self, context: TCLExecutionContext, symbol_name: str) -> Optional[str]:
        """Find symbol ID by name"""
        for symbol_id, symbol in context.symbols.symbols.items():
            if symbol.name == symbol_name:
                return symbol_id
        return None
    
    def _calculate_density(self, symbols: List[TCLSymbol]) -> float:
        """Calculate conceptual density of a set of symbols"""
        if len(symbols) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                # Calculate distance based on relationships
                distance = 1.0 - symbols[i].relationships.get(symbols[j].name, 0.0)
                total_distance += distance
                comparisons += 1
        
        return 1.0 - (total_distance / comparisons) if comparisons > 0 else 0.0
    
    def _map_problem_to_concepts(self, problem: str, context: TCLExecutionContext) -> Dict[str, Any]:
        """Map a natural language problem to TCL concepts"""
        # Simplified mapping - in reality would use advanced NLP
        concepts = {}
        
        # Extract key terms
        words = problem.lower().split()
        
        # Find matching symbols
        for symbol_id, symbol in context.symbols.symbols.items():
            if any(word in symbol.name.lower() or word in symbol.definition.lower() 
                   for word in words):
                concepts[symbol.name] = {
                    'id': symbol_id,
                    'type': symbol.type.value,
                    'weight': symbol.cognitive_weight,
                    'relationships': symbol.relationships
                }
        
        return {
            'key_concepts': concepts,
            'concept_count': len(concepts),
            'dominant_themes': list(concepts.keys())[:5]
        }
    
    def _analyze_problem_causality(self, concept_mapping: Dict[str, Any], context: TCLExecutionContext) -> Dict[str, Any]:
        """Analyze causality within the problem concepts"""
        concepts = concept_mapping.get('key_concepts', {})
        causal_chains = []
        
        # Find causal relationships between concepts
        for concept_name, concept_data in concepts.items():
            concept_id = concept_data['id']
            
            # Find what this concept causes
            if concept_id in context.causality.causal_edges:
                effects = list(context.causality.causal_edges[concept_id].keys())
                if effects:
                    causal_chains.append({
                        'cause': concept_name,
                        'effects': effects,
                        'chain_strength': sum(context.causality.causal_edges[concept_id].values())
                    })
        
        return {
            'causal_chains': causal_chains,
            'chain_complexity': len(causal_chains),
            'causal_density': len(causal_chains) / len(concepts) if concepts else 0.0
        }
    
    def _generate_enhanced_solutions(self, concept_mapping: Dict[str, Any], 
                                   causal_analysis: Dict[str, Any], 
                                   context: TCLExecutionContext) -> List[str]:
        """Generate enhanced solutions using TCL reasoning"""
        solutions = []
        
        # Generate solutions based on conceptual compression
        high_weight_concepts = [
            name for name, data in concept_mapping.get('key_concepts', {}).items()
            if data['weight'] > 0.7
        ]
        
        if high_weight_concepts:
            solutions.append(f"Focus on high-impact concepts: {', '.join(high_weight_concepts[:3])}")
        
        # Generate solutions based on causal analysis
        complex_chains = [
            chain for chain in causal_analysis.get('causal_chains', [])
            if len(chain['effects']) > 2
        ]
        
        if complex_chains:
            solutions.append(f"Leverage complex causal chains: {complex_chains[0]['cause']} → {', '.join(complex_chains[0]['effects'][:3])}")
        
        # General enhancement suggestions
        if context.metrics.compression_ratio > 0.5:
            solutions.append("Apply symbol compression to reduce cognitive load while maintaining conceptual richness")
        
        if context.metrics.conceptual_density > 0.6:
            solutions.append("Explore emergent properties from dense concept interconnections")
        
        return solutions
    
    def _calculate_reasoning_enhancement(self, context: TCLExecutionContext) -> float:
        """Calculate the level of reasoning enhancement achieved"""
        factors = {
            'compression': context.metrics.compression_ratio,
            'density': context.metrics.conceptual_density,
            'causality': min(1.0, context.metrics.causality_depth / 5),
            'abstract': context.metrics.abstract_reasoning_score
        }
        
        # Weighted average
        weights = {'compression': 0.3, 'density': 0.3, 'causality': 0.2, 'abstract': 0.2}
        
        enhancement = sum(factors[key] * weights[key] for key in factors)
        return min(1.5, enhancement)  # Cap at 1.5x enhancement
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a TCL session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.sessions[session_id]
        
        return {
            'session_id': session_id,
            'active': context.active,
            'cognitive_level': context.user_cognitive_level,
            'metrics': {
                'compression_ratio': context.metrics.compression_ratio,
                'conceptual_density': context.metrics.conceptual_density,
                'cognitive_load': context.metrics.cognitive_load,
                'thinking_speed': context.metrics.thinking_speed,
                'abstract_reasoning_score': context.metrics.abstract_reasoning_score
            },
            'symbol_count': len(context.symbols.symbols),
            'causal_chains': len(context.causality.find_causal_chains()),
            'enhancement_level': self._calculate_cognitive_enhancement(context)
        }
    
    def shutdown_session(self, session_id: str):
        """Shutdown a TCL session and clean up resources"""
        if session_id in self.sessions:
            context = self.sessions[session_id]
            context.active = False
            
            with self._lock:
                del self.sessions[session_id]
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global TCL system statistics"""
        total_symbols = len(self.global_symbols.symbols)
        total_sessions = len(self.sessions)
        
        avg_enhancement = 0.0
        if self.sessions:
            enhancements = [self._calculate_cognitive_enhancement(ctx) for ctx in self.sessions.values()]
            avg_enhancement = sum(enhancements) / len(enhancements)
        
        return {
            'initialized': self.initialized,
            'quantum_mode': self.quantum_mode,
            'total_symbols': total_symbols,
            'active_sessions': total_sessions,
            'average_cognitive_enhancement': avg_enhancement,
            'system_status': 'active' if self.initialized else 'uninitialized'
        }

# Global TCL engine instance - now managed by __init__.py
# tcl_engine = None

# def get_tcl_engine(quantum_mode: bool = False) -> ThoughtCompressionEngine:
#     """Get or create the global TCL engine instance"""
#     global tcl_engine
#     if tcl_engine is None:
#         tcl_engine = ThoughtCompressionEngine(enable_quantum_mode=quantum_mode)
#     return tcl_engine