"""
TCL Symbol System - Core symbol representation and concept graphs

One symbol = full abstract concept
Symbols are compressed representations of complex ideas
"""

import hashlib
import json
import re
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class SymbolType(Enum):
    """Types of TCL symbols based on their cognitive function"""
    PRIMITIVE = "primitive"      # Basic building blocks
    CONCEPT = "concept"          # Abstract concepts
    CAUSALITY = "causality"      # Causal relationships
    CONSTRAINT = "constraint"    # Logical constraints
    ACTION = "action"           # Executable operations
    META = "meta"               # Self-referential

@dataclass
class TCLSymbol:
    """A single TCL symbol representing a compressed concept"""
    id: str
    name: str
    type: SymbolType
    definition: str
    relationships: Dict[str, float]  # relationship -> strength (0-1)
    causal_links: List[str]         # IDs of symbols this causes/caused_by
    compression_ratio: float         # How compressed this concept is (1.0 = fully compressed)
    cognitive_weight: float         # Importance for thinking (0-1)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on symbol definition"""
        content = f"{self.name}:{self.definition}:{self.type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compress(self, other_symbols: List['TCLSymbol']) -> 'TCLSymbol':
        """Compress this symbol using relationships with other symbols"""
        if self.compression_ratio >= 1.0:
            return self
        
        # Increase compression based on relationships
        compression_boost = sum(self.relationships.values()) * 0.1
        new_ratio = min(1.0, self.compression_ratio + compression_boost)
        
        return TCLSymbol(
            id=self.id,
            name=self.name,
            type=self.type,
            definition=self.definition,
            relationships=self.relationships,
            causal_links=self.causal_links.copy(),
            compression_ratio=new_ratio,
            cognitive_weight=self.cognitive_weight * (1 + compression_boost)
        )

class ConceptGraph:
    """Graph structure representing interconnected concepts"""
    
    def __init__(self):
        self.symbols: Dict[str, TCLSymbol] = {}
        self.connections: Dict[str, Set[str]] = {}  # symbol_id -> connected_symbols
        self.conceptual_distances: Dict[Tuple[str, str], float] = {}
    
    def add_symbol(self, symbol: TCLSymbol):
        """Add a symbol to the graph"""
        self.symbols[symbol.id] = symbol
        self.connections[symbol.id] = set()
        
        # Update relationships in existing symbols
        for other_id, other_symbol in self.symbols.items():
            if other_id != symbol.id:
                if symbol.name in other_symbol.relationships:
                    self._connect_symbols(symbol.id, other_id)
    
    def _connect_symbols(self, id1: str, id2: str):
        """Create connection between two symbols"""
        self.connections[id1].add(id2)
        self.connections[id2].add(id1)
    
    def find_conceptual_neighbors(self, symbol_id: str, depth: int = 1) -> Set[str]:
        """Find symbols connected within specified depth"""
        if symbol_id not in self.connections:
            return set()
        
        visited = set()
        queue = [(symbol_id, 0)]
        result = set()
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
                
            visited.add(current_id)
            
            if current_depth > 0:
                result.add(current_id)
            
            for neighbor in self.connections[current_id]:
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
        
        return result
    
    def calculate_conceptual_distance(self, symbol1_id: str, symbol2_id: str) -> float:
        """Calculate conceptual distance between two symbols (0 = identical, 1 = completely different)"""
        if symbol1_id == symbol2_id:
            return 0.0
        
        # Check cache first
        key = tuple(sorted([symbol1_id, symbol2_id]))
        if key in self.conceptual_distances:
            return self.conceptual_distances[key]
        
        # Calculate based on relationship strength and connections
        if symbol1_id not in self.symbols or symbol2_id not in self.symbols:
            distance = 1.0
        else:
            sym1 = self.symbols[symbol1_id]
            sym2 = self.symbols[symbol2_id]
            
            # Base distance on relationship strength
            relationship_strength = 0.0
            if symbol2_id in sym1.relationships:
                relationship_strength = sym1.relationships[symbol2_id]
            elif symbol1_id in sym2.relationships:
                relationship_strength = sym2.relationships[symbol1_id]
            
            # Factor in connection distance
            neighbors1 = self.find_conceptual_neighbors(symbol1_id, depth=3)
            neighbors2 = self.find_conceptual_neighbors(symbol2_id, depth=3)
            
            connection_overlap = len(neighbors1.intersection(neighbors2)) / max(len(neighbors1.union(neighbors2)), 1)
            
            distance = 1.0 - (relationship_strength * 0.7 + connection_overlap * 0.3)
        
        self.conceptual_distances[key] = distance
        return distance
    
    def compress_graph(self) -> float:
        """Compress the entire graph and return average compression improvement"""
        total_improvement = 0.0
        symbols_list = list(self.symbols.values())
        
        for symbol in symbols_list:
            compressed = symbol.compress(symbols_list)
            self.symbols[symbol.id] = compressed
            total_improvement += compressed.compression_ratio - symbol.compression_ratio
        
        return total_improvement / len(symbols_list) if symbols_list else 0.0

class CausalityMap:
    """Maps causal relationships between symbols"""
    
    def __init__(self):
        self.causal_edges: Dict[str, Dict[str, float]] = {}  # cause -> {effect: strength}
        self.temporal_chains: List[List[str]] = []  # Ordered chains of causality
    
    def add_causal_link(self, cause_id: str, effect_id: str, strength: float = 1.0):
        """Add causal relationship between two symbols"""
        if cause_id not in self.causal_edges:
            self.causal_edges[cause_id] = {}
        self.causal_edges[cause_id][effect_id] = strength
        
        # Update symbol's causal links
        # Note: This would need to be handled by the engine
    
    def find_causal_chains(self, max_length: int = 10) -> List[List[str]]:
        """Find all causal chains in the system"""
        chains = []
        
        def explore_chain(current: str, visited: Set[str], current_chain: List[str]):
            if len(current_chain) >= max_length:
                return
            
            if current in visited:
                chains.append(current_chain.copy())
                return
            
            visited.add(current)
            current_chain.append(current)
            
            if current in self.causal_edges:
                for effect in self.causal_edges[current]:
                    explore_chain(effect, visited.copy(), current_chain.copy())
            else:
                chains.append(current_chain.copy())
        
        for cause in self.causal_edges:
            explore_chain(cause, set(), [])
        
        # Filter and optimize chains
        return [chain for chain in chains if len(chain) > 1]
    
    def predict_effects(self, cause_id: str) -> List[Tuple[str, float]]:
        """Predict likely effects of a cause symbol"""
        if cause_id not in self.causal_edges:
            return []
        
        effects = []
        for effect, strength in self.causal_edges[cause_id].items():
            # Calculate indirect effects
            indirect_effects = self._calculate_indirect_effects(effect, {cause_id})
            total_strength = strength + sum(indirect_effects.values())
            effects.append((effect, total_strength))
        
        return sorted(effects, key=lambda x: x[1], reverse=True)
    
    def _calculate_indirect_effects(self, symbol_id: str, visited: Set[str]) -> Dict[str, float]:
        """Calculate indirect effects through causal chains"""
        if symbol_id in visited:
            return {}
        
        visited.add(symbol_id)
        indirect = {}
        
        if symbol_id in self.causal_edges:
            for effect, strength in self.causal_edges[symbol_id].items():
                if effect not in visited:
                    indirect[effect] = strength
                    indirect.update(self._calculate_indirect_effects(effect, visited.copy()))
        
        return indirect

class PrimitiveSymbolFactory:
    """Factory for creating primitive TCL symbols"""
    
    @staticmethod
    def create_mathematical_primitives() -> List[TCLSymbol]:
        """Create fundamental mathematical symbols"""
        primitives = [
            TCLSymbol("∅", "nothing", SymbolType.PRIMITIVE, "The absence of all", {}, [], 0.5, 0.9),
            TCLSymbol("∞", "infinity", SymbolType.PRIMITIVE, "The unbounded", {}, [], 0.7, 0.8),
            TCLSymbol("∑", "sum", SymbolType.PRIMITIVE, "Aggregation of parts", {}, [], 0.6, 0.7),
            TCLSymbol("∫", "integral", SymbolType.PRIMITIVE, "Continuous accumulation", {}, [], 0.6, 0.7),
            TCLSymbol("∂", "change", SymbolType.PRIMITIVE, "Rate of change", {}, [], 0.5, 0.8),
            TCLSymbol("∀", "universal", SymbolType.PRIMITIVE, "For all cases", {}, [], 0.7, 0.6),
            TCLSymbol("∃", "existential", SymbolType.PRIMITIVE, "There exists", {}, [], 0.7, 0.6),
            TCLSymbol("¬", "negation", SymbolType.PRIMITIVE, "Logical NOT", {}, [], 0.8, 0.9),
        ]
        return primitives
    
    @staticmethod
    def create_cognitive_primitives() -> List[TCLSymbol]:
        """Create fundamental cognitive symbols"""
        primitives = [
            TCLSymbol("Ψ", "thought", SymbolType.PRIMITIVE, "Unit of thinking", {}, [], 0.9, 1.0),
            TCLSymbol("Γ", "concept", SymbolType.PRIMITIVE, "Abstract idea", {}, [], 0.8, 0.9),
            TCLSymbol("Λ", "logic", SymbolType.PRIMITIVE, "Reasoning structure", {}, [], 0.7, 0.9),
            TCLSymbol("Ω", "outcome", SymbolType.PRIMITIVE, "End state", {}, [], 0.6, 0.8),
            TCLSymbol("Φ", "meaning", SymbolType.PRIMITIVE, "Semantic content", {}, [], 0.8, 0.9),
            TCLSymbol("Δ", "difference", SymbolType.PRIMITIVE, "Change amount", {}, [], 0.7, 0.7),
        ]
        return primitives
    
    @staticmethod
    def create_temporal_primitives() -> List[TCLSymbol]:
        """Create fundamental temporal symbols"""
        primitives = [
            TCLSymbol("→", "causes", SymbolType.CAUSALITY, "Direct causation", {}, [], 0.8, 1.0),
            TCLSymbol("⟹", "implies", SymbolType.CAUSALITY, "Logical implication", {}, [], 0.7, 0.9),
            TCLSymbol("≡", "equivalent", SymbolType.CONCEPT, "Same meaning", {}, [], 0.6, 0.8),
            TCLSymbol("≠", "different", SymbolType.CONCEPT, "Not the same", {}, [], 0.7, 0.7),
            TCLSymbol("⊃", "contains", SymbolType.CONCEPT, "Subset relationship", {}, [], 0.5, 0.6),
            TCLSymbol("∪", "union", SymbolType.CONCEPT, "Combination", {}, [], 0.6, 0.6),
        ]
        return primitives

# Example high-level concept symbols
HIGH_LEVEL_SYMBOLS = [
    TCLSymbol("ΣΨ", "super_thought", SymbolType.CONCEPT, 
              "Combined thinking beyond individual thoughts", 
              {"Ψ": 0.8, "∑": 0.6}, ["→", "⟹"], 0.9, 1.0),
    
    TCLSymbol("ΓΛ", "conceptual_logic", SymbolType.CONCEPT,
              "Logic applied to abstract concepts",
              {"Γ": 0.9, "Λ": 0.8}, ["→", "≡"], 0.8, 0.9),
    
    TCLSymbol("∞Ψ", "infinite_thinking", SymbolType.CONCEPT,
              "Thinking without bounds",
              {"∞": 0.9, "Ψ": 0.9}, ["→", "≡"], 0.7, 1.0),
    
    TCLSymbol("∀Γ", "universal_concept", SymbolType.CONCEPT,
              "Concept applicable to all cases",
              {"∀": 0.8, "Γ": 0.7}, ["≡", "⊃"], 0.8, 0.8),
]