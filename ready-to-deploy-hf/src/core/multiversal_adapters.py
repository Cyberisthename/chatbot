"""
Multiversal Adapter Engine for JARVIS-2v
Implements parallel universes as compute nodes with non-destructive cross-universe learning
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from pathlib import Path
import time
import random

from .adapter_engine import Adapter, AdapterGraph, YZXBitRouter, AdapterStatus, QuantumArtifact


class UniverseState(Enum):
    """States of a universe in the multiverse"""
    ACTIVE = "active"
    DORMANT = "dormant"
    COLLAPSED = "collapsed"
    MERGED = "merged"
    ISOLATED = "isolated"


@dataclass 
class MultiversalAdapter:
    """Extended adapter with multiversal addressing and interference patterns"""
    id: str
    task_tags: List[str]
    y_bits: List[int]  # task/domain bits
    z_bits: List[int]  # difficulty/precision bits
    x_bits: List[int]  # experimental toggles
    parameters: Dict[str, Any] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    success_count: int = 0
    total_calls: int = 0
    domains: Set[str] = field(default_factory=set)
    status: str = "active"
    version: int = 1
    universe_id: str = field(default_factory=lambda: f"universe_{uuid.uuid4().hex[:8]}")
    universe_bits: List[int] = field(default_factory=lambda: [0] * 16)  # Multiversal dimension
    branch_path: List[str] = field(default_factory=list)  # Decision point history
    interference_weight: float = 0.0  # How much this universe influences others
    coherence_level: float = 1.0  # Quantum coherence (0-1)
    artifact_count: int = 0  # Number of artifacts generated
    cross_universe_success_rate: float = 0.0  # Success when borrowed by other universes
    parent_universe_ids: List[str] = field(default_factory=list)  # Branching history
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize multiversal adapter to dictionary"""
        return {
            "id": self.id,
            "task_tags": self.task_tags,
            "y_bits": self.y_bits,
            "z_bits": self.z_bits,
            "x_bits": self.x_bits,
            "parameters": self.parameters,
            "rules": self.rules,
            "prompts": self.prompts,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "domains": list(self.domains),
            "status": self.status,
            "version": self.version,
            "universe_id": self.universe_id,
            "universe_bits": self.universe_bits,
            "branch_path": self.branch_path,
            "interference_weight": self.interference_weight,
            "coherence_level": self.coherence_level,
            "artifact_count": self.artifact_count,
            "cross_universe_success_rate": self.cross_universe_success_rate,
            "parent_universe_ids": self.parent_universe_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiversalAdapter":
        """Deserialize multiversal adapter from dictionary"""
        return cls(
            id=data["id"],
            task_tags=data.get("task_tags", []),
            y_bits=data.get("y_bits", [0] * 16),
            z_bits=data.get("z_bits", [0] * 8),
            x_bits=data.get("x_bits", [0] * 8),
            parameters=data.get("parameters", {}),
            rules=data.get("rules", []),
            prompts=data.get("prompts", []),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", time.time()),
            success_count=data.get("success_count", 0),
            total_calls=data.get("total_calls", 0),
            domains=set(data.get("domains", [])),
            status=data.get("status", "active"),
            version=data.get("version", 1),
            universe_id=data.get("universe_id", f"universe_{uuid.uuid4().hex[:8]}"),
            universe_bits=data.get("universe_bits", [0] * 16),
            branch_path=data.get("branch_path", []),
            interference_weight=data.get("interference_weight", 0.0),
            coherence_level=data.get("coherence_level", 1.0),
            artifact_count=data.get("artifact_count", 0),
            cross_universe_success_rate=data.get("cross_universe_success_rate", 0.0),
            parent_universe_ids=data.get("parent_universe_ids", [])
        )


class MultiversalRoutingEngine:
    """Engine for routing queries across the multiverse using interference patterns"""
    
    def __init__(self, y_bits: int = 16, z_bits: int = 8, x_bits: int = 8, u_bits: int = 16):
        self.y_size = y_bits
        self.z_size = z_bits  
        self.x_size = x_bits
        self.u_size = u_bits  # Universe bits
        self.persistence_file = Path("./multiversal_patterns.json")
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, Any]:
        """Load multiversal routing patterns from disk"""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_patterns(self):
        """Save multiversal routing patterns to disk"""
        with open(self.persistence_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def generate_universe_signature(self, seed_data: Dict[str, Any]) -> List[int]:
        """Generate universe signature from problem context"""
        universe_bits = [0] * self.u_size
        
        # Map problem characteristics to universe bits
        problem_type = seed_data.get("type", "unknown")
        complexity = seed_data.get("complexity", 1)
        domain = seed_data.get("domain", "general")
        
        # Domain-based universe selection
        domain_map = {
            "medical": 0, "cancer": 1, "biology": 2, "chemistry": 3,
            "physics": 4, "quantum": 5, "computing": 6, "ai": 7,
            "engineering": 8, "mathematics": 9, "psychology": 10,
            "sociology": 11, "economics": 12, "art": 13, "literature": 14
        }
        
        if domain in domain_map:
            universe_bits[domain_map[domain]] = 1
            
        # Complexity affects multiple bits
        complexity_bits = min(15, int(complexity * 8))
        for i in range(complexity_bits):
            if i < self.u_size - 15:
                universe_bits[15 + i] = 1
                
        return universe_bits
    
    def calculate_interference_weight(self, source_universe: str, target_universe: str, 
                                    source_adapter: MultiversalAdapter, 
                                    target_problem: Dict[str, Any]) -> float:
        """Calculate interference weight between universes for cross-universe knowledge transfer"""
        
        # Base interference from source adapter
        base_weight = source_adapter.interference_weight
        
        # Domain similarity boost
        problem_domain = target_problem.get("domain", "general")
        source_domains = list(source_adapter.domains)
        domain_match = 1.0 if problem_domain in source_domains else 0.3
        
        # Coherence factor (more coherent universes have stronger interference)
        coherence_factor = source_adapter.coherence_level
        
        # Success rate boost
        success_boost = 1.0 + source_adapter.cross_universe_success_rate
        
        # Universe distance (closer universes interfere more)
        universe_distance = self._calculate_universe_distance(source_universe, target_universe)
        distance_factor = math.exp(-universe_distance * 0.1)
        
        final_weight = base_weight * domain_match * coherence_factor * success_boost * distance_factor
        return min(1.0, final_weight)  # Cap at 1.0
    
    def _calculate_universe_distance(self, universe1: str, universe2: str) -> float:
        """Calculate quantum distance between two universes"""
        if universe1 == universe2:
            return 0.0
            
        # Use hash-based distance for deterministic universe relationships
        hash1 = int(universe1[-8:], 16) if len(universe1) >= 8 else 0
        hash2 = int(universe2[-8:], 16) if len(universe2) >= 8 else 0
        
        xor_diff = hash1 ^ hash2
        max_bits = 32
        distance = bin(xor_diff).count('1') / max_bits
        return distance
    
    def route_to_parallel_universes(self, query: Dict[str, Any], 
                                   available_adapters: List[MultiversalAdapter],
                                   target_universe: str = None) -> List[Tuple[MultiversalAdapter, float]]:
        """Route query to best universes using interference patterns"""
        
        if target_universe is None:
            target_universe = f"universe_{uuid.uuid4().hex[:8]}"
            
        scored_universes = []
        
        for adapter in available_adapters:
            if adapter.status != AdapterStatus.ACTIVE:
                continue
                
            # Skip if same universe unless specifically looking for cross-universe transfer
            if adapter.universe_id == target_universe:
                continue
                
            interference_weight = self.calculate_interference_weight(
                adapter.universe_id, target_universe, adapter, query
            )
            
            # Boost if this universe has solved similar problems
            problem_domain = query.get("domain", "general")
            if problem_domain in adapter.domains:
                interference_weight *= 1.5
                
            # Boost high-coherence universes
            if adapter.coherence_level > 0.8:
                interference_weight *= 1.3
                
            if interference_weight > 0.1:  # Threshold for relevance
                scored_universes.append((adapter, interference_weight))
        
        # Sort by interference weight and return top universes
        scored_universes.sort(key=lambda x: x[1], reverse=True)
        return scored_universes[:5]  # Top 5 universes
    
    def amplify_successful_universe(self, successful_adapter: MultiversalAdapter, 
                                  source_problem: Dict[str, Any]) -> None:
        """Amplify a successful universe's interference pattern"""
        
        # Increase interference weight based on success
        success_boost = 0.1 + (successful_adapter.success_count / max(1, successful_adapter.total_calls)) * 0.2
        successful_adapter.interference_weight = min(1.0, successful_adapter.interference_weight + success_boost)
        
        # Increase coherence level
        successful_adapter.coherence_level = min(1.0, successful_adapter.coherence_level + 0.05)
        
        # Update cross-universe success rate
        if successful_adapter.total_calls > 0:
            current_rate = successful_adapter.cross_universe_success_rate
            new_rate = (current_rate * 0.9) + (0.1 * (successful_adapter.success_count / successful_adapter.total_calls))
            successful_adapter.cross_universe_success_rate = new_rate
        
        # Save updated patterns
        self.patterns[f"universe_{successful_adapter.universe_id}"] = {
            "interference_weight": successful_adapter.interference_weight,
            "coherence_level": successful_adapter.coherence_level,
            "cross_universe_success_rate": successful_adapter.cross_universe_success_rate,
            "last_updated": time.time()
        }
        self._save_patterns()


@dataclass
class Universe:
    """Represents a parallel universe in the multiverse"""
    universe_id: str
    parent_universe_id: Optional[str] = None
    decision_point: Optional[str] = None
    branch_timestamp: float = field(default_factory=time.time)
    state: str = "active"  # Use string instead of enum
    coherence_level: float = 1.0
    artifact_count: int = 0
    total_solutions: int = 0
    successful_solutions: int = 0
    interference_reach: float = 0.5  # How far this universe's influence extends
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_id": self.universe_id,
            "parent_universe_id": self.parent_universe_id,
            "decision_point": self.decision_point,
            "branch_timestamp": self.branch_timestamp,
            "state": self.state,
            "coherence_level": self.coherence_level,
            "artifact_count": self.artifact_count,
            "total_solutions": self.total_solutions,
            "successful_solutions": self.successful_solutions,
            "interference_reach": self.interference_reach
        }


class MultiversalComputeEngine:
    """Main engine for multiversal computing with parallel universe simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get("multiverse", {}).get("storage_path", "./multiverse"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.universes: Dict[str, Universe] = {}
        self.multiversal_routing = MultiversalRoutingEngine(
            config.get("bits", {}).get("y_bits", 16),
            config.get("bits", {}).get("z_bits", 8),
            config.get("bits", {}).get("x_bits", 8),
            config.get("bits", {}).get("u_bits", 16)  # Universe bits
        )
        
        # Load existing universes
        self._load_universes()
        
    def create_parallel_universe(self, parent_universe_id: str, decision_point: str,
                               problem_context: Dict[str, Any]) -> str:
        """Create a new parallel universe from a decision point"""
        
        new_universe_id = f"universe_{uuid.uuid4().hex[:8]}"
        
        # Create new universe
        new_universe = Universe(
            universe_id=new_universe_id,
            parent_universe_id=parent_universe_id,
            decision_point=decision_point,
            coherence_level=0.9  # Slightly less coherent than parent
        )
        
        self.universes[new_universe_id] = new_universe
        
        # Update parent universe
        if parent_universe_id in self.universes:
            parent = self.universes[parent_universe_id]
            parent.artifact_count += 1
            # Parent's interference reach might expand
            parent.interference_reach = min(1.0, parent.interference_reach + 0.1)
        
        # Save universe
        self._save_universe(new_universe)
        
        print(f"🌌 Created parallel universe {new_universe_id} from {parent_universe_id}")
        print(f"   Decision point: {decision_point}")
        
        return new_universe_id
    
    def simulate_universe_evolution(self, universe_id: str, steps: int = 10) -> Dict[str, Any]:
        """Simulate evolution of a universe over time steps"""
        
        if universe_id not in self.universes:
            return {"error": f"Universe {universe_id} not found"}
        
        universe = self.universes[universe_id]
        evolution_log = []
        
        for step in range(steps):
            # Simulate quantum fluctuations
            coherence_change = random.gauss(0, 0.05)
            universe.coherence_level = max(0.1, min(1.0, universe.coherence_level + coherence_change))
            
            # Random branching events
            if random.random() < 0.1:  # 10% chance per step
                new_universe_id = self.create_parallel_universe(
                    universe_id, 
                    f"branch_{step}_{random.randint(1000, 9999)}",
                    {"evolution_step": step}
                )
                evolution_log.append({
                    "step": step,
                    "event": "branching",
                    "new_universe": new_universe_id,
                    "coherence": universe.coherence_level
                })
            
            # Interference events
            if random.random() < 0.2:  # 20% chance per step
                universe.interference_reach = min(1.0, universe.interference_reach + 0.05)
                evolution_log.append({
                    "step": step,
                    "event": "interference_amplification",
                    "coherence": universe.coherence_level,
                    "interference_reach": universe.interference_reach
                })
            
            evolution_log.append({
                "step": step,
                "coherence": universe.coherence_level,
                "artifact_count": universe.artifact_count,
                "total_solutions": universe.total_solutions
            })
        
        # Save updated universe
        self._save_universe(universe)
        
        return {
            "universe_id": universe_id,
            "evolution_steps": steps,
            "final_coherence": universe.coherence_level,
            "final_interference_reach": universe.interference_reach,
            "events": evolution_log
        }
    
    def find_successful_universes(self, problem_domain: str, 
                                similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find universes that have been successful with similar problems"""
        
        successful_universes = []
        
        for universe_id, universe in self.universes.items():
            if universe.state != UniverseState.ACTIVE:
                continue
                
            success_rate = universe.successful_solutions / max(1, universe.total_solutions)
            
            # Consider coherence and success rate
            if universe.coherence_level > 0.6 and success_rate > similarity_threshold:
                # Calculate reach factor
                reach_factor = universe.interference_reach * universe.coherence_level
                successful_universes.append((universe_id, reach_factor))
        
        # Sort by reach factor
        successful_universes.sort(key=lambda x: x[1], reverse=True)
        return successful_universes[:10]  # Top 10
    
    def find_successful_universes(self, problem_domain: str, 
                                similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find universes that have been successful with similar problems"""
        
        successful_universes = []
        
        for universe_id, universe in self.universes.items():
            if universe.state != "active":
                continue
                
            success_rate = universe.successful_solutions / max(1, universe.total_solutions)
            
            # Consider coherence and success rate
            if universe.coherence_level > 0.6 and success_rate > similarity_threshold:
                # Calculate reach factor
                reach_factor = universe.interference_reach * universe.coherence_level
                successful_universes.append((universe_id, reach_factor))
        
        # Sort by reach factor
        successful_universes.sort(key=lambda x: x[1], reverse=True)
        return successful_universes[:10]  # Top 10
    
    def borrow_knowledge_from_parallel_universe(self, source_universe_id: str, 
                                              target_problem: Dict[str, Any]) -> Dict[str, Any]:
        """Borrow knowledge from a successful parallel universe"""
        
        if source_universe_id not in self.universes:
            return {"error": f"Source universe {source_universe_id} not found"}
        
        source_universe = self.universes[source_universe_id]
        
        # Create "echo" artifact from source universe
        echo_artifact = {
            "type": "multiversal_echo",
            "source_universe": source_universe_id,
            "target_problem": target_problem,
            "echo_strength": source_universe.coherence_level * source_universe.interference_reach,
            "borrowed_at": time.time(),
            "adaptation_notes": f"Borrowed from {source_universe_id} with coherence {source_universe.coherence_level:.2f}"
        }
        
        # Update source universe statistics
        source_universe.successful_solutions += 1
        source_universe.interference_reach = min(1.0, source_universe.interference_reach + 0.02)
        self._save_universe(source_universe)
        
        return {
            "success": True,
            "echo_artifact": echo_artifact,
            "source_universe_stats": {
                "coherence_level": source_universe.coherence_level,
                "interference_reach": source_universe.interference_reach,
                "success_rate": source_universe.successful_solutions / max(1, source_universe.total_solutions)
            }
        }
    
    def get_multiverse_overview(self) -> Dict[str, Any]:
        """Get overview of the entire multiverse"""
        
        total_universes = len(self.universes)
        active_universes = sum(1 for u in self.universes.values() if u.state == UniverseState.ACTIVE)
        total_artifacts = sum(u.artifact_count for u in self.universes.values())
        avg_coherence = sum(u.coherence_level for u in self.universes.values()) / max(1, total_universes)
        
        # Find most successful universe
        best_universe = None
        best_score = 0
        for universe in self.universes.values():
            score = universe.coherence_level * (universe.successful_solutions / max(1, universe.total_solutions))
            if score > best_score:
                best_score = score
                best_universe = universe.universe_id
        
        return {
            "total_universes": total_universes,
            "active_universes": active_universes,
            "total_artifacts": total_artifacts,
            "average_coherence": avg_coherence,
            "most_successful_universe": best_universe,
            "multiverse_health": avg_coherence * (active_universes / max(1, total_universes))
        }
    
    def _load_universes(self):
        """Load existing universes from disk"""
        universe_file = self.storage_path / "universes.json"
        if universe_file.exists():
            try:
                with open(universe_file, 'r') as f:
                    data = json.load(f)
                    
                for universe_data in data.get("universes", []):
                    # Ensure state is a string, not an enum
                    if "state" in universe_data and hasattr(universe_data["state"], "value"):
                        universe_data["state"] = universe_data["state"].value
                    elif "state" not in universe_data:
                        universe_data["state"] = "active"
                        
                    universe = Universe(**universe_data)
                    self.universes[universe.universe_id] = universe
                    
            except (json.JSONDecodeError, TypeError):
                pass
    
    def _save_universe(self, universe: Universe):
        """Save universe to disk"""
        # Update storage
        all_universes = [u.to_dict() for u in self.universes.values()]
        
        universe_file = self.storage_path / "universes.json"
        with open(universe_file, 'w') as f:
            json.dump({"universes": all_universes}, f, indent=2)
    
    def _save_universes(self):
        """Save all universes to disk"""
        universe_file = self.storage_path / "universes.json"
        all_universes = [u.to_dict() for u in self.universes.values()]
        
        with open(universe_file, 'w') as f:
            json.dump({"universes": all_universes}, f, indent=2)


__all__ = [
    "MultiversalAdapter", "MultiversalRoutingEngine", "MultiversalComputeEngine", 
    "Universe", "UniverseState"
]