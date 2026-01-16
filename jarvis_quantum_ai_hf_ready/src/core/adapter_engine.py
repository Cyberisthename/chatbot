"""
Core Adapter Engine for JARVIS-2v
Implements modular AI adapters with graph relationships and Y/Z/X bit routing
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from pathlib import Path


class AdapterStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen" 
    DEPRECATED = "deprecated"


@dataclass
class Adapter:
    """Modular AI adapter with metadata, metrics, and Y/Z/X bit patterns"""
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
    status: AdapterStatus = AdapterStatus.ACTIVE
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize adapter to dictionary"""
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
            "status": self.status.value,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Adapter":
        """Deserialize adapter from dictionary"""
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
            status=AdapterStatus(data.get("status", "active")),
            version=data.get("version", 1)
        )


class AdapterGraph:
    """Simplified adapter graph without heavy dependencies"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}
        self._load_graph()
    
    def _load_graph(self):
        """Load adapter graph from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return

        # Native lightweight format.
        if isinstance(data.get("nodes"), dict) and isinstance(data.get("edges"), dict):
            self.nodes = data.get("nodes", {})
            self.edges = data.get("edges", {})
            return

        # Backward compatibility: networkx node_link_data format.
        nodes_list = data.get("nodes")
        links_list = data.get("links") or data.get("links")
        if isinstance(nodes_list, list) and isinstance(links_list, list):
            self.nodes = {}
            for node in nodes_list:
                node_id = node.get("id") or node.get("key")
                if node_id is None:
                    continue
                self.nodes[str(node_id)] = dict(node)

            self.edges = {}
            for link in links_list:
                src = link.get("source")
                tgt = link.get("target")
                if src is None or tgt is None:
                    continue
                w = float(link.get("weight", 1.0))
                self.edges.setdefault(str(src), []).append([str(tgt), w])
            return
    
    def _save_graph(self):
        """Save adapter graph to disk"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_adapter(self, adapter: Adapter):
        """Add adapter as node to graph"""
        self.nodes[adapter.id] = adapter.to_dict()
        self._save_graph()
    
    def add_dependency(self, parent_id: str, child_id: str, weight: float = 1.0):
        """Add dependency edge between adapters"""
        if parent_id not in self.edges:
            self.edges[parent_id] = []
        self.edges[parent_id].append((child_id, weight))
        self._save_graph()
    
    def get_adapter(self, adapter_id: str) -> Optional[Adapter]:
        """Retrieve adapter from graph"""
        if adapter_id in self.nodes:
            return Adapter.from_dict(self.nodes[adapter_id])
        return None
    
    def find_best_path(self, target_adapter_id: str) -> List[str]:
        """Find optimal adapter path"""
        if target_adapter_id not in self.nodes:
            return []
        
        adapter_data = self.nodes[target_adapter_id]
        if adapter_data.get("status") == AdapterStatus.ACTIVE.value:
            return [target_adapter_id]
        return []
    
    def get_related_adapters(self, adapter_id: str, depth: int = 1) -> List[str]:
        """Get related adapters within N hops"""
        if adapter_id not in self.edges:
            return []
        
        related = set()
        for child_id, _ in self.edges.get(adapter_id, []):
            related.add(child_id)
            if depth > 1:
                for sub_child in self.get_related_adapters(child_id, depth - 1):
                    related.add(sub_child)
        
        return list(related)


class YZXBitRouter:
    """Y/Z/X bit-based routing system for adapter selection"""
    
    def __init__(self, y_bits: int = 16, z_bits: int = 8, x_bits: int = 8):
        self.y_size = y_bits
        self.z_size = z_bits
        self.x_size = x_bits
        self.persistence_file = Path("./bit_patterns.json")
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load bit patterns from disk"""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_patterns(self):
        """Save bit patterns to disk"""
        with open(self.persistence_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def infer_bits_from_input(self, prompt: str, context: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
        """
        Infer Y/Z/X bit patterns from input prompt and context
        Uses semantic analysis to assign bit values
        """
        # Simple rule-based inference
        prompt_lower = prompt.lower()
        
        # Y-bits: task/domain classification
        y_bits = [0] * self.y_size
        if any(word in prompt_lower for word in ["code", "program", "function", "debug"]):
            y_bits[0] = 1  # programming domain
        if any(word in prompt_lower for word in ["math", "calculate", "equation", "number"]):
            y_bits[1] = 1  # mathematics domain
        if any(word in prompt_lower for word in ["quantum", "physics", "experiment", "simulation"]):
            y_bits[2] = 1  # scientific domain
        if any(word in prompt_lower for word in ["explain", "describe", "what", "how"]):
            y_bits[3] = 1  # explanation domain
        
        # Z-bits: difficulty/precision
        z_bits = [0] * self.z_size
        word_count = len(prompt.split())
        if word_count > 100:
            z_bits[0] = 1  # long input
        if any(word in prompt_lower for word in ["complex", "advanced", "expert"]):
            z_bits[1] = 1  # high complexity
        
        # X-bits: experimental toggles
        x_bits = [0] * self.x_size
        if "quantum_sim" in context.get("features", []):
            x_bits[0] = 1  # use quantum simulation
        if "recall_only" in context.get("features", []):
            x_bits[1] = 1  # use memory only
        
        return y_bits, z_bits, x_bits
    
    def select_adapters(self, y_bits: List[int], z_bits: List[int], x_bits: List[int], 
                       available_adapters: List[Adapter]) -> List[Adapter]:
        """
        Select best adapters based on bit patterns
        Uses weighted matching algorithm
        """
        scored_adapters = []
        
        for adapter in available_adapters:
            if adapter.status != AdapterStatus.ACTIVE:
                continue
                
            # Calculate bit pattern similarity
            y_match = self._bit_similarity(y_bits, adapter.y_bits)
            z_match = self._bit_similarity(z_bits, adapter.z_bits)
            x_match = self._bit_similarity(x_bits, adapter.x_bits)
            
            # Weighted score
            score = (y_match * 0.5 + z_match * 0.3 + x_match * 0.2)
            
            # Boost by success rate
            if adapter.total_calls > 0:
                success_rate = adapter.success_count / adapter.total_calls
                score *= (0.8 + success_rate * 0.2)
            
            scored_adapters.append((adapter, score))
        
        # Sort by score and return top adapters
        scored_adapters.sort(key=lambda x: x[1], reverse=True)
        return [adapter for adapter, _ in scored_adapters[:3]]  # Top 3 adapters
    
    def _bit_similarity(self, bits1: List[int], bits2: List[int]) -> float:
        """Calculate similarity between two bit vectors"""
        if len(bits1) != len(bits2):
            return 0.0
        
        matching = sum(1 for b1, b2 in zip(bits1, bits2) if b1 == b2 and b1 == 1)
        total_ones = sum(bits1) + sum(bits2)
        
        if total_ones == 0:
            return 1.0
        
        return (2 * matching) / total_ones


class AdapterEngine:
    """Main adapter engine for JARVIS-2v"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters_path = Path(config.get("adapters", {}).get("storage_path", "./adapters"))
        self.graph_path = config.get("adapters", {}).get("graph_path", "./adapters_graph.json")
        self.auto_create = config.get("adapters", {}).get("auto_create", True)
        self.freeze_after_creation = config.get("adapters", {}).get("freeze_after_creation", True)
        
        # Initialize components
        self.adapter_graph = AdapterGraph(self.graph_path)
        self.bit_router = YZXBitRouter(
            config.get("bits", {}).get("y_bits", 16),
            config.get("bits", {}).get("z_bits", 8),
            config.get("bits", {}).get("x_bits", 8)
        )
        
        self.adapters_path.mkdir(parents=True, exist_ok=True)
    
    def create_adapter(self, task_tags: List[str], y_bits: List[int], z_bits: List[int], 
                      x_bits: List[int], parameters: Dict[str, Any] = None, 
                      parent_ids: List[str] = None) -> Adapter:
        """Create new adapter with non-destructive learning"""
        adapter_id = f"adapter_{uuid.uuid4().hex[:8]}"
        
        adapter = Adapter(
            id=adapter_id,
            task_tags=task_tags,
            y_bits=y_bits,
            z_bits=z_bits,
            x_bits=x_bits,
            parameters=parameters or {},
            parent_ids=parent_ids or []
        )
        
        # Add to graph
        self.adapter_graph.add_adapter(adapter)
        
        # Add parent relationships
        for parent_id in parent_ids or []:
            self.adapter_graph.add_dependency(parent_id, adapter_id)
        
        # Freeze if enabled
        if self.freeze_after_creation:
            adapter.status = AdapterStatus.FROZEN
        
        # Persist to disk
        self._save_adapter(adapter)
        
        return adapter
    
    def get_adapter(self, adapter_id: str) -> Optional[Adapter]:
        """Retrieve adapter by ID"""
        return self.adapter_graph.get_adapter(adapter_id) or self._load_adapter(adapter_id)
    
    def list_adapters(self, status: AdapterStatus = None) -> List[Adapter]:
        """List all adapters, optionally filtered by status"""
        adapters = []
        for adapter_file in self.adapters_path.glob("*.json"):
            adapter = self._load_adapter(adapter_file.stem)
            if adapter and (status is None or adapter.status == status):
                adapters.append(adapter)
        return adapters
    
    def route_task(self, input_text: str, context: Dict[str, Any]) -> List[Adapter]:
        """
        Route task to appropriate adapters using Y/Z/X bits
        Returns sorted list of adapters by relevance
        """
        # Infer bit patterns
        y_bits, z_bits, x_bits = self.bit_router.infer_bits_from_input(input_text, context)
        
        # Get available adapters
        available_adapters = self.list_adapters(status=AdapterStatus.ACTIVE)
        
        # Select best adapters
        selected_adapters = self.bit_router.select_adapters(y_bits, z_bits, x_bits, available_adapters)
        
        # Log routing decision
        print(f"🔀 Routing: Y={y_bits[:4]}... Z={z_bits[:4]}... X={x_bits[:4]}... -> {[a.id for a in selected_adapters[:2]]}")
        
        return selected_adapters
    
    def freeze_adapter(self, adapter_id: str) -> bool:
        """Freeze adapter to prevent further modification"""
        adapter = self.get_adapter(adapter_id)
        if adapter:
            adapter.status = AdapterStatus.FROZEN
            self._save_adapter(adapter)
            self.adapter_graph.add_adapter(adapter)
            return True
        return False
    
    def _save_adapter(self, adapter: Adapter):
        """Save adapter to disk"""
        adapter_path = self.adapters_path / f"{adapter.id}.json"
        with open(adapter_path, 'w') as f:
            json.dump(adapter.to_dict(), f, indent=2)
    
    def _load_adapter(self, adapter_id: str) -> Optional[Adapter]:
        """Load adapter from disk"""
        adapter_path = self.adapters_path / f"{adapter_id}.json"
        if adapter_path.exists():
            try:
                with open(adapter_path, 'r') as f:
                    data = json.load(f)
                    return Adapter.from_dict(data)
            except json.JSONDecodeError:
                return None
        return None


class QuantumArtifact:
    """Synthetic quantum experiment artifact with adapter linkage"""

    def __init__(
        self,
        artifact_id: str,
        experiment_type: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        linked_adapter_ids: List[str],
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.artifact_id = artifact_id
        self.experiment_type = experiment_type
        self.config = config
        self.results = results
        self.linked_adapter_ids = linked_adapter_ids
        self.created_at = created_at if created_at is not None else time.time()
        self.metadata = metadata or {
            "synthetic_simulation": True,
            "lab_data_source": "simulated",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "experiment_type": self.experiment_type,
            "config": self.config,
            "results": self.results,
            "linked_adapter_ids": self.linked_adapter_ids,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


__all__ = ["Adapter", "AdapterGraph", "YZXBitRouter", "AdapterEngine", "QuantumArtifact", "AdapterStatus"]