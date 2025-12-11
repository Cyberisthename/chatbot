"""
JARVIS-2v Main API Server
FastAPI-based REST API with modular architecture
"""

import os
import sys
import yaml
import time
import uvicorn
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.adapter_engine import AdapterEngine, YZXBitRouter, AdapterStatus
from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class InferRequest(BaseModel):
    """Request model for /api/infer endpoint"""
    query: str = Field(..., description="Text query to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    features: Optional[List[str]] = Field(default_factory=list, description="Feature flags")


class InferResponse(BaseModel):
    """Response model for /api/infer endpoint"""
    response: str
    adapters_used: List[str] = Field(default_factory=list)
    bit_patterns: Dict[str, List[int]] = Field(default_factory=dict)
    processing_time: float = 0.0


class AdapterCreateRequest(BaseModel):
    """Request model for creating adapters"""
    task_tags: List[str]
    parameters: Optional[Dict[str, Any]] = None
    parent_ids: Optional[List[str]] = None
    y_bits: Optional[List[int]] = None
    z_bits: Optional[List[int]] = None
    x_bits: Optional[List[int]] = None


class ExperimentRequest(BaseModel):
    """Request model for quantum experiments"""
    experiment_type: str = Field(..., description="Type of experiment to run")
    iterations: int = Field(default=1000, ge=1, le=10000)
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    mode: str
    adapters_count: int
    artifacts_count: int
    timestamp: float


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration"""
    mode: Optional[str] = None
    quantum_enabled: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """Global configuration manager"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def update_mode(self, mode: str):
        """Update deployment mode"""
        valid_modes = ["low_power", "standard", "jetson_orin"]
        if mode in valid_modes:
            self.config["engine"]["mode"] = mode
            self.save_config()
            return True
        return False
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration"""
        return {
            "engine": {"name": "JARVIS-2v", "version": "2.0.0", "mode": "standard"},
            "model": {
                "path": "./models/jarvis-7b-q4_0.gguf",
                "context_size": 2048,
                "temperature": 0.7,
                "gpu_layers": 0,
                "device": "cpu"
            },
            "adapters": {
                "storage_path": "./adapters",
                "graph_path": "./adapters_graph.json",
                "auto_create": True,
                "freeze_after_creation": True
            },
            "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8},
            "quantum": {
                "artifacts_path": "./quantum_artifacts",
                "simulation_mode": True,
                "max_iterations": 1000,
                "noise_level": 0.1
            },
            "api": {"host": "0.0.0.0", "port": 8000, "enable_cors": True}
        }


# ============================================================================
# Main JARVIS API Application
# ============================================================================

class JarvisAPI:
    """Main JARVIS-2v API application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path or "./config.yaml")
        self.config = self.config_manager.config
        
        self.app = FastAPI(
            title="JARVIS-2v API",
            description="Modular Edge AI & Synthetic Quantum Lab",
            version=self.config.get("engine", {}).get("version", "2.0.0"),
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.adapter_engine: Optional[AdapterEngine] = None
        self.quantum_engine: Optional[SyntheticQuantumEngine] = None
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """Health check endpoint - returns system status"""
            artifacts_count = 0
            try:
                artifacts_path = Path(self.config.get("quantum", {}).get("artifacts_path", "./quantum_artifacts"))
                if artifacts_path.exists():
                    artifacts_count = len(list(artifacts_path.glob("*.json"))) - 1  # Exclude registry.json
            except:
                pass
            
            return HealthResponse(
                status="ok",
                version=self.config.get("engine", {}).get("version", "2.0.0"),
                mode=self.config.get("engine", {}).get("mode", "standard"),
                adapters_count=len(self.adapter_engine.list_adapters()) if self.adapter_engine else 0,
                artifacts_count=max(0, artifacts_count),
                timestamp=time.time()
            )
        
        @self.app.post("/api/infer", response_model=InferResponse, tags=["Inference"])
        async def infer(request: InferRequest):
            """
            Run inference through the adapter engine
            
            This endpoint routes your query through the Y/Z/X bit routing system
            to find the best adapters and generate a response.
            """
            start_time = time.time()
            
            try:
                # Prepare context with features
                context = request.context.copy()
                if request.features:
                    context["features"] = request.features
                
                # Route task to adapters
                adapters = self.adapter_engine.route_task(request.query, context)
                
                # Infer bit patterns
                y_bits, z_bits, x_bits = self.adapter_engine.bit_router.infer_bits_from_input(
                    request.query, context
                )
                
                # Update adapter metrics
                for adapter in adapters:
                    adapter.total_calls += 1
                    adapter.success_count += 1
                    adapter.last_used = time.time()
                    self.adapter_engine._save_adapter(adapter)
                
                # Generate mock response (replace with actual LLM if available)
                response_text = self._generate_response(request.query, adapters)
                
                processing_time = time.time() - start_time
                
                return InferResponse(
                    response=response_text,
                    adapters_used=[a.id for a in adapters[:3]],
                    bit_patterns={
                        "y_bits": y_bits,
                        "z_bits": z_bits,
                        "x_bits": x_bits
                    },
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/adapters", tags=["Adapters"])
        async def list_adapters(status: Optional[str] = Query(None, description="Filter by status: active, frozen, deprecated")):
            """
            List all adapters with their metadata and metrics
            
            Optionally filter by status (active, frozen, deprecated)
            """
            try:
                if status:
                    try:
                        adapter_status = AdapterStatus(status)
                        adapters = self.adapter_engine.list_adapters(status=adapter_status)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
                else:
                    adapters = self.adapter_engine.list_adapters()
                
                # Sort by last used (most recent first)
                adapters.sort(key=lambda a: a.last_used, reverse=True)
                
                adapter_list = []
                for adapter in adapters:
                    adapter_dict = adapter.to_dict()
                    # Add computed metrics
                    success_rate = (adapter.success_count / adapter.total_calls * 100) if adapter.total_calls > 0 else 0
                    adapter_dict["success_rate"] = round(success_rate, 2)
                    adapter_list.append(adapter_dict)
                
                return {
                    "adapters": adapter_list,
                    "total": len(adapter_list)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"List adapters error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/adapters", tags=["Adapters"])
        async def create_adapter(request: AdapterCreateRequest):
            """
            Create a new adapter with specified task tags and parameters
            
            Y/Z/X bits will be auto-inferred if not provided
            """
            try:
                # Auto-infer bit patterns if not provided
                y_bits = request.y_bits or [0] * self.config.get("bits", {}).get("y_bits", 16)
                z_bits = request.z_bits or [0] * self.config.get("bits", {}).get("z_bits", 8)
                x_bits = request.x_bits or [0] * self.config.get("bits", {}).get("x_bits", 8)
                
                adapter = self.adapter_engine.create_adapter(
                    task_tags=request.task_tags,
                    y_bits=y_bits,
                    z_bits=z_bits,
                    x_bits=x_bits,
                    parameters=request.parameters or {},
                    parent_ids=request.parent_ids or []
                )
                
                return {
                    "adapter_id": adapter.id,
                    "status": "created",
                    "adapter": adapter.to_dict()
                }
                
            except Exception as e:
                logger.error(f"Adapter creation error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/adapters/{adapter_id}", tags=["Adapters"])
        async def get_adapter(adapter_id: str):
            """Get details of a specific adapter"""
            try:
                adapter = self.adapter_engine.get_adapter(adapter_id)
                if not adapter:
                    raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
                
                adapter_dict = adapter.to_dict()
                success_rate = (adapter.success_count / adapter.total_calls * 100) if adapter.total_calls > 0 else 0
                adapter_dict["success_rate"] = round(success_rate, 2)
                
                return adapter_dict
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get adapter error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/quantum/experiment", tags=["Quantum"])
        async def run_quantum_experiment(request: ExperimentRequest):
            """
            Run a synthetic quantum experiment
            
            Available experiment types:
            - interference_experiment
            - bell_pair_simulation
            - chsh_test
            - noise_field_scan
            """
            try:
                config = ExperimentConfig(
                    experiment_type=request.experiment_type,
                    iterations=request.iterations,
                    noise_level=request.noise_level,
                    seed=request.seed,
                    parameters=request.parameters
                )
                
                # Route to appropriate experiment
                if request.experiment_type == "interference_experiment":
                    artifact = self.quantum_engine.run_interference_experiment(config)
                elif request.experiment_type == "bell_pair_simulation":
                    artifact = self.quantum_engine.run_bell_pair_simulation(config)
                elif request.experiment_type == "chsh_test":
                    artifact = self.quantum_engine.run_chsh_test(config)
                elif request.experiment_type == "noise_field_scan":
                    artifact = self.quantum_engine.run_noise_field_scan(config)
                else:
                    available = self.quantum_engine.list_experiments()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown experiment type. Available: {available}"
                    )
                
                return {
                    "artifact_id": artifact.artifact_id,
                    "experiment_type": artifact.experiment_type,
                    "created_at": artifact.created_at,
                    "linked_adapters": artifact.linked_adapter_ids,
                    "results_summary": artifact.results.get("statistics", {})
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Quantum experiment error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/artifacts", tags=["Quantum"])
        async def list_artifacts():
            """List all quantum artifacts"""
            try:
                artifacts_path = Path(self.config.get("quantum", {}).get("artifacts_path", "./quantum_artifacts"))
                
                if not artifacts_path.exists():
                    return {"artifacts": [], "total": 0}
                
                artifacts = []
                for artifact_file in artifacts_path.glob("*.json"):
                    if artifact_file.name == "registry.json":
                        continue
                    
                    try:
                        with open(artifact_file, 'r') as f:
                            import json
                            artifact_data = json.load(f)
                            artifacts.append({
                                "artifact_id": artifact_data["artifact_id"],
                                "experiment_type": artifact_data["experiment_type"],
                                "created_at": artifact_data["created_at"],
                                "linked_adapters": artifact_data["linked_adapter_ids"],
                                "statistics": artifact_data.get("results", {}).get("statistics", {})
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load artifact {artifact_file}: {e}")
                        continue
                
                # Sort by created_at (most recent first)
                artifacts.sort(key=lambda a: a["created_at"], reverse=True)
                
                return {
                    "artifacts": artifacts,
                    "total": len(artifacts)
                }
                
            except Exception as e:
                logger.error(f"List artifacts error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/artifacts/{artifact_id}", tags=["Quantum"])
        async def get_artifact(artifact_id: str):
            """Get full details of a specific quantum artifact"""
            try:
                artifact = self.quantum_engine.replay_artifact(artifact_id)
                if not artifact:
                    raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
                
                return artifact.to_dict()
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get artifact error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/config", tags=["System"])
        async def update_config(request: ConfigUpdateRequest):
            """Update runtime configuration"""
            try:
                updated = {}
                
                if request.mode is not None:
                    if self.config_manager.update_mode(request.mode):
                        updated["mode"] = request.mode
                        self.config = self.config_manager.config
                    else:
                        raise HTTPException(status_code=400, detail="Invalid mode")
                
                if request.quantum_enabled is not None:
                    self.config["quantum"]["simulation_mode"] = request.quantum_enabled
                    self.config_manager.save_config()
                    updated["quantum_enabled"] = request.quantum_enabled
                
                if request.settings:
                    for key, value in request.settings.items():
                        if key in self.config:
                            self.config[key].update(value)
                    self.config_manager.save_config()
                    updated["settings"] = request.settings
                
                return {
                    "status": "updated",
                    "changes": updated
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Config update error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/config", tags=["System"])
        async def get_config():
            """Get current configuration"""
            return self.config
    
    def _generate_response(self, query: str, adapters: List[Any]) -> str:
        """Generate mock response (replace with actual LLM inference)"""
        adapter_info = f" (using adapters: {[a.id for a in adapters[:2]]})" if adapters else ""
        
        # Simple pattern matching for demo
        query_lower = query.lower()
        
        if "quantum" in query_lower or "experiment" in query_lower:
            return f"I can help you run quantum experiments{adapter_info}. Available experiment types include interference patterns, Bell pair simulations, CHSH tests, and noise field scans. Would you like me to run one?"
        
        if "adapter" in query_lower:
            return f"The adapter engine uses Y/Z/X bit routing to select specialized modules{adapter_info}. I currently have {len(self.adapter_engine.list_adapters())} adapters available for various tasks."
        
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return f"Hello! I'm J.A.R.V.I.S. 2v, your modular AI assistant{adapter_info}. I can help with quantum experiments, adapter management, and general queries."
        
        return f"I understand your query: '{query}'{adapter_info}. I'm currently running in demonstration mode. For full capabilities, connect an LLM model."
    
    async def initialize(self):
        """Initialize JARVIS-2v API components"""
        try:
            logger.info("🚀 Initializing JARVIS-2v API...")
            
            # Initialize adapter engine
            logger.info("📦 Initializing Adapter Engine...")
            self.adapter_engine = AdapterEngine(self.config)
            logger.info(f"✓ Adapter Engine ready with {len(self.adapter_engine.list_adapters())} adapters")
            
            # Initialize quantum engine
            logger.info("⚛️  Initializing Synthetic Quantum Engine...")
            quantum_config = self.config.get("quantum", {})
            self.quantum_engine = SyntheticQuantumEngine(
                quantum_config.get("artifacts_path", "./quantum_artifacts"),
                self.adapter_engine
            )
            logger.info("✓ Quantum Engine ready")
            
            logger.info("✅ JARVIS-2v API initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize JARVIS-2v API: {e}", exc_info=True)
            raise
    
    def get_app(self):
        """Get FastAPI application instance"""
        return self.app


# ============================================================================
# Application Factory & Entry Point
# ============================================================================

# Global JARVIS instance
_jarvis_instance: Optional[JarvisAPI] = None


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Factory function to create JARVIS-2v API instance"""
    global _jarvis_instance
    _jarvis_instance = JarvisAPI(config_path)
    
    @_jarvis_instance.app.on_event("startup")
    async def startup_event():
        """Initialize components on startup"""
        await _jarvis_instance.initialize()
    
    return _jarvis_instance.get_app()


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    # Get configuration
    config_path = os.getenv("JARVIS_CONFIG", "./config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {"api": {"host": "0.0.0.0", "port": 8000}}
    
    port = int(os.getenv("PORT", config.get("api", {}).get("port", 8000)))
    host = os.getenv("HOST", config.get("api", {}).get("host", "0.0.0.0"))
    
    logger.info(f"🌟 Starting JARVIS-2v API server on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
