"""
JARVIS-2v Main API Server
FastAPI-based REST API with modular architecture
"""

import os
import sys
import time
import yaml
import uvicorn
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from ..core.adapter_engine import AdapterEngine, YZXBitRouter, AdapterStatus
from ..quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from ...inference import JarvisInferenceBackend, load_memory, save_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    messages: List[Dict[str, str]]
    session_id: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    message: Dict[str, Any]
    usage: Dict[str, Any]
    performance: Dict[str, Any]
    adapters_used: List[str] = Field(default_factory=list)
    quantum_context: Optional[str] = None


class AdapterRequest(BaseModel):
    """Request model for adapter operations"""
    task_tags: List[str]
    parameters: Optional[Dict[str, Any]] = None
    parent_ids: Optional[List[str]] = None
    y_bits: Optional[List[int]] = None
    z_bits: Optional[List[int]] = None
    x_bits: Optional[List[int]] = None


class ExperimentRequest(BaseModel):
    """Request model for quantum experiments"""
    experiment_type: str
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    llm_ready: bool
    version: str
    mode: str
    adapters_count: int


class Config:
    """Global configuration manager"""
    _instance = None
    
    @classmethod
    def load(cls, config_path: str = "./config.yaml") -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls._default_config()
    
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
            "adapters": {"storage_path": "./adapters", "auto_create": True},
            "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8},
            "api": {"host": "0.0.0.0", "port": 3001}
        }


class JarvisAPI:
    """Main JARVIS-2v API application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config.load(config_path or "./config.yaml")
        self.app = FastAPI(
            title="JARVIS-2v API",
            description="Modular Edge AI & Synthetic Quantum Lab",
            version=self.config.get("engine", {}).get("version", "2.0.0")
        )
        
        # Initialize components
        self.llm_engine = None
        self.adapter_engine = None
        self.quantum_engine = None
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if self.config.get("api", {}).get("enable_cors", True) else [],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "llm_ready": self.llm_engine and self.llm_engine.is_initialized if self.llm_engine else False,
                "version": self.config.get("engine", {}).get("version", "2.0.0"),
                "mode": self.config.get("engine", {}).get("mode", "standard"),
                "adapters_count": len(self.adapter_engine.list_adapters()) if self.adapter_engine else 0
            }
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Main chat endpoint with adapter routing"""
            if not self.llm_engine or not self.llm_engine.is_initialized:
                raise HTTPException(status_code=503, detail="LLM engine not ready")
            
            try:
                # Route task to adapters
                last_message = request.messages[-1]["content"] if request.messages else ""
                adapters = self.adapter_engine.route_task(last_message, request.options)
                
                # Generate response with adapters as context
                adapted_prompt = self._enrich_with_adapters(request.messages, adapters)
                result = self.llm_engine.chat(adapted_prompt, **request.options)
                
                # Update adapter metrics
                for adapter in adapters:
                    adapter.total_calls += 1
                    # Success if we got a reasonable response
                    adapter.success_count += 1 if result.get("message", {}).get("content") else 0
                
                return ChatResponse(
                    message=result.get("message", {}),
                    usage=result.get("usage", {}),
                    performance=result.get("performance", {}),
                    adapters_used=[a.id for a in adapters[:2]]
                )
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/adapters")
        async def create_adapter(request: AdapterRequest):
            """Create new adapter"""
            try:
                # Auto-infer bit patterns if not provided
                if not request.y_bits:
                    y_bits = [0] * self.config.get("bits", {}).get("y_bits", 16)
                    z_bits = [0] * self.config.get("bits", {}).get("z_bits", 8)
                    x_bits = [0] * self.config.get("bits", {}).get("x_bits", 8)
                else:
                    y_bits = request.y_bits
                    z_bits = request.z_bits or [0] * 8
                    x_bits = request.x_bits or [0] * 8
                
                adapter = self.adapter_engine.create_adapter(
                    task_tags=request.task_tags,
                    y_bits=y_bits,
                    z_bits=z_bits,
                    x_bits=x_bits,
                    parameters=request.parameters or {},
                    parent_ids=request.parent_ids or []
                )
                
                return {"adapter_id": adapter.id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Adapter creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/adapters")
        async def list_adapters(status: Optional[str] = None):
            """List adapters"""
            try:
                if status:
                    adapters = self.adapter_engine.list_adapters(status=AdapterStatus(status))
                else:
                    adapters = self.adapter_engine.list_adapters()
                
                return {
                    "adapters": [a.to_dict() for a in adapters],
                    "total": len(adapters)
                }
            except Exception as e:
                logger.error(f"List adapters error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/quantum/experiment")
        async def run_quantum_experiment(request: ExperimentRequest):
            """Run quantum experiment and generate artifact"""
            try:
                config = ExperimentConfig(experiment_type=request.experiment_type, **request.config)
                
                if request.experiment_type == "interference_experiment":
                    artifact = self.quantum_engine.run_interference_experiment(config)
                elif request.experiment_type == "bell_pair_simulation":
                    artifact = self.quantum_engine.run_bell_pair_simulation(config)
                elif request.experiment_type == "chsh_test":
                    artifact = self.quantum_engine.run_chsh_test(config)
                elif request.experiment_type == "noise_field_scan":
                    artifact = self.quantum_engine.run_noise_field_scan(config)
                elif request.experiment_type == "negative_information_experiment":
                    artifact = self.quantum_engine.run_negative_information_experiment(config)
                else:
                    raise HTTPException(status_code=400, detail="Unknown experiment type")
                
                return {
                    "artifact_id": artifact.artifact_id,
                    "experiment_type": artifact.experiment_type,
                    "created_at": artifact.created_at,
                    "linked_adapters": artifact.linked_adapter_ids
                }
                
            except Exception as e:
                logger.error(f"Quantum experiment error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/quantum/replay")
        async def replay_artifact(artifact_id: str):
            """Replay quantum artifact"""
            try:
                artifact = self.quantum_engine.replay_artifact(artifact_id)
                if not artifact:
                    raise HTTPException(status_code=404, detail="Artifact not found")
                return artifact.to_dict()
            except Exception as e:
                logger.error(f"Replay artifact error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/quantum/context")
        async def artifact_context(artifact_id: str, query: str):
            """Use artifact as context for queries"""
            try:
                context = self.quantum_engine.use_artifact_as_context(artifact_id, query)
                return {"context": context}
            except Exception as e:
                logger.error(f"Artifact context error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/memory")
        async def get_memory():
            """Get memory contents"""
            memory = load_memory()
            return memory
        
        @self.app.post("/memory/clear")
        async def clear_memory():
            """Clear memory"""
            try:
                empty_memory = {
                    "facts": [],
                    "chats": [],
                    "topics": {},
                    "preferences": {},
                    "last_topics": []
                }
                save_memory(empty_memory)
                return {"status": "cleared"}
            except Exception as e:
                logger.error(f"Clear memory error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _enrich_with_adapters(self, messages: List[Dict[str, str]], adapters: List[Any]) -> List[Dict[str, str]]:
        """Enrich messages with adapter context"""
        if not adapters:
            return messages
        
        # Add adapter context to system prompt
        system_prompt = f"""You are J.A.R.V.I.S. with modular adapter enhancements.
        
Active Adapters: {[a.id for a in adapters[:2]]}
Adapter Capabilities: {[a.task_tags for a in adapters[:2]]}

Use these specialized modules to enhance your responses."""
        
        enriched = messages.copy()
        if enriched and enriched[0]["role"] == "system":
            enriched[0]["content"] = system_prompt + "\n\n" + enriched[0]["content"]
        else:
            enriched.insert(0, {"role": "system", "content": system_prompt})
        
        return enriched
    
    async def initialize(self):
        """Initialize JARVIS-2v API components"""
        try:
            logger.info("Initializing JARVIS-2v API...")
            
            # Initialize LLM engine
            model_path = self.config.get("model", {}).get("path", "./models/jarvis-7b-q4_0.gguf")
            llm_config = {
                "context_size": self.config.get("model", {}).get("context_size", 2048),
                "temperature": self.config.get("model", {}).get("temperature", 0.7),
                "gpu_layers": self.config.get("model", {}).get("gpu_layers", 0)
            }
            
            self.llm_engine = JarvisInferenceBackend(model_path, llm_config)
            if not self.llm_engine.initialize():
                logger.warning("LLM engine initialization failed, continuing in degraded mode")
            
            # Initialize adapter engine
            self.adapter_engine = AdapterEngine(self.config)
            
            # Initialize quantum engine
            quantum_config = self.config.get("quantum", {})
            self.quantum_engine = SyntheticQuantumEngine(
                quantum_config.get("artifacts_path", "./quantum_artifacts"),
                self.adapter_engine
            )
            
            logger.info("JARVIS-2v API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize JARVIS-2v API: {e}")
            raise
    
    def run(self):
        """Run the API server"""
        port = self.config.get("api", {}).get("port", 3001)
        host = self.config.get("api", {}).get("host", "0.0.0.0")
        
        logger.info(f"Starting JARVIS-2v API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_app(config_path: Optional[str] = None) -> JarvisAPI:
    """Factory function to create JARVIS-2v API instance"""
    return JarvisAPI(config_path)


if __name__ == "__main__":
    import asyncio
    
    # Create and initialize app
    app = create_app()
    
    # Initialize components
    asyncio.run(app.initialize())
    
    # Run server
    app.run()