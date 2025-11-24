from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis5090x import (
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
    PhaseDetector,
)
from experiments.discovery_suite import (
    run_time_reversal_test,
    unsupervised_phase_discovery,
    replay_drift_scaling,
)

devices = [
    AdapterDevice(
        id="quantum_0",
        label="Quantum Simulator",
        kind=DeviceKind.VIRTUAL,
        perf_score=50.0,
        max_concurrency=8,
        capabilities={OperationKind.QUANTUM},
    ),
]
orchestrator = Jarvis5090X(devices)
detector = PhaseDetector(orchestrator)

app = FastAPI(
    title="Jarvis Lab API",
    description="REST API for Jarvis-2v quantum phase detector and discovery suite",
    version="1.0.0",
)


class PhaseRequest(BaseModel):
    phase_type: str = Field(
        default="ising_symmetry_breaking",
        description="Phase type: ising_symmetry_breaking, spt_cluster, trivial_product, pseudorandom",
    )
    system_size: int = Field(default=32, ge=4, le=256, description="System size (4-256)")
    depth: int = Field(default=8, ge=1, le=32, description="Circuit depth (1-32)")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    bias: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Bias parameter (0.0-1.0)")


class TRIRequest(BaseModel):
    phase_type: str = Field(default="ising_symmetry_breaking", description="Phase type")
    system_size: int = Field(default=32, ge=4, le=256, description="System size")
    depth: int = Field(default=8, ge=1, le=32, description="Circuit depth")
    bias: float = Field(default=0.7, ge=0.0, le=1.0, description="Forward bias parameter")
    seed: int = Field(default=42, description="Random seed")


class DiscoveryRequest(BaseModel):
    phases: list[str] = Field(
        default=["ising_symmetry_breaking", "spt_cluster", "trivial_product", "pseudorandom"],
        description="List of phase types to discover",
    )
    num_per_phase: int = Field(default=20, ge=5, le=100, description="Samples per phase")
    k: Optional[int] = Field(default=None, ge=2, le=10, description="Number of clusters (defaults to len(phases))")
    iterations: int = Field(default=25, ge=10, le=100, description="K-means iterations")


class ReplayDriftRequest(BaseModel):
    phase_type: str = Field(default="ising_symmetry_breaking", description="Phase type")
    system_size: int = Field(default=32, ge=4, le=256, description="System size")
    base_depth: int = Field(default=6, ge=1, le=32, description="Base depth")
    seed: int = Field(default=123, description="Random seed")
    depth_factors: list[int] = Field(default=[1, 2, 3], description="Depth multipliers")


@app.get("/")
def root():
    return {
        "service": "Jarvis Lab API",
        "version": "1.0.0",
        "description": "Quantum phase detector and discovery suite API",
        "endpoints": {
            "/run_phase_experiment": "Run a single phase experiment",
            "/tri": "Time-Reversal Instability test",
            "/discovery": "Unsupervised phase discovery (k-means clustering)",
            "/replay_drift": "Replay drift scaling experiment",
            "/health": "Health check endpoint",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "lab": "operational", "devices": len(devices)}


@app.post("/run_phase_experiment")
def run_phase_experiment(req: PhaseRequest) -> Dict[str, Any]:
    try:
        kwargs = req.model_dump()
        if kwargs["bias"] is None:
            kwargs.pop("bias")
        result = detector.run_phase_experiment(**kwargs)
        return {
            "experiment_id": result["experiment_id"],
            "phase_type": result["phase_type"],
            "feature_vector": result["feature_vector"],
            "summary": result["summary"],
            "params": {
                "phase_type": req.phase_type,
                "system_size": req.system_size,
                "depth": req.depth,
                "seed": req.seed,
                "bias": req.bias,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}")


@app.post("/tri")
def tri_endpoint(req: TRIRequest) -> Dict[str, Any]:
    try:
        result = run_time_reversal_test(
            detector,
            phase_type=req.phase_type,
            system_size=req.system_size,
            depth=req.depth,
            bias=req.bias,
            seed=req.seed,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TRI experiment failed: {str(e)}")


@app.post("/discovery")
def discovery_endpoint(req: DiscoveryRequest) -> Dict[str, Any]:
    try:
        k = req.k if req.k is not None else len(req.phases)
        result = unsupervised_phase_discovery(
            detector,
            phases=req.phases,
            num_per_phase=req.num_per_phase,
            k=k,
            iterations=req.iterations,
        )
        return {
            "cluster_label_stats": result["cluster_label_stats"],
            "num_samples": len(result["samples"]),
            "num_clusters": len(result["centroids"]),
            "phases": req.phases,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@app.post("/replay_drift")
def replay_drift_endpoint(req: ReplayDriftRequest) -> Dict[str, Any]:
    try:
        runs = replay_drift_scaling(
            detector,
            phase_type=req.phase_type,
            system_size=req.system_size,
            base_depth=req.base_depth,
            seed=req.seed,
            depth_factors=req.depth_factors,
        )
        return {
            "phase_type": req.phase_type,
            "base_depth": req.base_depth,
            "runs": runs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Replay drift failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Jarvis Lab API...")
    print("📡 Endpoints available at http://127.0.0.1:8000")
    print("📚 API docs at http://127.0.0.1:8000/docs")
    print("🔬 Quantum detector initialized with {} devices".format(len(devices)))
    uvicorn.run(app, host="127.0.0.1", port=8000)
