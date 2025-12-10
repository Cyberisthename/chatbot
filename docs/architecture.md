# Architecture Overview

## High-Level System Architecture

```
User
  ↓
React Dashboard (UI Layer)
  ↓
FastAPI REST API (src/api/main.py)
  ↓
┌─────────────────────────────────────┐
│         Core Processing Layer        │
├─────────────────────────────────────┤
│  Adapter Engine  │  Y/Z/X Router  │
│  YZXBitRouter   │  AdapterGraph  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│         Service Layer              │
├─────────────────────────────────────┤
│  LLM Engine    │  Quantum Module │
│  Memory Mgmt   │  Artifact Store │
└─────────────────────────────────────┘
  ↓
Persistence Layer (JSON, SQLite)
```

## Key Components

### 1. Adapter Engine (src/core/adapter_engine.py)
- **Adapter**: Modular AI module with Y/Z/X bit patterns, metrics, and relationships
- **AdapterGraph**: NetworkX-based directed graph for adapter relationships
- **YZXBitRouter**: Bit vector-based routing system with 16/8/8 bits
- **AdapterEngine**: Main orchestrator for adapter lifecycle and routing

### 2. Quantum Module (src/quantum/synthetic_quantum.py)
- **SyntheticQuantumEngine**: Runs synthetic quantum experiments
- **ExperimentConfig**: Configuration for different experiment types
- **QuantumArtifact**: Stores experiment results with adapter linkage
- **Experiments**: Interference, Bell pair, CHSH test, noise field scan

### 3. FastAPI Server (src/api/main.py)
- **RESTful endpoints**: /chat, /adapters, /quantum, /health
- **Pydantic models**: Type-safe request/response validation
- **Middleware**: CORS, error handling, logging
- **Async support**: Non-blocking I/O operations

### 4. React Dashboard (src/ui/)
- **Modern UI**: Next.js + TypeScript + Tailwind CSS
- **Real-time updates**: WebSocket/SSE integration
- **Visualization**: Adapter graph, quantum results, metrics
- **Command palette**: Quick access to system commands

## Data Flow

### Chat Request Flow
1. User sends message via UI or API
2. System infers Y/Z/X bits from input
3. Router selects best matching adapters
4. Adapters enrich context and prompt
5. LLM generates response with adapter insights
6. Response returned with adapter attribution
7. Metrics updated for used adapters

### Quantum Experiment Flow
1. Experiment requested via API
2. SyntheticQuantumEngine runs simulation
3. Results stored as QuantumArtifact
4. New adapter created from artifact
5. Adapter linked to artifact for future reference
6. Results visualized in dashboard

## Design Principles

### Non-Destructive Learning
- Every new task creates new adapters
- Existing adapters frozen (immutable)
- No catastrophic forgetting
- Progressive network architecture

### Explainable AI
- All routing decisions logged
- Bit pattern similarity scores
- Adapter success metrics
- Transparent reasoning chain

### Edge Optimization
- Modular architecture for resource constraints
- Configurable GPU layers and memory usage
- Offline operation support
- Low-power mode for battery operation

## Security Considerations

- Input validation with Pydantic models
- CORS configuration for web access
- Rate limiting (can be added)
- Secure file operations with Path objects
- No external dependencies for offline mode

## Scalability

- Horizontal scaling: Multiple API instances behind load balancer
- Vertical scaling: GPU layer tuning, memory optimization
- Adapter deduplication: Merge similar adapters over time
- Lazy loading: Adapters loaded on demand

## Future Extensions

- Plugin system for external tools
- Federated adapter exchange between instances
- Auto-curriculum generation
- Hardware acceleration for bit operations
- Distributed graph database for large-scale adapter storage