# JARVIS-2v Backend API

FastAPI-based REST API for the JARVIS-2v modular AI system.

## Features

- **Adapter Engine**: Y/Z/X bit routing system for modular AI
- **Quantum Lab**: Synthetic quantum experiments with artifact generation
- **Edge-Friendly**: Lightweight, CPU-only mode available
- **Auto-Documentation**: OpenAPI/Swagger docs at `/docs`

## Quick Start

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Run Server

```bash
# Development mode
python main.py

# Production mode with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# With environment variables
PORT=3001 HOST=127.0.0.1 python main.py
```

### Environment Variables

- `PORT`: API server port (default: 8000)
- `HOST`: API server host (default: 0.0.0.0)
- `JARVIS_CONFIG`: Path to config.yaml (default: ./config.yaml)

## API Endpoints

### System

- `GET /health` - Health check and system status
- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration

### Inference

- `POST /api/infer` - Run inference through adapter engine

### Adapters

- `GET /api/adapters` - List all adapters
- `POST /api/adapters` - Create new adapter
- `GET /api/adapters/{id}` - Get adapter details

### Quantum Lab

- `POST /api/quantum/experiment` - Run quantum experiment
- `GET /api/artifacts` - List quantum artifacts
- `GET /api/artifacts/{id}` - Get artifact details

## Documentation

- Interactive API docs: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc

## Configuration

Edit `config.yaml` in the project root to configure:

- Deployment mode (low_power, standard, jetson_orin)
- Adapter system settings
- Y/Z/X bit dimensions
- Quantum simulation parameters
- API server settings

## Development

### Project Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

The backend imports core modules from `src/`:
- `src/core/adapter_engine.py` - Adapter system
- `src/quantum/synthetic_quantum.py` - Quantum experiments

### Testing API

```bash
# Health check
curl http://localhost:8000/health

# Run inference
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello JARVIS", "context": {}}'

# List adapters
curl http://localhost:8000/api/adapters

# Run quantum experiment
curl -X POST http://localhost:8000/api/quantum/experiment \
  -H "Content-Type: application/json" \
  -d '{"experiment_type": "interference_experiment", "iterations": 1000, "noise_level": 0.1}'
```

## Deployment

See the main project README and `docs/DEPLOYMENT.md` for deployment instructions for:

- Vercel (serverless functions)
- Netlify (Edge Functions)
- Docker (self-hosted)
- shiper.app (containerized deployment)
