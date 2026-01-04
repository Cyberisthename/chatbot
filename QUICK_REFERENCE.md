# JARVIS-2v Quick Reference Card

One-page reference for developers working with JARVIS-2v.

---

## 🚀 Quick Commands

```bash
# Start everything locally
./scripts/start_all_local.sh

# Start backend only
./scripts/start_backend.sh

# Start frontend only
./scripts/start_frontend.sh

# Docker
docker-compose up -d        # Start
docker-compose down         # Stop
docker-compose logs -f      # View logs

# Testing
curl http://localhost:8000/health
curl http://localhost:3000
```

---

## 📁 Project Structure

```
jarvis-2v/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # App Router pages
│   │   ├── page.tsx       # Dashboard
│   │   ├── adapters/      # Adapters page
│   │   ├── quantum/       # Quantum Lab
│   │   ├── console/       # Chat console
│   │   └── settings/      # Settings
│   ├── components/        # React components
│   └── lib/              # API client & utilities
├── src/                   # Core engine
│   ├── core/             # Adapter engine
│   └── quantum/          # Quantum engine
├── scripts/              # Startup scripts
└── docs/                 # Documentation
```

---

## 🔌 API Endpoints

### System
- `GET /health` - Health check
- `GET /api/config` - Get configuration
- `POST /api/config` - Update configuration

### Inference
- `POST /api/infer` - Run inference
  ```json
  {"query": "Your question", "context": {}, "features": []}
  ```

### Adapters
- `GET /api/adapters` - List adapters
- `POST /api/adapters` - Create adapter
- `GET /api/adapters/{id}` - Get adapter details

### Quantum
- `POST /api/quantum/experiment` - Run experiment
  ```json
  {
    "experiment_type": "interference_experiment",
    "iterations": 1000,
    "noise_level": 0.1
  }
  ```
- `GET /api/artifacts` - List artifacts
- `GET /api/artifacts/{id}` - Get artifact details

---

## 🎨 Frontend Pages

- `/` - Dashboard (system overview)
- `/adapters` - Adapter management
- `/quantum` - Quantum Lab
- `/console` - Chat interface
- `/settings` - Configuration

---

## 🔧 Configuration

Located in `config.yaml`:

```yaml
engine:
  mode: "standard"  # low_power, standard, jetson_orin

adapters:
  storage_path: "./adapters"
  auto_create: true

quantum:
  artifacts_path: "./quantum_artifacts"
  simulation_mode: true

bits:
  y_bits: 16  # Task/domain
  z_bits: 8   # Difficulty
  x_bits: 8   # Experimental
```

---

## 🐍 Core Classes

### AdapterEngine
```python
from src.core.adapter_engine import AdapterEngine

engine = AdapterEngine(config)
adapters = engine.route_task("query", context)
adapter = engine.create_adapter(task_tags, y_bits, z_bits, x_bits)
```

### SyntheticQuantumEngine
```python
from src.quantum.synthetic_quantum import SyntheticQuantumEngine

qe = SyntheticQuantumEngine(artifacts_path, adapter_engine)
artifact = qe.run_interference_experiment(config)
```

---

## 🌐 Environment Variables

### Backend
```bash
HOST=0.0.0.0
PORT=8000
JARVIS_CONFIG=./config.yaml
```

### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📦 Dependencies

### Backend (Python)
- fastapi - Web framework
- uvicorn - ASGI server
- pyyaml - Config parsing
- numpy - Numerical computing
- networkx - Graph operations

### Frontend (Node.js)
- next - React framework
- react - UI library
- tailwindcss - Styling
- lucide-react - Icons

---

## 🐳 Docker

### docker-compose.yml
```yaml
services:
  backend:
    build: Dockerfile.backend
    ports: ["8000:8000"]
  frontend:
    build: Dockerfile.frontend
    ports: ["3000:3000"]
```

### Single Container
```bash
docker build -t jarvis-2v .
docker run -p 8000:8000 -p 3000:3000 jarvis-2v
```

---

## 🔍 Debugging

### Backend Logs
```bash
# Check if running
ps aux | grep uvicorn

# View logs
tail -f /tmp/backend.log

# Test health
curl http://localhost:8000/health
```

### Frontend Logs
```bash
# Check if running
ps aux | grep next

# View browser console (F12)

# Test API connection
curl http://localhost:3000
```

---

## 🧪 Testing

### Test Backend
```bash
# Start backend
./scripts/start_backend.sh

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}'
```

### Test Frontend
```bash
cd frontend
npm run build  # Should complete without errors
npm run dev    # Start dev server
```

---

## 🚨 Common Issues

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Module not found
```bash
# Backend
pip install --break-system-packages -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### Cannot connect to backend
```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS (should be enabled by default)
# Restart backend if needed
```

---

## 📊 Bit Routing System

### Y Bits (Task/Domain)
- Bits 0-3: Task type (code, text, data, etc.)
- Bits 4-7: Domain (science, web, system, etc.)
- Bits 8-15: Sub-domain specific

### Z Bits (Difficulty/Precision)
- Bits 0-3: Complexity level (0=simple, 15=expert)
- Bits 4-7: Precision requirement

### X Bits (Experimental Features)
- Toggle experimental features
- Enable/disable modules
- Feature flags

---

## 🎯 Use Cases

### 1. Run AI Inference
```bash
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum computing"}'
```

### 2. Create Specialized Adapter
```bash
curl -X POST http://localhost:8000/api/adapters \
  -H "Content-Type: application/json" \
  -d '{
    "task_tags": ["science", "physics"],
    "parameters": {"specialty": "quantum"}
  }'
```

### 3. Run Quantum Experiment
```bash
curl -X POST http://localhost:8000/api/quantum/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_type": "bell_pair_simulation",
    "iterations": 5000
  }'
```

---

## 🌍 Deployment

### Vercel (Frontend only)
```bash
vercel --prod
# Set NEXT_PUBLIC_API_URL in dashboard
```

### Railway (Backend)
```bash
railway init
railway up
```

### Docker (Full stack)
```bash
docker-compose up -d
```

---

## 📚 Documentation

- `README.md` - Main documentation
- `QUICKSTART.md` - 5-minute quick start
- `TESTING_GUIDE.md` - Testing procedures
- `DEPLOYMENT_PLATFORMS.md` - Deployment guides
- `docs/DEPLOYMENT.md` - Detailed deployment
- `backend/README.md` - Backend API docs
- `frontend/README.md` - Frontend docs

---

## 💡 Tips

1. **Always test locally first** with `./scripts/start_all_local.sh`
2. **Check logs** when something doesn't work
3. **Use `/docs`** endpoint for interactive API documentation
4. **CORS is enabled** by default for development
5. **Health check** endpoint is your friend: `/health`
6. **Docker is recommended** for production deployment

---

## 🆘 Get Help

1. Check logs: `tail -f /tmp/backend.log`
2. Review `TESTING_GUIDE.md`
3. Check API docs: `http://localhost:8000/docs`
4. Run health check: `curl http://localhost:8000/health`
5. Open GitHub issue with logs

---

## ✅ Quick Validation

```bash
# Backend running?
curl -f http://localhost:8000/health && echo "✅ Backend OK"

# Frontend building?
cd frontend && npm run build && echo "✅ Frontend OK"

# API working?
curl -f -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' && echo "✅ API OK"
```

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**License**: MIT  
**Repository**: https://github.com/Cyberisthename/chatbot
