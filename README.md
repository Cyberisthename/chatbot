# JARVIS-2v – Modular Edge AI & Synthetic Quantum Lab

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![Node](https://img.shields.io/badge/node-18%2B-green)

**A production-ready, modular AI system with adapter-based routing, synthetic quantum experiments, and a modern web interface – designed for edge devices and cloud deployment.**

---

## ✨ What's New in v2.0

- 🎨 **Modern Next.js Frontend** - Beautiful, responsive UI with Dashboard, Adapters, Quantum Lab, Console, and Settings
- 🚀 **Clean FastAPI Backend** - RESTful API with auto-generated docs
- 🐳 **Docker Support** - One-command deployment with Docker/Docker Compose
- ☁️ **Cloud-Ready** - Deploy to Vercel, Netlify, or shiper.app in minutes
- 📊 **Real-time Monitoring** - Live system metrics and adapter performance
- ⚛️ **Enhanced Quantum Lab** - Interactive experiment runner with artifact visualization

---

## 🚀 Quick Start

### Local Development (Easiest)

```bash
# Clone the repository
git clone https://github.com/Cyberisthename/chatbot.git
cd chatbot

# Start both backend and frontend
./scripts/start_all_local.sh
```

Access the app:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Start Services Separately

```bash
# Terminal 1: Backend
./scripts/start_backend.sh

# Terminal 2: Frontend
./scripts/start_frontend.sh
```

### Docker (One Command)

```bash
docker-compose up
```

---

## 🏗️ Architecture

### Modular Components

```
┌─────────────────────────────────────────────┐
│           Next.js Frontend (Port 3000)       │
│  Dashboard │ Adapters │ Quantum │ Console   │
└──────────────────┬──────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────┐
│         FastAPI Backend (Port 8000)          │
│   /health │ /api/infer │ /api/adapters      │
└──────────────────┬──────────────────────────┘
                   │
     ┌─────────────┴─────────────┐
     ▼                           ▼
┌─────────────┐         ┌─────────────────┐
│   Adapter   │         │    Quantum       │
│   Engine    │         │    Engine        │
│  (Y/Z/X)    │         │  (Artifacts)     │
└─────────────┘         └─────────────────┘
```

### Key Features

**🧠 Adapter Engine**
- Y/Z/X bit routing (16/8/8 dimensions)
- Non-destructive learning
- Graph-based relationships
- Success rate tracking

**⚛️ Quantum Module**
- Interference experiments
- Bell pair simulations
- CHSH inequality tests
- Noise field scans

**🎯 Edge-Ready**
- Lightweight & fast
- CPU-only mode
- Jetson Orin support
- Offline operation

---

## 📁 Project Structure

```
jarvis-2v/
├── backend/                    # FastAPI backend
│   ├── main.py                # API server
│   ├── requirements.txt       # Python deps
│   └── README.md
├── frontend/                   # Next.js frontend
│   ├── app/                   # Pages (App Router)
│   │   ├── page.tsx          # Dashboard
│   │   ├── adapters/         # Adapter management
│   │   ├── quantum/          # Quantum lab
│   │   ├── console/          # Chat interface
│   │   └── settings/         # System settings
│   ├── components/            # React components
│   ├── lib/                   # API client & utils
│   └── package.json
├── src/                       # Core Python modules
│   ├── core/                  # Adapter engine
│   ├── quantum/               # Quantum experiments
│   └── api/                   # (Legacy API - use backend/)
├── scripts/                   # Deployment scripts
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   └── start_all_local.sh
├── docs/                      # Documentation
│   └── DEPLOYMENT.md          # Deployment guide
├── Dockerfile                 # Production Docker image
├── docker-compose.yml         # Development setup
├── vercel.json               # Vercel configuration
├── netlify.toml              # Netlify configuration
└── config.yaml               # System configuration
```

---

## 🎯 Features

### Frontend Pages

**Dashboard** (`/`)
- System health status
- Active mode indicator
- Adapter & artifact counts
- Recent activity

**Adapters** (`/adapters`)
- View all adapters
- Create new adapters
- Filter by status
- View Y/Z/X bit patterns
- Success rate metrics

**Quantum Lab** (`/quantum`)
- Run experiments
- Configure parameters
- View artifacts
- Inspect results

**Console** (`/console`)
- Chat with JARVIS
- Real-time responses
- Adapter routing info
- Processing time metrics

**Settings** (`/settings`)
- Change deployment mode
- Toggle quantum features
- View system info
- Save configuration

### Backend API

```http
GET  /health                      # System status
POST /api/infer                   # Run inference
GET  /api/adapters                # List adapters
POST /api/adapters                # Create adapter
GET  /api/adapters/{id}           # Get adapter details
POST /api/quantum/experiment      # Run experiment
GET  /api/artifacts               # List artifacts
GET  /api/artifacts/{id}          # Get artifact details
GET  /api/config                  # Get configuration
POST /api/config                  # Update configuration
```

Full API docs: http://localhost:8000/docs

---

## 🔧 Configuration

Edit `config.yaml` for system settings:

```yaml
engine:
  name: "JARVIS-2v"
  version: "2.0.0"
  mode: "standard"  # low_power | standard | jetson_orin

adapters:
  storage_path: "./adapters"
  auto_create: true
  freeze_after_creation: true

bits:
  y_bits: 16  # Task/domain classification
  z_bits: 8   # Difficulty/precision
  x_bits: 8   # Experimental toggles

quantum:
  artifacts_path: "./quantum_artifacts"
  simulation_mode: true

api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
```

---

## 🚀 Deployment

### Vercel (Frontend)

```bash
cd frontend
vercel --prod
```

Set environment variable:
- `NEXT_PUBLIC_API_URL` = Your backend URL

### Netlify (Frontend)

```bash
cd frontend
netlify deploy --prod
```

### shiper.app (Full Stack)

```bash
# Build and push Docker image
docker build -t your-username/jarvis-2v:latest .
docker push your-username/jarvis-2v:latest

# Deploy on shiper.app dashboard
```

### Self-Hosted (Docker)

```bash
# Single container
docker run -d -p 8000:8000 -p 3000:3000 jarvis-2v:latest

# Or with docker-compose
docker-compose up -d
```

📖 **Full Deployment Guide**: See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🧪 Development

### Backend Development

```bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend linting
cd frontend
npm run lint
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| API Response Time | ~50-100ms |
| Frontend Load Time | <2s |
| Memory Usage (Backend) | ~200MB |
| Memory Usage (Frontend) | ~100MB |
| Adapters Per Request | 1-3 |
| Concurrent Users | 100+ |

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **NetworkX** - Graph algorithms
- **NumPy** - Numerical computing
- **uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide Icons** - Icon library

### Deployment
- **Docker** - Containerization
- **Vercel** - Frontend hosting
- **Netlify** - Frontend hosting
- **shiper.app** - Full-stack hosting

---

## 📖 Documentation

- 📘 [Deployment Guide](docs/DEPLOYMENT.md) - Complete deployment instructions
- 📗 [Backend README](backend/README.md) - Backend API documentation
- 📙 [Frontend README](frontend/README.md) - Frontend development guide
- 📕 [API Reference](http://localhost:8000/docs) - Interactive API docs

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Check ports
lsof -i :8000

# View logs
./scripts/start_backend.sh
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache
rm -rf frontend/.next frontend/node_modules
cd frontend && npm install
```

### Connection issues
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in frontend
- Verify CORS settings in backend

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- FastAPI team for the excellent framework
- Next.js team for the React framework
- llama.cpp team for GGUF model support
- NVIDIA Jetson team for edge AI tools
- The open-source community

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Cyberisthename/chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Cyberisthename/chatbot/discussions)
- **Email**: support@jarvis-2v.com (if available)

---

**JARVIS-2v** - *Bringing modular AI to the edge, one adapter at a time.* 🚀

---

## Quick Links

- [🏠 Dashboard](http://localhost:3000)
- [🔧 API Docs](http://localhost:8000/docs)
- [📚 Deployment Guide](docs/DEPLOYMENT.md)
- [🐛 Report Bug](https://github.com/Cyberisthename/chatbot/issues/new)
- [💡 Request Feature](https://github.com/Cyberisthename/chatbot/issues/new)
