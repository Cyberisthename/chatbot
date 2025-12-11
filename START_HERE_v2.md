# 🚀 Start Here: JARVIS-2v v2.0

Welcome to JARVIS-2v! This is your starting point for understanding and using the system.

---

## 🎯 What is JARVIS-2v?

JARVIS-2v is a **modular AI system** with:
- 🧠 **Adapter Engine**: Y/Z/X bit routing for dynamic AI modules
- ⚛️ **Quantum Lab**: Synthetic quantum experiments without hardware
- 🌐 **Web UI**: Modern Next.js interface
- 🔌 **REST API**: FastAPI backend with auto-docs
- 🐳 **Docker Ready**: One-command deployment
- ☁️ **Cloud Deployable**: Vercel, Netlify, Railway, etc.

---

## ⚡ Quick Start (3 Steps)

### 1. Clone & Setup
```bash
git clone https://github.com/Cyberisthename/chatbot.git
cd chatbot
```

### 2. Start Everything
```bash
./scripts/start_all_local.sh
```

### 3. Open Browser
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**That's it!** You're running JARVIS-2v. 🎉

---

## 📚 Documentation Guide

### I'm New Here
1. Read this file (you're doing it!)
2. Run the Quick Start above
3. Read `QUICKSTART.md` for more details
4. Explore the Dashboard at http://localhost:3000

### I'm a Developer
1. `QUICK_REFERENCE.md` - One-page cheat sheet
2. `backend/README.md` - Backend API docs
3. `frontend/README.md` - Frontend structure
4. http://localhost:8000/docs - Interactive API docs

### I Want to Deploy
1. `DEPLOYMENT_PLATFORMS.md` - Choose your platform
2. `TESTING_GUIDE.md` - Test before deploying
3. `docs/DEPLOYMENT.md` - Detailed instructions

### I Want to Understand the Code
1. `CHANGES_SUMMARY.md` - What's new in v2.0
2. `IMPLEMENTATION_COMPLETE.md` - Full feature list
3. `src/core/adapter_engine.py` - Core adapter system
4. `backend/main.py` - API implementation

---

## 🏗️ Project Structure (Simplified)

```
jarvis-2v/
├── 📱 frontend/          # Next.js web UI
│   ├── app/             # Pages (Dashboard, Adapters, etc.)
│   └── lib/             # API client
├── 🔧 backend/           # FastAPI server
│   └── main.py          # REST API
├── 🧠 src/               # Core engine
│   ├── core/            # Adapter system
│   └── quantum/         # Quantum experiments
├── 🐳 Docker files       # Containerization
├── 📜 scripts/           # Startup scripts
└── 📖 Docs              # Documentation (you are here)
```

---

## 🎮 What Can I Do?

### 1. Chat with JARVIS
- Go to http://localhost:3000/console
- Type a message
- See response with adapters used

### 2. Manage Adapters
- Go to http://localhost:3000/adapters
- View adapter graph
- Create new adapters
- See performance metrics

### 3. Run Quantum Experiments
- Go to http://localhost:3000/quantum
- Select experiment type
- Adjust parameters
- View results

### 4. Monitor System
- Go to http://localhost:3000
- See live system metrics
- Check adapter status
- View recent artifacts

### 5. Configure Settings
- Go to http://localhost:3000/settings
- Change deployment mode
- Update configuration
- Save changes

---

## 🛠️ Common Commands

```bash
# Start everything
./scripts/start_all_local.sh

# Start backend only
./scripts/start_backend.sh

# Start frontend only
./scripts/start_frontend.sh

# Docker
docker-compose up -d

# Test API
curl http://localhost:8000/health

# Run inference
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello JARVIS"}'
```

---

## 🎨 UI Preview

### Dashboard
- System status (ONLINE/OFFLINE)
- Active mode (low_power/standard/jetson_orin)
- Adapter count
- Artifact count
- Recent activity

### Adapters Page
- List all adapters
- Filter by status
- Create new adapters
- View bit patterns
- See success rates

### Quantum Lab
- Select experiment type:
  - Interference experiment
  - Bell pair simulation
  - CHSH test
  - Noise field scan
- Adjust iterations and noise level
- Run experiments
- View results and artifacts

### Console
- Chat-like interface
- Send queries
- See responses
- View adapters used
- Check processing time

### Settings
- Select deployment mode
- Toggle quantum features
- Update configuration
- View current settings

---

## 🔌 API Endpoints

### System
- `GET /health` - Health check
- `GET /api/config` - Configuration
- `POST /api/config` - Update config

### Inference
- `POST /api/infer` - Run inference

### Adapters
- `GET /api/adapters` - List adapters
- `POST /api/adapters` - Create adapter
- `GET /api/adapters/{id}` - Get adapter

### Quantum
- `POST /api/quantum/experiment` - Run experiment
- `GET /api/artifacts` - List artifacts
- `GET /api/artifacts/{id}` - Get artifact

**Full API docs**: http://localhost:8000/docs

---

## 🐳 Docker Usage

### Start with Docker Compose
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

### Single Container
```bash
docker build -t jarvis-2v .
docker run -p 8000:8000 -p 3000:3000 jarvis-2v
```

---

## ☁️ Cloud Deployment

### Vercel (Frontend)
1. Push to GitHub
2. Import in Vercel
3. Set `NEXT_PUBLIC_API_URL`
4. Deploy

### Railway (Backend)
1. Connect GitHub repo
2. Select `backend` directory
3. Deploy
4. Get backend URL

### Full Guide
See `DEPLOYMENT_PLATFORMS.md` for detailed instructions.

---

## 🧪 Testing

### Quick Health Check
```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# API
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Full Test Suite
```bash
# See TESTING_GUIDE.md for comprehensive tests
```

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Install dependencies
pip install --break-system-packages -r backend/requirements.txt

# Check port
lsof -i :8000

# View logs
tail -f /tmp/backend.log
```

### Frontend won't build
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run build
```

### Can't connect
```bash
# Check backend is running
curl http://localhost:8000/health

# Check API URL
echo $NEXT_PUBLIC_API_URL
```

More help: `TESTING_GUIDE.md` → Troubleshooting section

---

## 📖 Learning Path

### Beginner (15 minutes)
1. ✅ Run Quick Start
2. ✅ Open Dashboard
3. ✅ Read `QUICKSTART.md`
4. ✅ Try Console page

### Intermediate (1 hour)
1. ✅ Run quantum experiment
2. ✅ Create an adapter
3. ✅ Explore API docs
4. ✅ Read `QUICK_REFERENCE.md`

### Advanced (Half day)
1. ✅ Deploy to cloud
2. ✅ Read core engine code
3. ✅ Customize adapters
4. ✅ Add authentication

---

## 🎯 Next Steps

After getting started:

### Immediate
- [ ] Explore all pages in the UI
- [ ] Try creating an adapter
- [ ] Run a quantum experiment
- [ ] Test the API endpoints

### Short Term
- [ ] Read the full documentation
- [ ] Deploy to a cloud platform
- [ ] Customize the configuration
- [ ] Integrate with your LLM

### Long Term
- [ ] Add custom adapters
- [ ] Create new experiments
- [ ] Build plugins
- [ ] Contribute back

---

## 💡 Pro Tips

1. **API Docs are your friend**: http://localhost:8000/docs
2. **Check logs first**: When something breaks, check logs
3. **Use scripts**: Don't manually start services
4. **Docker is reliable**: When local setup fails, use Docker
5. **Test locally first**: Always test before deploying

---

## 🆘 Getting Help

### Self-Help
1. Check `QUICK_REFERENCE.md` for commands
2. Read `TESTING_GUIDE.md` for troubleshooting
3. View logs: `tail -f /tmp/backend.log`
4. Test health: `curl http://localhost:8000/health`

### Ask for Help
1. Check existing GitHub issues
2. Open new issue with:
   - What you tried
   - Error messages
   - Logs
   - System info (OS, Python/Node versions)

---

## ✨ Key Features

### Modular Adapters
- Dynamic routing based on Y/Z/X bits
- Graph-based relationships
- Auto-creation for new tasks
- Performance tracking

### Quantum Lab
- Interference experiments
- Bell pair simulations
- CHSH inequality tests
- Noise field scanning
- Artifact generation

### Production Ready
- RESTful API
- Modern UI
- Docker support
- Cloud deployable
- Comprehensive docs

---

## 🎊 You're Ready!

You now know:
- ✅ What JARVIS-2v is
- ✅ How to start it
- ✅ Where to find docs
- ✅ What you can do
- ✅ How to get help

**Go build something amazing!** 🚀

---

## 📞 Quick Links

- **Main Docs**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **Cheat Sheet**: `QUICK_REFERENCE.md`
- **Testing**: `TESTING_GUIDE.md`
- **Deployment**: `DEPLOYMENT_PLATFORMS.md`
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000

---

**Version**: 2.0.0  
**Status**: 🟢 Production Ready  
**License**: MIT  
**Repository**: https://github.com/Cyberisthename/chatbot

Happy coding! 💻✨
