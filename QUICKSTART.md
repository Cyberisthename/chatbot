# JARVIS-2v Quick Start Guide

Get up and running with JARVIS-2v in under 5 minutes! ⚡

---

## 🚀 Option 1: Automatic Start (Recommended)

**One command to start everything:**

```bash
./scripts/start_all_local.sh
```

This will:
1. ✅ Start the FastAPI backend on port 8000
2. ✅ Start the Next.js frontend on port 3000
3. ✅ Open your browser automatically

**Access the app:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔧 Option 2: Manual Start

### Step 1: Start Backend

```bash
./scripts/start_backend.sh
```

Wait for: `✅ Backend is ready!`

### Step 2: Start Frontend (in another terminal)

```bash
./scripts/start_frontend.sh
```

Wait for: `✅ Frontend started!`

### Step 3: Open Browser

Visit: http://localhost:3000

---

## 🐳 Option 3: Docker

```bash
docker-compose up
```

Wait a few seconds, then visit: http://localhost:3000

---

## 📱 What You'll See

### Dashboard (/)
- System status indicator (should be green/online)
- Number of adapters and quantum artifacts
- Recent activity

### Adapters (/adapters)
- List of all AI adapters
- Click "Create Adapter" to add a new one
- View Y/Z/X bit patterns and success rates

### Quantum Lab (/quantum)
- Run synthetic quantum experiments
- Choose from 4 experiment types
- Adjust iterations and noise level
- View generated artifacts

### Console (/console)
- Chat with JARVIS
- Type a message and press Enter
- See which adapters were used

### Settings (/settings)
- Change deployment mode
- Toggle features
- View system information

---

## ✅ First Steps

### 1. Create Your First Adapter

1. Go to **Adapters** page
2. Click **Create Adapter**
3. Enter tags: `testing, demo`
4. Click **Create**
5. See your new adapter appear!

### 2. Run Your First Experiment

1. Go to **Quantum Lab** page
2. Select **Interference Experiment**
3. Click **Run Experiment**
4. Wait ~2-3 seconds
5. See your artifact in the list below!

### 3. Chat with JARVIS

1. Go to **Console** page
2. Type: `Hello JARVIS, what can you do?`
3. Press Enter
4. See the response!

---

## 🔍 Quick Test Commands

### Test Backend Health

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

Expected output:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "mode": "standard",
  "adapters_count": 0,
  "artifacts_count": 0,
  "timestamp": 1234567890.123
}
```

### List Adapters

```bash
curl http://localhost:8000/api/adapters | python3 -m json.tool
```

### Run Inference

```bash
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello JARVIS"}' \
  | python3 -m json.tool
```

---

## 🛑 Stopping the Services

### If using start_all_local.sh

Press `Ctrl+C` in the terminal

### If started separately

Press `Ctrl+C` in each terminal window

### If using Docker

```bash
docker-compose down
```

---

## 🐛 Troubleshooting

### Backend won't start

**Error**: `Port 8000 already in use`

**Solution**:
```bash
lsof -ti:8000 | xargs kill -9
./scripts/start_backend.sh
```

### Frontend won't start

**Error**: `Port 3000 already in use`

**Solution**:
```bash
lsof -ti:3000 | xargs kill -9
./scripts/start_frontend.sh
```

### Connection refused

**Problem**: Frontend can't reach backend

**Solution**:
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check frontend .env: `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. Restart both services

### Dependencies missing

**Backend**:
```bash
pip install -r backend/requirements.txt
```

**Frontend**:
```bash
cd frontend
npm install
```

---

## 📚 Next Steps

Now that you're up and running:

1. 📖 Read the full [README.md](README.md)
2. 🚀 Check [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for cloud deployment
3. 🧪 Explore the API at http://localhost:8000/docs
4. 🎨 Customize the frontend theme in `frontend/tailwind.config.ts`
5. ⚙️ Modify settings in `config.yaml`

---

## 💡 Pro Tips

### Development Mode

Both backend and frontend support hot reload:
- Backend: Edit `backend/main.py` and it auto-reloads
- Frontend: Edit any file in `frontend/` and page refreshes

### View Logs

```bash
# Backend logs
tail -f backend_test.log

# Frontend logs (already in terminal)
```

### Quick Restart

```bash
# Kill all and restart
pkill -f uvicorn; pkill -f next
./scripts/start_all_local.sh
```

### API Exploration

Visit http://localhost:8000/docs for:
- Interactive API documentation
- Try out endpoints directly
- See request/response schemas

---

## 🎉 You're All Set!

JARVIS-2v is now running on your machine. Explore the UI, create adapters, run quantum experiments, and chat with JARVIS!

**Need help?** Check the [README.md](README.md) or [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

**Happy coding! 🚀**
