# JARVIS-2v Deployment Guide

Complete guide for deploying JARVIS-2v to various platforms.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Vercel Deployment](#vercel-deployment)
4. [Netlify Deployment](#netlify-deployment)
5. [shiper.app Deployment](#shiperapp-deployment)
6. [Production Considerations](#production-considerations)

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Cyberisthename/chatbot.git
   cd chatbot
   ```

2. **Start both backend and frontend**
   ```bash
   ./scripts/start_all_local.sh
   ```

   Or start them separately:
   
   **Backend:**
   ```bash
   ./scripts/start_backend.sh
   ```
   
   **Frontend (in another terminal):**
   ```bash
   ./scripts/start_frontend.sh
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

---

## Docker Deployment

### Single Container (Recommended for Production)

1. **Build the image**
   ```bash
   docker build -t jarvis-2v:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name jarvis-2v \
     -p 8000:8000 \
     -p 3000:3000 \
     -v $(pwd)/adapters:/app/adapters \
     -v $(pwd)/quantum_artifacts:/app/quantum_artifacts \
     jarvis-2v:latest
   ```

3. **Check logs**
   ```bash
   docker logs -f jarvis-2v
   ```

### Using Docker Compose (Development)

1. **Start services**
   ```bash
   docker-compose up -d
   ```

2. **View logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop services**
   ```bash
   docker-compose down
   ```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Vercel Deployment

Vercel is ideal for the **frontend only**. The backend needs to be deployed separately.

### Deploy Frontend to Vercel

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Configure environment**
   ```bash
   cd frontend
   vercel env add NEXT_PUBLIC_API_URL production
   # Enter your backend URL (e.g., https://jarvis-backend.fly.io)
   ```

3. **Deploy**
   ```bash
   vercel --prod
   ```

### Deploy Backend Separately

Options for backend:
- **Railway**: `railway up` (requires Railway CLI)
- **Fly.io**: `flyctl deploy` (requires Fly CLI)
- **Render**: Connect GitHub repo
- **DigitalOcean App Platform**: Connect GitHub repo

### Example: Deploy Backend to Fly.io

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**
   ```bash
   flyctl auth login
   ```

3. **Create `fly.toml` in project root**
   ```toml
   app = "jarvis-backend"
   
   [build]
     dockerfile = "Dockerfile.backend"
   
   [[services]]
     http_checks = []
     internal_port = 8000
     protocol = "tcp"
   
     [[services.ports]]
       handlers = ["http"]
       port = 80
   
     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443
   
   [env]
     HOST = "0.0.0.0"
     PORT = "8000"
   ```

4. **Deploy**
   ```bash
   flyctl deploy
   ```

5. **Get backend URL**
   ```bash
   flyctl info
   # Use the hostname in Vercel frontend env
   ```

---

## Netlify Deployment

Similar to Vercel - frontend on Netlify, backend elsewhere.

### Deploy Frontend to Netlify

1. **Connect GitHub repository**
   - Go to https://app.netlify.com
   - Click "Add new site" > "Import an existing project"
   - Connect your GitHub repo

2. **Configure build settings**
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/.next`

3. **Add environment variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.com
   ```

4. **Deploy**
   - Click "Deploy site"

### Using Netlify CLI

```bash
# Install CLI
npm i -g netlify-cli

# Login
netlify login

# Deploy
cd frontend
netlify deploy --prod
```

---

## shiper.app Deployment

shiper.app supports Docker containers.

### Prerequisites

- Docker Hub or GitHub Container Registry account
- shiper.app account

### Deploy to shiper.app

1. **Build and push Docker image**
   ```bash
   # Build
   docker build -t your-username/jarvis-2v:latest .
   
   # Login to Docker Hub
   docker login
   
   # Push
   docker push your-username/jarvis-2v:latest
   ```

2. **Deploy on shiper.app**
   - Go to https://shiper.app
   - Create new app
   - Select "Docker Container"
   - Enter image: `your-username/jarvis-2v:latest`
   - Configure ports: 8000 (backend), 3000 (frontend)
   - Add environment variables:
     ```
     HOST=0.0.0.0
     PORT=8000
     NEXT_PUBLIC_API_URL=http://localhost:8000
     ```
   - Deploy!

3. **Access your app**
   - shiper.app will provide URLs for both services

### Using Docker Compose on shiper.app

Upload `docker-compose.yml` and configure:

```yaml
version: '3.8'
services:
  jarvis:
    image: your-username/jarvis-2v:latest
    ports:
      - "8000:8000"
      - "3000:3000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
```

---

## Production Considerations

### Environment Variables

**Backend:**
```bash
HOST=0.0.0.0
PORT=8000
JARVIS_CONFIG=/app/config.yaml
```

**Frontend:**
```bash
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

### Security

1. **CORS Configuration**
   - Update `backend/main.py` to restrict origins:
   ```python
   allow_origins=["https://your-frontend.com"]
   ```

2. **API Keys** (if added)
   - Use environment variables
   - Never commit to Git

3. **HTTPS**
   - Always use HTTPS in production
   - Most platforms provide free SSL

### Scaling

1. **Backend**
   - Use multiple worker processes: `--workers 4`
   - Enable caching for adapter lookups
   - Use Redis for session storage

2. **Frontend**
   - Enable Next.js caching
   - Use CDN for static assets
   - Enable ISR (Incremental Static Regeneration)

### Monitoring

1. **Health Checks**
   - Backend: `GET /health`
   - Set up uptime monitoring (UptimeRobot, Pingdom)

2. **Logging**
   - Configure log aggregation (Papertrail, Logtail)
   - Set up error tracking (Sentry)

3. **Metrics**
   - Use FastAPI middleware for request metrics
   - Monitor adapter success rates

### Database (Optional)

For production, consider:
- PostgreSQL for adapter/artifact persistence
- Redis for caching
- S3-compatible storage for artifacts

### Performance

1. **Backend**
   - Use `gunicorn` with `uvicorn` workers:
     ```bash
     gunicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
     ```

2. **Frontend**
   - Enable Next.js production optimizations
   - Use image optimization
   - Enable code splitting

---

## Troubleshooting

### Backend won't start

```bash
# Check Python version
python3 --version  # Should be 3.10+

# Check dependencies
pip install -r backend/requirements.txt

# Check ports
lsof -i :8000
```

### Frontend build fails

```bash
# Clear cache
rm -rf frontend/.next frontend/node_modules

# Reinstall
cd frontend
npm install
npm run build
```

### CORS errors

- Ensure backend CORS is configured
- Check API_URL environment variable
- Verify backend is accessible from frontend domain

### Connection refused

- Ensure backend is running
- Check firewall rules
- Verify ports are open

---

## Support

- GitHub Issues: https://github.com/Cyberisthename/chatbot/issues
- Docs: See project README.md

---

## Quick Reference

### Start Commands

```bash
# Local development
./scripts/start_all_local.sh

# Backend only
./scripts/start_backend.sh

# Frontend only
./scripts/start_frontend.sh

# Docker
docker-compose up

# Production Docker
docker run -p 8000:8000 -p 3000:3000 jarvis-2v:latest
```

### URLs

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Next Steps

After deployment:

1. Configure your domain
2. Set up SSL certificates
3. Configure monitoring
4. Set up backups for adapters/artifacts
5. Review security settings
6. Test all functionality
7. Monitor logs and metrics

Happy deploying! 🚀
