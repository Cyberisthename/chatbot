# JARVIS-2v Deployment Guide

Complete guide for deploying JARVIS-2v on various platforms.

## 🎯 Deployment Overview

JARVIS-2v can be deployed in multiple configurations:

1. **Local Development** - Run on your machine for testing
2. **Docker** - Containerized deployment with docker-compose
3. **Cloud Platforms** - Vercel, Netlify, Render, Railway
4. **Edge Devices** - Jetson Orin NX, Raspberry Pi 4/5
5. **Kubernetes** - Production-scale deployment

## 📋 Prerequisites

### Core Requirements
- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- 4GB RAM minimum (8GB+ recommended)
- 10GB disk space (models + data)

### Optional Requirements
- Docker & Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA (for GPU acceleration)
- Ollama (for alternative model serving)

## 🚀 Deployment Methods

### 1. Local Development

#### Quick Start

```bash
# Clone repository
git clone https://github.com/Cyberisthename/chatbot.git
cd chatbot

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Download or place your GGUF model in ./models/
# Update config.yaml with correct model path

# Start Python backend
python inference.py models/jarvis-7b-q4_0.gguf --port 8000 &

# Start Node.js frontend
npm start
```

Access at: http://localhost:3001

#### Training Adapters Locally

```bash
# Train from local files
python scripts/train_adapters.py --input data/raw

# Stream from IDI dataset (requires internet + datasets library)
pip install datasets
python scripts/train_idi_stream.py --max-books 100 --language en

# Ingest quick knowledge
python scripts/ingest_knowledge.py --fact "JARVIS-2v uses Y/Z/X routing" --tags ai knowledge
```

### 2. Docker Deployment

#### Using Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Training with Docker

```bash
# Start training container
docker-compose --profile training up -d jarvis-trainer

# Run training inside container
docker-compose exec jarvis-trainer python scripts/train_adapters.py --input data/raw

# Stream IDI training
docker-compose exec jarvis-trainer python scripts/train_idi_stream.py --max-books 50

# Stop training container
docker-compose --profile training down
```

#### Custom Docker Build

```bash
# Build custom image
docker build -t jarvis2v:latest .

# Run backend only
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/adapters:/app/adapters \
  -v $(pwd)/data:/app/data \
  --name jarvis-backend \
  jarvis2v:latest
```

### 3. Vercel Deployment

For web UI only (no model files):

```bash
# Use the clean deployment branch
git checkout deploy/vercel-clean-webapp-no-lfs

# Push to Vercel
vercel --prod

# Or link existing repo
vercel link
vercel --prod
```

**Note**: Vercel deployment runs in MOCK_MODE since models are too large. For full functionality, use backend deployment separately.

See: [QUICKSTART_VERCEL.md](../QUICKSTART_VERCEL.md)

### 4. Netlify Deployment

Similar to Vercel:

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod
```

Configure in `netlify.toml`:

```toml
[build]
  command = "npm install"
  publish = "."

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/:splat"
  status = 200
```

### 5. Render.com Deployment

#### Web Service

1. Connect GitHub repository
2. Select "Web Service"
3. Build Command: `pip install -r requirements.txt && npm install`
4. Start Command: `npm start`
5. Environment Variables:
   - `MOCK_MODE=1` (if no model files)
   - `NODE_ENV=production`

#### Background Worker (for backend with models)

1. Create "Background Worker"
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `python inference.py models/jarvis-7b-q4_0.gguf --port 8000`
4. Add disk storage for models

### 6. Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

Configure in `railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python inference.py models/jarvis-7b-q4_0.gguf --port 8000",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### 7. Edge Device Deployment (Jetson Orin NX)

#### Prerequisites

```bash
# On Jetson device
sudo apt-get update
sudo apt-get install -y python3-pip git cmake build-essential

# Install CUDA-enabled dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

#### Jetson-Specific Configuration

Use `config_jetson.yaml`:

```yaml
engine:
  mode: "jetson_orin"

model:
  device: "cuda"
  gpu_layers: 35  # Adjust based on model size
  quantization: "Q4_0"

edge:
  jetson_cuda_arch: "72"  # Orin NX
  low_power_mode: false
```

#### Run on Jetson

```bash
# Use Jetson startup script
./scripts/start_jetson.sh

# Or manually
python inference.py models/jarvis-7b-q4_0.gguf --port 8000
```

### 8. Kubernetes Deployment

#### Create Kubernetes Manifests

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis2v
spec:
  replicas: 2
  selector:
    matchLabels:
      app: jarvis2v
  template:
    metadata:
      labels:
        app: jarvis2v
    spec:
      containers:
      - name: jarvis-backend
        image: jarvis2v:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: adapters
          mountPath: /app/adapters
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: jarvis-models-pvc
      - name: adapters
        persistentVolumeClaim:
          claimName: jarvis-adapters-pvc
```

Deploy:

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl logs -f deployment/jarvis2v
```

### 9. Ollama Integration

Use Ollama for model serving:

```bash
# Create Ollama model
cd /path/to/jarvis-2v
ollama create jarvis2v -f ollama/Modelfile

# Run with Ollama
ollama run jarvis2v

# Use Ollama API endpoint
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis2v",
  "prompt": "Hello JARVIS!"
}'
```

See: [ollama/README.md](../ollama/README.md)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOCK_MODE` | Run without model files | `0` |
| `NODE_ENV` | Node environment | `development` |
| `PYTHONUNBUFFERED` | Python output buffering | `1` |
| `MODEL_PATH` | Path to GGUF model | `./models/jarvis-7b-q4_0.gguf` |
| `PORT` | Backend port | `8000` |
| `FRONTEND_PORT` | Frontend port | `3001` |

### Config Files

- `config.yaml` - Main configuration (CPU/GPU settings)
- `config_jetson.yaml` - Jetson-specific config
- `ollama/Modelfile` - Ollama model definition

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3001/health

# Docker health
docker-compose ps
```

### Logs

```bash
# View Python logs
tail -f logs/jarvis.log

# View Docker logs
docker-compose logs -f jarvis-backend

# View Kubernetes logs
kubectl logs -f deployment/jarvis2v
```

### Backup Data

Important files to backup:

```bash
# Adapters
tar -czf adapters-backup.tar.gz adapters/

# Memory
cp jarvis_memory.json jarvis_memory.json.backup

# Quantum artifacts
tar -czf quantum-backup.tar.gz quantum_artifacts/

# Training metadata
cp idi_training_metadata.json idi_training_metadata.json.backup
cp local_training_metadata.json local_training_metadata.json.backup
```

### Updates

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
npm install

# Restart services
docker-compose restart
# or
systemctl restart jarvis2v
```

## 🐛 Troubleshooting

### "Model not found" error

```bash
# Check model path
ls -lh models/*.gguf

# Update config.yaml
# Ensure path matches actual file location
```

### "Out of memory" error

Reduce context size in `config.yaml`:

```yaml
model:
  context_size: 1024  # Reduce from 2048
  gpu_layers: 0  # Use CPU only
```

### Adapter routing not working

```bash
# Check adapter files exist
ls -lh adapters/*.json

# Verify adapter graph
cat adapters_graph.json

# Retrain if needed
python scripts/train_adapters.py --input data/raw
```

### Docker build fails

```bash
# Clean build cache
docker-compose build --no-cache

# Check disk space
df -h

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory
```

### Slow inference

```bash
# Use smaller quantization
# Q4_0 < Q5_0 < Q8_0 (smaller = faster)

# Enable GPU acceleration
# In config.yaml:
model:
  gpu_layers: 35  # Offload to GPU
  device: "cuda"
```

## 🔒 Security Considerations

### Production Deployment

1. **API Authentication**: Add authentication middleware
2. **Rate Limiting**: Implement request rate limits
3. **HTTPS**: Use SSL/TLS certificates
4. **Environment Variables**: Never commit secrets to git
5. **Input Validation**: Sanitize user inputs
6. **CORS**: Configure allowed origins

Example nginx proxy:

```nginx
server {
    listen 443 ssl;
    server_name jarvis.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000/;
    }
}
```

## 📚 Additional Resources

- [JARVIS-2v Architecture](../README.md)
- [Training Guide](./TRAINING.md)
- [API Documentation](./API.md)
- [Ollama Integration](../ollama/README.md)
- [Vercel Deployment](../QUICKSTART_VERCEL.md)

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify configuration: `cat config.yaml`
3. Test health endpoints
4. Review GitHub issues
5. Join Discord community (if available)

## 📝 License

See [LICENSE](../LICENSE) for details.
