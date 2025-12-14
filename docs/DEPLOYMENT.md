# JARVIS-2v Deployment Guide

## Overview

This guide covers deploying JARVIS-2v for local development, cloud environments, and production scenarios. JARVIS-2v supports multiple deployment modes:

- **Local Development**: Full functionality with local models
- **Web Demo**: Public deployment without model files
- **Production**: Scalable cloud deployment with optimizations

## Quick Start Deployment

### 1. Local Development Setup

```bash
# 1. Clone/extract JARVIS-2v
git clone <repository> jarvis-2v
cd jarvis-2v

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install optional dependencies for RAG
pip install faiss-cpu numpy PyPDF2 networkx PyYAML

# 4. Create environment
cp .env.example .env
# Edit .env with your settings

# 5. Train your JARVIS
python scripts/train_adapters.py --input ./training-data --profile standard

# 6. Start API server
python -m src.api.main

# 7. Test installation
curl http://localhost:3001/health
```

### 2. Create Ollama Model (Optional)

```bash
# Install Ollama (see docs/OLLAMA.md for details)
# Create JARVIS model
ollama create jarvis2v -f ollama/Modelfile

# Test model
ollama run jarvis2v
```

### 3. Web Demo Deployment

For public access without model files:

```bash
# Use the deploy branch pattern
git checkout deploy/vercel-clean-webapp-no-lfs

# Configure for demo mode
echo "DEMO_MODE=true" >> .env

# Deploy to Vercel/Netlify
vercel --prod  # or use Netlify CLI
```

## Deployment Options

### Option 1: Docker Deployment

#### Basic Docker Setup

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies
RUN pip install faiss-cpu numpy PyPDF2 networkx PyYAML

# Copy application code
COPY . /app
WORKDIR /app

# Create directories
RUN mkdir -p data models adapters

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Start application
CMD ["python", "-m", "src.api.main"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  jarvis-2v:
    build: .
    ports:
      - "3001:3001"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=3001
      - ENGINE_MODE=standard
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./adapters:/app/adapters
      - ./jarvis_memory.json:/app/jarvis_memory.json
    restart: unless-stopped
    
  # Optional: Ollama service
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

**Deploy:**
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f jarvis-2v

# Scale if needed
docker-compose up --scale jarvis-2v=3
```

#### Docker with GPU Support

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip3.11 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
COPY requirements.txt .
RUN pip3.11 install --no-cache-dir -r requirements.txt
RUN pip3.11 install faiss-cpu numpy PyPDF2 networkx PyYAML

# Copy application
COPY . /app
WORKDIR /app

EXPOSE 3001
CMD ["python3.11", "-m", "src.api.main"]
```

**docker-compose.gpu.yml:**
```yaml
version: '3.8'

services:
  jarvis-gpu:
    build: 
      context: .
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "3001:3001"
    environment:
      - GPU_LAYERS=32
      - DEVICE=cuda
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
```

### Option 2: Vercel Deployment (Web Demo)

**vercel.json:**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "server.js",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/server.js"
    }
  ],
  "env": {
    "DEMO_MODE": "true",
    "API_HOST": "0.0.0.0",
    "API_PORT": "3001"
  },
  "functions": {
    "server.js": {
      "maxDuration": 30
    }
  }
}
```

**deploy command:**
```bash
# Set environment for demo mode
vercel env add DEMO_MODE
# Set to: true

# Deploy
vercel --prod
```

**Demo Mode Configuration:**
```javascript
// server.js - demo mode
const DEMO_MODE = process.env.DEMO_MODE === 'true';

if (DEMO_MODE) {
  // Mock responses instead of using actual models
  app.post('/chat', (req, res) => {
    res.json({
      message: {
        content: "Demo mode: JARVIS-2v would respond here with knowledge base integration."
      },
      usage: { tokens: 0 },
      adapters_used: ["demo_adapter"],
      kb_context_used: false
    });
  });
}
```

### Option 3: Netlify Deployment

**netlify.toml:**
```toml
[build]
  command = "echo 'Build completed'"
  publish = "."

[functions]
  directory = "netlify/functions"

[[redirects]]
  from = "/*"
  to = "/.netlify/functions/server"
  status = 200

[context.production.environment]
  DEMO_MODE = "true"

[build.environment]
  NODE_VERSION = "18"
```

**netlify/functions/server.js:**
```javascript
const { createClient } = require('@netlify/functions');

exports.handler = createClient(async (event) => {
  // Demo mode responses
  const response = {
    message: {
      content: "JARVIS-2v Demo Mode - Training from custom documents would happen here."
    },
    usage: { tokens: 0 },
    adapters_used: ["demo_adapter"],
    kb_context_used: false
  };

  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(response)
  };
});
```

### Option 4: Cloud Platform Deployment

#### AWS Deployment

**AWS ECS with Fargate:**
```json
{
  "family": "jarvis-2v",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "jarvis-2v",
      "image": "your-account.dkr.ecr.region.amazonaws.com/jarvis-2v:latest",
      "portMappings": [
        {
          "containerPort": 3001,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "API_PORT", 
          "value": "3001"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/jarvis-2v",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**AWS Lambda (Serverless):**
```python
# lambda_function.py
import json
import os
from src.api.main import create_app

# Initialize app (cold start)
app = create_app()

def lambda_handler(event, context):
    # Simple health check for demo
    if event.get('path') == '/health':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'healthy',
                'mode': 'lambda_demo',
                'llm_ready': False
            })
        }
    
    # Mock chat response
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            'message': {'content': 'JARVIS-2v Lambda Demo Mode'},
            'usage': {'tokens': 0},
            'adapters_used': ['demo'],
            'kb_context_used': False
        })
    }
```

#### Google Cloud Run

**Cloud Run deployment:**
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT-ID/jarvis-2v

# Deploy to Cloud Run
gcloud run deploy jarvis-2v \
    --image gcr.io/PROJECT-ID/jarvis-2v \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 3001 \
    --memory 2Gi \
    --cpu 2
```

### Option 5: Kubernetes Deployment

**k8s-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis-2v
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jarvis-2v
  template:
    metadata:
      labels:
        app: jarvis-2v
    spec:
      containers:
      - name: jarvis-2v
        image: jarvis-2v:latest
        ports:
        - containerPort: 3001
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "3001"
        - name: ENGINE_MODE
          value: "standard"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: jarvis-2v-service
spec:
  selector:
    app: jarvis-2v
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3001
  type: LoadBalancer
```

## Production Optimization

### Performance Tuning

**For High-Traffic:**
```yaml
# Horizontal scaling
replicas: 10

# Resource allocation
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"

# Load balancer settings
type: LoadBalancer
```

**For Low-Power/Edge:**
```yaml
# Edge deployment
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

# Low power mode
env:
- name: ENGINE_MODE
  value: "low_power"
- name: GPU_LAYERS
  value: "0"
```

### Monitoring and Logging

**Health Checks:**
```python
# Enhanced health endpoint
@app.get("/health")
async def enhanced_health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "llm_ready": llm_engine.is_initialized,
        "kb_stats": knowledge_base.get_stats() if knowledge_base else None,
        "adapter_count": len(adapter_engine.list_adapters()),
        "system_info": {
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
            "disk_usage": psutil.disk_usage('/').percent
        }
    }
```

**Logging Configuration:**
```python
# Structured logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)
```

**Metrics Collection:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('jarvis_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('jarvis_request_duration_seconds', 'Request latency')
KB_CHUNKS = Gauge('jarvis_kb_chunks_total', 'Knowledge base chunks')
ACTIVE_ADAPTERS = Gauge('jarvis_active_adapters', 'Active adapters')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    
    return response
```

### Security

**API Authentication:**
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer(auto_error=False)

def verify_api_key(api_key: str = Depends(security)):
    if api_key and api_key.credentials == os.getenv("API_KEY"):
        return True
    raise HTTPException(status_code=401, detail="Invalid API key")

# Apply to endpoints
@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def secure_chat(request: ChatRequest):
    # ... existing logic
```

**Rate Limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def rate_limited_chat(request: Request, request_data: ChatRequest):
    # ... existing logic
```

## Troubleshooting

### Common Deployment Issues

**Memory Issues:**
```bash
# Monitor memory usage
docker stats
# or
kubectl top pod jarvis-2v

# Increase memory limits
resources:
  limits:
    memory: "4Gi"
```

**Model Loading Failures:**
```python
# Graceful degradation
try:
    self.llm_engine = JarvisInferenceBackend(model_path, config)
    self.llm_engine.initialize()
except Exception as e:
    logger.warning(f"LLM initialization failed: {e}")
    self.llm_engine = None
```

**Database/Storage Issues:**
```python
# Persistent storage
volumes:
- name: data-volume
  persistentVolumeClaim:
    claimName: jarvis-data-pvc
```

### Health Monitoring

**Comprehensive Health Check:**
```bash
#!/bin/bash
# health-check.sh

API_URL=${API_URL:-"http://localhost:3001"}

echo "Checking JARVIS-2v health..."

# Basic health
curl -f "${API_URL}/health" || exit 1

# KB functionality
curl -X POST "${API_URL}/kb/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "/tmp/test.txt"}' || exit 1

# Adapter functionality
curl -f "${API_URL}/adapters" || exit 1

echo "All health checks passed!"
```

## CI/CD Pipeline

**GitHub Actions (.github/workflows/deploy.yml):**
```yaml
name: Deploy JARVIS-2v

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install faiss-cpu numpy PyPDF2 networkx PyYAML
    
    - name: Run tests
      run: python test_kb.py
    
    - name: Train adapters
      run: python scripts/train_adapters.py --input ./training-data --profile standard
    
    - name: Build Docker image
      run: docker build -t jarvis-2v:${{ github.sha }} .
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        docker tag jarvis-2v:${{ github.sha }} staging-registry/jarvis-2v:latest
        # ... deployment commands
```

## Backup and Recovery

**Data Backup:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/jarvis-2v/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical data
cp -r data/ "$BACKUP_DIR/"
cp -r adapters/ "$BACKUP_DIR/"
cp jarvis_memory.json "$BACKUP_DIR/"
cp config.yaml "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
```

**Recovery Script:**
```bash
#!/bin/bash
# restore.sh

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

# Stop services
docker-compose down

# Restore data
cp -r "$BACKUP_DIR/data/" ./
cp -r "$BACKUP_DIR/adapters/" ./
cp "$BACKUP_DIR/jarvis_memory.json" ./
cp "$BACKUP_DIR/config.yaml" ./

# Restart services
docker-compose up -d

echo "Recovery completed from: $BACKUP_DIR"
```

## Next Steps

1. **Choose deployment method** based on your requirements
2. **Set up monitoring** and health checks
3. **Configure backup** strategies
4. **Set up CI/CD** for automated deployment
5. **Test thoroughly** in staging before production
6. **Monitor performance** and scale as needed

For specific platform deployment (AWS, GCP, Azure), consult the platform-specific guides or use the provided Docker/Kubernetes configurations as starting points.