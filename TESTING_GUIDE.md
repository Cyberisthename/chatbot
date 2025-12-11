# JARVIS-2v Testing & Validation Guide

This guide helps you verify that all components of JARVIS-2v are working correctly.

## Quick Health Check

### 1. Backend API Test

Start the backend and verify all endpoints:

```bash
# Start backend
./scripts/start_backend.sh

# Wait for startup (3-5 seconds)
sleep 3

# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "ok",
#   "version": "2.0.0",
#   "mode": "standard",
#   "adapters_count": 0,
#   "artifacts_count": 0,
#   "timestamp": 1234567890.123
# }
```

### 2. Frontend Build Test

Verify the frontend builds successfully:

```bash
cd frontend
npm install
npm run build

# Should complete without errors
# Expected output: "✓ Compiled successfully"
```

### 3. Full Stack Integration Test

Test both backend and frontend together:

```bash
# Start both services
./scripts/start_all_local.sh

# In another terminal, test the API
curl http://localhost:8000/health

# Open browser to http://localhost:3000
# You should see the Dashboard
```

---

## Detailed API Endpoint Tests

### Health Check

```bash
curl http://localhost:8000/health | jq
```

### Run Inference

```bash
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hello JARVIS, tell me about quantum experiments"
  }' | jq
```

Expected: Response with `adapters_used`, `bit_patterns`, and `response` text.

### Create an Adapter

```bash
curl -X POST http://localhost:8000/api/adapters \
  -H "Content-Type: application/json" \
  -d '{
    "task_tags": ["quantum", "analysis"],
    "parameters": {"model": "demo"}
  }' | jq
```

Expected: Response with `adapter_id` and adapter details.

### List Adapters

```bash
curl http://localhost:8000/api/adapters | jq
```

Expected: JSON with `adapters` array and `total` count.

### Run Quantum Experiment

```bash
curl -X POST http://localhost:8000/api/quantum/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_type": "interference_experiment",
    "iterations": 1000,
    "noise_level": 0.1
  }' | jq
```

Expected: Response with `artifact_id`, `experiment_type`, and `results_summary`.

### List Quantum Artifacts

```bash
curl http://localhost:8000/api/artifacts | jq
```

Expected: JSON with `artifacts` array and `total` count.

### Get Configuration

```bash
curl http://localhost:8000/api/config | jq
```

Expected: Full configuration object.

### Update Configuration

```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "low_power"
  }' | jq
```

Expected: Response with `status: "updated"` and changes applied.

---

## Frontend Component Tests

### 1. Dashboard Page

- Navigate to http://localhost:3000
- Check that:
  - ✅ System status shows "ONLINE"
  - ✅ Active mode displays correctly
  - ✅ Adapter and artifact counts are visible
  - ✅ Recent adapters and artifacts load (if any exist)

### 2. Adapters Page

- Navigate to http://localhost:3000/adapters
- Check that:
  - ✅ Adapter list loads
  - ✅ "Create Adapter" button works
  - ✅ Filter by status works
  - ✅ Adapter details show bit patterns and metrics

### 3. Quantum Lab Page

- Navigate to http://localhost:3000/quantum
- Check that:
  - ✅ Experiment form loads
  - ✅ Can select experiment type
  - ✅ Can adjust iterations and noise level
  - ✅ "Run Experiment" button works
  - ✅ Results display after experiment completes

### 4. Console Page

- Navigate to http://localhost:3000/console
- Check that:
  - ✅ Chat input field works
  - ✅ Can send messages
  - ✅ Responses appear in chat history
  - ✅ Shows adapters used and processing time

### 5. Settings Page

- Navigate to http://localhost:3000/settings
- Check that:
  - ✅ Current mode displays
  - ✅ Can change deployment mode
  - ✅ Configuration updates save

---

## Docker Deployment Tests

### Test Docker Compose

```bash
# Build and start services
docker-compose up -d

# Wait for services to be ready
sleep 10

# Test backend health
curl http://localhost:8000/health

# Test frontend (should return HTML)
curl http://localhost:3000

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Test Single Container Build

```bash
# Build the all-in-one container
docker build -t jarvis-2v:test .

# Run the container
docker run -d \
  --name jarvis-test \
  -p 8000:8000 \
  -p 3000:3000 \
  jarvis-2v:test

# Test endpoints
sleep 10
curl http://localhost:8000/health
curl http://localhost:3000

# View logs
docker logs jarvis-test

# Cleanup
docker stop jarvis-test
docker rm jarvis-test
```

---

## Performance & Load Tests

### Backend Performance

```bash
# Test inference latency (requires apache bench)
ab -n 100 -c 10 \
  -p post_data.json \
  -T application/json \
  http://localhost:8000/api/infer

# Where post_data.json contains:
# {"query": "Test query"}
```

### Concurrent Experiments

```bash
# Run multiple quantum experiments in parallel
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/quantum/experiment \
    -H "Content-Type: application/json" \
    -d '{"experiment_type": "interference_experiment", "iterations": 100}' &
done
wait
```

---

## Common Issues & Troubleshooting

### Backend won't start

**Symptoms**: `ModuleNotFoundError` or connection refused

**Solutions**:
```bash
# Install dependencies
pip install --break-system-packages -r backend/requirements.txt

# Check if port 8000 is in use
lsof -i :8000

# Kill existing processes
pkill -f uvicorn
```

### Frontend build fails

**Symptoms**: TypeScript errors or build errors

**Solutions**:
```bash
cd frontend

# Clean install
rm -rf node_modules package-lock.json
npm install

# Clear Next.js cache
rm -rf .next
npm run build
```

### Cannot connect to backend from frontend

**Symptoms**: "Cannot connect to backend" error in UI

**Solutions**:
```bash
# Check backend is running
curl http://localhost:8000/health

# Check NEXT_PUBLIC_API_URL is set correctly
echo $NEXT_PUBLIC_API_URL

# Restart frontend with correct API URL
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

### Docker build fails

**Symptoms**: Build errors or container won't start

**Solutions**:
```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker-compose build --no-cache

# Check logs
docker-compose logs -f
```

---

## Automated Test Script

Save this as `test_all.sh`:

```bash
#!/bin/bash

echo "🧪 JARVIS-2v Integration Test Suite"
echo "===================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Test function
test_endpoint() {
  local name=$1
  local cmd=$2
  echo -n "Testing $name... "
  if eval $cmd > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
  else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAILED++))
  fi
}

# Start backend in background
echo "Starting backend..."
./scripts/start_backend.sh > /tmp/test_backend.log 2>&1 &
BACKEND_PID=$!
sleep 5

# Run tests
test_endpoint "Health check" "curl -f http://localhost:8000/health"
test_endpoint "Inference" "curl -f -X POST http://localhost:8000/api/infer -H 'Content-Type: application/json' -d '{\"query\":\"test\"}'"
test_endpoint "List adapters" "curl -f http://localhost:8000/api/adapters"
test_endpoint "List artifacts" "curl -f http://localhost:8000/api/artifacts"
test_endpoint "Get config" "curl -f http://localhost:8000/api/config"
test_endpoint "Quantum experiment" "curl -f -X POST http://localhost:8000/api/quantum/experiment -H 'Content-Type: application/json' -d '{\"experiment_type\":\"interference_experiment\",\"iterations\":10}'"

# Cleanup
kill $BACKEND_PID 2>/dev/null

# Summary
echo ""
echo "===================================="
echo "Results: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC}"
echo "===================================="

if [ $FAILED -eq 0 ]; then
  echo "✅ All tests passed!"
  exit 0
else
  echo "❌ Some tests failed"
  exit 1
fi
```

Make it executable and run:
```bash
chmod +x test_all.sh
./test_all.sh
```

---

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Test JARVIS-2v

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install backend dependencies
      run: pip install -r backend/requirements.txt
    
    - name: Test backend imports
      run: python3 -c "from backend.main import app; print('OK')"
    
    - name: Install frontend dependencies
      run: cd frontend && npm ci
    
    - name: Build frontend
      run: cd frontend && npm run build
    
    - name: Run integration tests
      run: ./test_all.sh
```

---

## Success Criteria

Your JARVIS-2v installation is working correctly if:

- ✅ Backend health check returns `"status": "ok"`
- ✅ All API endpoints respond without errors
- ✅ Frontend builds successfully
- ✅ Dashboard loads and displays system status
- ✅ Can create adapters and run quantum experiments
- ✅ Docker Compose starts both services
- ✅ No critical errors in logs

If all checks pass, you're ready to deploy to production! 🚀
