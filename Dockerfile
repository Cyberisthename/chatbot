# JARVIS-2v Docker Image
# Multi-stage build for optimal image size

FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for training
RUN pip install --no-cache-dir \
    datasets \
    transformers \
    pyyaml \
    networkx

# Final stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Create directories for data persistence
RUN mkdir -p \
    /app/adapters \
    /app/quantum_artifacts \
    /app/data/raw \
    /app/models \
    /app/logs

# Expose ports
# 3001 - Node.js server
# 8000 - Python inference backend
EXPOSE 3001 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Default command (can be overridden)
CMD ["python", "inference.py", "models/jarvis-7b-q4_0.gguf", "--port", "8000"]
