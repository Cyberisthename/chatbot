# Multi-stage build for JARVIS-2v
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies for RAG
RUN pip install --no-cache-dir faiss-cpu numpy PyPDF2 networkx PyYAML python-multipart

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data adapters quantum_artifacts vector_index models

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Expose port
EXPOSE 3001

# Set environment
ENV PYTHONPATH=/app/src

# Start application
CMD ["python", "-m", "src.api.main"]

# Development stage
FROM base as development

# Copy application code
COPY . .

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio

# Create directories
RUN mkdir -p data adapters quantum_artifacts vector_index models

# Set environment
ENV PYTHONPATH=/app/src

# Expose port for development
EXPOSE 3001

# Start in development mode
CMD ["python", "-m", "src.api.main"]