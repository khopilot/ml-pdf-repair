# Khmer PDF ML Correction - Production Dockerfile
# Optimized for NVIDIA GPU training on Northflank/Cloud platforms

# Build stage - Use official PyTorch base with CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY models/ ./models/
COPY training/ ./training/
COPY data/ ./data/
COPY inference/ ./inference/

# Copy training scripts
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh

# Create directories for outputs
RUN mkdir -p runs checkpoints logs

# Health check endpoint (if running as service)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command: Run training
# Override with docker run or docker-compose command
CMD ["python", "training/train_hybrid.py", \
     "--data-dir", "data/training_pairs_mega_331p", \
     "--output-dir", "runs/hybrid_production", \
     "--batch-size", "16", \
     "--epochs", "50", \
     "--lr", "1e-4", \
     "--device", "cuda", \
     "--atomic-embed-dim", "128", \
     "--atomic-hidden-dim", "256", \
     "--atomic-layers", "3", \
     "--refiner-d-model", "128", \
     "--refiner-nhead", "4", \
     "--refiner-layers", "3", \
     "--refiner-ffn-dim", "512", \
     "--early-stopping", "10"]

# Metadata
LABEL maintainer="Khmer PDF Recovery Team"
LABEL version="1.0.0"
LABEL description="Hybrid ML model for Khmer PDF text correction"
LABEL gpu.required="true"
LABEL gpu.memory="16GB+"
