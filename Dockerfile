# Khmer PDF ML Correction - Production Dockerfile with Auto-Model-Serving
# Optimized for NVIDIA GPU training on Northflank/Cloud platforms
#
# KEY FEATURES:
# - Trains model with automatic checkpoint resumption
# - Automatically starts HTTP server after training completes
# - Easy model download via browser or curl
# - Configurable via environment variables

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Configuration: Auto-serve models after training?
# Set to "false" to exit after training (for automated pipelines)
# Set to "true" to keep container running and serve models via HTTP
ENV AUTO_SERVE_MODELS=true \
    SERVE_PORT=8000

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
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Create directories for outputs
RUN mkdir -p runs checkpoints logs

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

echo "🚀 Starting Khmer PDF ML Training..."
echo "📊 Configuration:"
echo "   Data: $DATA_DIR"
echo "   Output: $OUTPUT_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Auto-serve models: $AUTO_SERVE_MODELS"
echo ""

# Create lock file for health check
touch /tmp/training.lock

# Run training
python training/train_hybrid.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --device "$DEVICE" \
    --atomic-embed-dim "$ATOMIC_EMBED_DIM" \
    --atomic-hidden-dim "$ATOMIC_HIDDEN_DIM" \
    --atomic-layers "$ATOMIC_LAYERS" \
    --refiner-d-model "$REFINER_D_MODEL" \
    --refiner-nhead "$REFINER_NHEAD" \
    --refiner-layers "$REFINER_LAYERS" \
    --refiner-ffn-dim "$REFINER_FFN_DIM" \
    --early-stopping "$EARLY_STOPPING" \
    --resume

TRAINING_EXIT_CODE=$?

# Remove training lock file
rm -f /tmp/training.lock

echo ""
echo "============================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
else
    echo "⚠️  Training exited with code $TRAINING_EXIT_CODE"
fi
echo "============================================"

# List generated files
echo ""
echo "📦 Generated model files:"
if [ -d "$OUTPUT_DIR" ]; then
    ls -lh "$OUTPUT_DIR"
else
    echo "⚠️  Output directory not found: $OUTPUT_DIR"
fi

# Serve models via HTTP if enabled
if [ "$AUTO_SERVE_MODELS" = "true" ]; then
    echo ""
    echo "============================================"
    echo "🌐 STARTING MODEL DOWNLOAD SERVER"
    echo "============================================"
    echo "Port: $SERVE_PORT"
    echo "Directory: /root/.cache/runs/"
    echo ""
    echo "📥 To download your models:"
    echo "   1. Find your service's public URL in Northflank"
    echo "   2. Navigate to: https://[your-url]/[run-name]/"
    echo "   3. Download: best_model.pt, vocab.json, etc."
    echo ""
    echo "🛑 To stop billing: Pause this service after download"
    echo "============================================"
    echo ""

    # Start HTTP server (this keeps container running)
    cd /root/.cache/runs/
    exec python3 -m http.server "$SERVE_PORT"
else
    echo ""
    echo "✅ Training complete. Container will exit."
    echo "💡 Set AUTO_SERVE_MODELS=true to enable model serving."
    exit $TRAINING_EXIT_CODE
fi
ENTRYPOINT_EOF

RUN chmod +x /app/entrypoint.sh

# Default training parameters (override via environment variables or Northflank UI)
# CRITICAL FIX: Use pre-filtered dataset to avoid 6-hour hang
# Filtered dataset (2,615 pairs), 4x larger model, batch=32, BF16 precision
ENV DATA_DIR="data/training_pairs_filtered_clean" \
    OUTPUT_DIR="/root/.cache/runs/filtered_clean_training" \
    BATCH_SIZE="32" \
    EPOCHS="50" \
    LEARNING_RATE="1e-4" \
    DEVICE="cuda" \
    ATOMIC_EMBED_DIM="256" \
    ATOMIC_HIDDEN_DIM="512" \
    ATOMIC_LAYERS="4" \
    REFINER_D_MODEL="256" \
    REFINER_NHEAD="8" \
    REFINER_LAYERS="4" \
    REFINER_FFN_DIM="1024" \
    EARLY_STOPPING="10"

# Expose port for model download server
EXPOSE 8000

# Health check: Verify training is running or server is up (FIXED: more robust)
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=5 \
  CMD curl -f http://localhost:${SERVE_PORT}/ || [ -f /tmp/training.lock ] || pgrep -f "python.*train_hybrid" || exit 1

# Execute entrypoint
CMD ["/app/entrypoint.sh"]

# Metadata
LABEL maintainer="Khmer PDF Recovery Team"
LABEL version="2.0.0"
LABEL description="Hybrid ML model for Khmer PDF text correction with auto-model-serving"
LABEL gpu.required="true"
LABEL gpu.memory="16GB+"
LABEL features="auto-checkpoint-resume,auto-model-serving,http-download"
