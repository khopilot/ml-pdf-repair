#!/bin/bash
# Khmer ML Correction - CUDA/GPU Training Script
# Optimized for Northflank GPU instances (T4/A10G/A100)

set -e  # Exit on error

echo "======================================"
echo "Khmer ML Correction - GPU Training"
echo "======================================"

# Detect GPU type
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# Configuration
DATA_DIR="${DATA_DIR:-data/training_pairs_mega_331p}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/hybrid_cuda_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"

# GPU-specific batch size recommendations
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if [[ "$GPU_NAME" == *"T4"* ]]; then
    echo "Detected NVIDIA T4 (16GB VRAM)"
    BATCH_SIZE="${BATCH_SIZE:-12}"
elif [[ "$GPU_NAME" == *"A10"* ]]; then
    echo "Detected NVIDIA A10G (24GB VRAM)"
    BATCH_SIZE="${BATCH_SIZE:-24}"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    echo "Detected NVIDIA A100 (40GB+ VRAM)"
    BATCH_SIZE="${BATCH_SIZE:-64}"
else
    echo "Unknown GPU: $GPU_NAME"
    echo "Using default batch size: $BATCH_SIZE"
fi

echo ""
echo "Training Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Device: cuda"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
python3 training/train_hybrid.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device cuda \
    --atomic-embed-dim 128 \
    --atomic-hidden-dim 256 \
    --atomic-layers 3 \
    --refiner-d-model 128 \
    --refiner-nhead 4 \
    --refiner-layers 3 \
    --refiner-ffn-dim 512 \
    --early-stopping 10 \
    ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"}

echo ""
echo "======================================"
echo "Training Complete!"
echo "Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "======================================"
