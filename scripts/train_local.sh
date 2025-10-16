#!/bin/bash
# Khmer ML Correction - Local Training Script
# Supports CPU, MPS (Apple Silicon), and CUDA

set -e

echo "======================================"
echo "Khmer ML Correction - Local Training"
echo "======================================"

# Detect device
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "Detected: NVIDIA CUDA GPU"
    BATCH_SIZE="${BATCH_SIZE:-16}"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "Detected: Apple Silicon (MPS)"
    BATCH_SIZE="${BATCH_SIZE:-4}"  # Conservative for MPS
else
    DEVICE="cpu"
    echo "Detected: CPU only (slow)"
    BATCH_SIZE="${BATCH_SIZE:-2}"
fi

# Configuration
DATA_DIR="${DATA_DIR:-data/training_pairs_mega_331p}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/hybrid_local_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"

echo ""
echo "Training Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Device: $DEVICE"
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
    --device "$DEVICE" \
    --atomic-embed-dim 128 \
    --atomic-hidden-dim 256 \
    --atomic-layers 3 \
    --refiner-d-model 128 \
    --refiner-nhead 4 \
    --refiner-layers 3 \
    --refiner-ffn-dim 512 \
    --early-stopping 10

echo ""
echo "======================================"
echo "Training Complete!"
echo "Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "======================================"
