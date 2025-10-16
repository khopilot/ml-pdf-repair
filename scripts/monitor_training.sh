#!/bin/bash
# Monitor Training Progress - Real-time metrics and GPU usage

set -e

LOG_FILE="${1:-training_hybrid_bs4.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    echo "Usage: $0 [log_file]"
    echo "Example: $0 training_hybrid_bs4.log"
    exit 1
fi

echo "======================================"
echo "Training Monitor"
echo "======================================"
echo "Log file: $LOG_FILE"
echo ""

# Function to show GPU stats
show_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "=== GPU Usage ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
        echo ""
    fi
}

# Function to extract latest metrics from log
show_metrics() {
    echo "=== Latest Training Metrics ==="

    # Extract latest epoch info
    LATEST_EPOCH=$(grep "Epoch" "$LOG_FILE" | tail -1)
    echo "$LATEST_EPOCH"

    # Extract latest validation metrics
    LATEST_VAL=$(grep "Val CER:" "$LOG_FILE" | tail -1)
    if [ ! -z "$LATEST_VAL" ]; then
        echo "$LATEST_VAL"
    fi

    # Extract progress bar
    LATEST_PROGRESS=$(grep "Train\]:" "$LOG_FILE" | tail -1 | grep -o "[0-9]*%.*")
    if [ ! -z "$LATEST_PROGRESS" ]; then
        echo "Progress: $LATEST_PROGRESS"
    fi

    echo ""
}

# Function to show loss trend
show_loss_trend() {
    echo "=== Loss Trend (Last 10 Batches) ==="
    grep "loss=" "$LOG_FILE" | tail -10 | grep -o "loss=[0-9.]*" | cut -d'=' -f2 | nl
    echo ""
}

# Monitoring mode
if [ "$2" == "--watch" ]; then
    echo "Starting live monitor (Ctrl+C to exit)..."
    echo ""

    while true; do
        clear
        echo "======================================"
        echo "Training Monitor - Live"
        echo "======================================"
        echo "Time: $(date)"
        echo ""

        show_gpu
        show_metrics
        show_loss_trend

        echo "Press Ctrl+C to exit"
        sleep 5
    done
else
    # One-time report
    show_gpu
    show_metrics
    show_loss_trend

    echo "For live monitoring, run: $0 $LOG_FILE --watch"
fi
