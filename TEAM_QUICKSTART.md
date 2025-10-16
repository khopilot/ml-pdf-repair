# Khmer ML Correction - Team Quickstart Guide

**Get training in 5 minutes!**

## What This Does

Trains a hybrid deep learning model to correct corrupted Khmer text extracted from PDFs.

- **Input**: Corrupted PDF text (wrong characters)
- **Output**: Corrected Khmer text
- **Method**: Hybrid architecture (Atomic Mapper + Context Refiner)

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (for production training)
- OR Apple Silicon Mac (for testing)
- 16GB+ RAM
- 10GB disk space

## Quick Start (Local Testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test training (auto-detects device)
bash scripts/train_local.sh

# Expected output:
#   - Epoch 1 progress bars
#   - Loss decreasing (5.4 → 4.4 → 3.8...)
#   - Checkpoints saved to runs/hybrid_local_*/
```

## GPU Training (Northflank/Cloud)

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t khmer-ml .

# Run on GPU
docker run --gpus all \
  -v $(pwd)/runs:/app/runs \
  khmer-ml

# Monitor
docker logs -f <container_id>
```

### Option 2: Direct Script

```bash
# CUDA training (auto-tunes batch size for GPU)
bash scripts/train_cuda.sh

# Custom config
BATCH_SIZE=24 EPOCHS=30 bash scripts/train_cuda.sh
```

### Option 3: Northflank Deployment

```bash
# Install Northflank CLI
npm install -g @northflank/cli

# Login
northflank login

# Deploy
northflank create deployment --file northflank.yaml

# Monitor
northflank logs khmer-ml-training --follow
```

## Expected Performance

| GPU | Batch Size | Time/Epoch | Total Time (50 epochs) | Cost |
|-----|-----------|-----------|---------------------|------|
| **T4** (16GB) | 12-16 | ~3 min | ~2.5 hours | $0.875 |
| **A10G** (24GB) | 24-32 | ~1.5 min | ~1.25 hours | $1.25 |
| **A100** (40GB) | 64+ | <1 min | <1 hour | $3.00 |
| **MPS** (Mac) | 4-8 | ~8 min | ~6.5 hours | Local |

## What To Expect

### Training Metrics
- **Initial Loss**: ~5.5 (random predictions)
- **Target Loss**: <1.0 (good performance)
- **CER**: Aim for <10% (current baseline: 77%)

### Output Files
```
runs/hybrid_<timestamp>/
├── best_model.pt          # Best checkpoint
├── last_model.pt          # Latest checkpoint
├── train_metrics.csv      # Training history
└── config.json            # Hyperparameters
```

## Monitoring Training

```bash
# Real-time monitoring
bash scripts/monitor_training.sh <log_file> --watch

# One-time check
bash scripts/monitor_training.sh training_hybrid_bs4.log
```

## Common Issues

### Out of Memory (OOM)
- **Solution**: Reduce batch size
  ```bash
  BATCH_SIZE=8 bash scripts/train_cuda.sh
  ```

### Slow Training (CPU)
- **Cause**: No GPU detected
- **Solution**: Check CUDA installation or use cloud GPU

### Module Not Found
- **Solution**: Install requirements
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Info

- **Training**: 251 pairs (corrupted → correct)
- **Validation**: 28 pairs
- **Test**: 46 pairs
- **Total Size**: 2.9MB
- **Source**: Gold standard + forensic extraction

## Next Steps

1. **Test locally** → Verify environment works
2. **Deploy to GPU** → Run production training
3. **Monitor metrics** → Watch CER decrease
4. **Evaluate model** → Test on held-out set
5. **Deploy inference** → Use trained model

## Support

- **Full docs**: See [README.md](README.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) (if needed)
- **Docker details**: See [docker-compose.yml](docker-compose.yml)
- **North flank**: See [northflank.yaml](northflank.yaml)

## Verification Checklist

- [ ] Python 3.10+ installed (`python3 --version`)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] GPU detected (`nvidia-smi` or `python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Dataset present (`ls data/training_pairs_mega_331p/`)
- [ ] Scripts executable (`ls -l scripts/*.sh`)

---

**Ready to train!** Start with `bash scripts/train_local.sh` for immediate testing.
