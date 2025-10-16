# Northflank Deployment Guide - Quick Reference

**Repository**: https://github.com/khopilot/ml-pdf-repair
**Service Name**: ml-pdf-extract
**Cost**: ~$1.32 for complete training run (~45 minutes)

---

## 📋 Step-by-Step Configuration

### 1. Basic Information
```
Service name: ml-pdf-extract
Tags: ml-training, khmer-pdf, hybrid-model
Region: Asia - Southeast (Singapore)
```

### 2. Deployment Source
```
Type: Combined (Build and deploy a Git repo)
Repository: https://github.com/khopilot/ml-pdf-repair
Branch: main
Dockerfile path: Dockerfile
Build context: . (root)
```

### 3. GPU Configuration ✅
```
GPU Model: NVIDIA A100
VRAM: 80 GB
GPUs per instance: 1
Price: $1.76/hour
```

### 4. Compute Resources ✅
```
Compute plan: nf-gpu-a100-80-1g
vCPU: 12 dedicated
Memory: 170 GB
Instances: 1
```

### 5. Storage ✅
```
Ephemeral storage: 250 GB
SHM size: 170 GB
```

**OPTIONAL - Add Persistent Volume:**
```
Name: model-checkpoints
Mount path: /app/runs
Size: 50 GB
Type: Persistent
```

### 6. Environment Variables (CRITICAL!)

**Copy from `.env.northflank.simple` or paste these:**

```bash
# Training
DEVICE=cuda
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=1e-4
EARLY_STOPPING=10

# Model Architecture
ATOMIC_EMBED_DIM=128
ATOMIC_HIDDEN_DIM=256
ATOMIC_LAYERS=3
REFINER_D_MODEL=128
REFINER_NHEAD=4
REFINER_LAYERS=3
REFINER_FFN_DIM=512

# Paths
DATA_DIR=data/training_pairs_mega_331p
OUTPUT_DIR=runs/hybrid_a100

# System
PYTHONUNBUFFERED=1
TORCH_HOME=/app/.cache/torch
CUDA_VISIBLE_DEVICES=0
```

### 7. Networking (Optional)
```
Skip or add:
  Port: 8000
  Protocol: HTTP
  Public: Yes (for monitoring)
```

### 8. Advanced Options ✅
```
Docker runtime mode: Default configuration
Health checks: Skip (batch job, not a server)
```

---

## 🚀 After Creation

### Monitor Training
```bash
# Install CLI (if needed)
npm install -g @northflank/cli

# Login
northflank login

# Watch logs
northflank logs ml-pdf-extract --follow

# Check GPU usage
northflank exec ml-pdf-extract -- nvidia-smi

# Check training progress
northflank exec ml-pdf-extract -- tail -f /app/training_hybrid_bs4.log
```

### Expected Log Output
```
Epoch 1/50 [Train]: 100% |████████| loss=3.8
Epoch 1/50 [Val]: CER=0.45, Char Acc=0.55
✅ Saved best model (CER: 0.45)

Epoch 10/50 [Train]: loss=1.2
Epoch 10/50 [Val]: CER=0.15, Char Acc=0.85
✅ Saved best model (CER: 0.15)

...

Training complete! Best CER: 0.089
Best model: /app/runs/hybrid_a100/best_model.pt
```

### Download Trained Model
```bash
# After training completes
northflank download ml-pdf-extract:/app/runs/hybrid_a100/best_model.pt ./

# Or download entire runs directory
northflank download ml-pdf-extract:/app/runs/ ./runs_backup/
```

---

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| A100 GPU (1 GPU) | $1.76/hour |
| Build time (~5-10 min) | $0.15-0.30 |
| Training time (~45 min) | $1.32 |
| **Total Expected** | **~$1.50** |

**Compare to:**
- Local MPS: FREE but 3.75 days
- T4 GPU: $0.88 for ~2.5 hours
- A10G: $1.25 for ~1.25 hours
- **A100: $1.50 for ~45 minutes** ⭐ BEST VALUE

---

## 🎯 Performance Expectations

### Training Speed
- **Batch size**: 64 (vs 4 on MPS)
- **Speed**: ~5-10 seconds per batch
- **Epoch time**: ~1 minute (vs 90 minutes on MPS)
- **Total time**: ~45 minutes for 50 epochs

### Model Performance
- **Target CER**: <10% (vs baseline 76.49%)
- **Expected**: 5-8% CER (excellent)
- **Best case**: <5% CER (production-ready)

---

## ✅ Verification Checklist

Before clicking "Create Service":

- [ ] Repository: `https://github.com/khopilot/ml-pdf-repair`
- [ ] Branch: `main`
- [ ] GPU: A100 80GB (1 GPU)
- [ ] Compute: 12 vCPU, 170GB RAM
- [ ] Environment variables: All 17 variables added
- [ ] Storage: 250GB ephemeral configured
- [ ] Docker runtime: Default configuration

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Check build logs
northflank logs ml-pdf-extract --build

# Common issues:
# - Dockerfile syntax error
# - Missing requirements.txt
# - Git branch incorrect
```

### Training Doesn't Start
```bash
# Check container logs
northflank logs ml-pdf-extract --follow

# Verify environment variables
northflank env list ml-pdf-extract

# Check CUDA available
northflank exec ml-pdf-extract -- python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
```bash
# Reduce batch size (shouldn't happen on A100)
# Update environment variable:
BATCH_SIZE=32  # (instead of 64)
```

### Training Too Slow
```bash
# Check GPU utilization
northflank exec ml-pdf-extract -- nvidia-smi

# Should show:
# - GPU utilization: 80-100%
# - Memory used: 40-60GB
# - Process: Python
```

---

## 📊 Real-Time Monitoring Commands

```bash
# Full dashboard view
watch -n 5 'northflank exec ml-pdf-extract -- nvidia-smi && echo "---" && tail -5 /app/training_hybrid_bs4.log'

# Just loss progression
northflank logs ml-pdf-extract --follow | grep "loss="

# Check if training completed
northflank exec ml-pdf-extract -- ls -lh /app/runs/hybrid_a100/
```

---

## 🎉 Success Indicators

You'll know training is working when you see:

1. ✅ **Build completes**: ~5-10 minutes
2. ✅ **Container starts**: Logs show "Starting training"
3. ✅ **GPU detected**: "Device: cuda"
4. ✅ **Loss decreasing**: Each epoch shows lower loss
5. ✅ **CER improving**: Validation CER dropping
6. ✅ **Checkpoints saved**: "✅ Saved best model"
7. ✅ **Training completes**: "Training complete! Best CER: X.XX"

---

## 📁 Files Available

- `.env.northflank` - Full configuration with comments
- `.env.northflank.simple` - Minimal config for quick setup
- `NORTHFLANK_SETUP_GUIDE.md` - This guide

---

**Ready to deploy!** Just copy the environment variables and follow the checklist above. Training should start automatically after the container is built. 🚀

**Estimated total time**: ~1 hour (10 min build + 45 min training)
**Estimated total cost**: ~$1.50

Good luck! 🎯
