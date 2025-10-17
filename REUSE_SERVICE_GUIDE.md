# 🔄 Reuse Existing Service for New Training (Smart Strategy!)

## Why Reuse Instead of Creating New Service?

✅ **Same volume** - Your 50-epoch model is already there, safe!
✅ **Same configuration** - GPU, region, resources already set up
✅ **Cost efficient** - No duplicate volume costs
✅ **Simpler** - Just update & redeploy, don't recreate everything
✅ **Version history** - Keep all models in one volume organized by folders

---

## Strategy Overview

**Volume Structure:**
```
/root/.cache/runs/
├── hybrid_production/           ← OLD 331-pair model (50 epochs) ✅ KEEP THIS!
│   ├── best_model.pt
│   ├── vocab.json
│   └── training_log.csv
│
└── phearun_2483_training/       ← NEW 2,483-pair model (will be created)
    ├── best_model.pt
    ├── vocab.json
    └── training_log.csv
```

**Result:** Both models safe on same volume, organized by folder!

---

## Step-by-Step Plan

### Phase 1: Download Old Model (CRITICAL - Do First!)

**Why first?** Just in case something goes wrong during update.

1. Follow [MODEL_SAFETY_GUIDE.md](MODEL_SAFETY_GUIDE.md) to download your 50-epoch model
2. Verify files are safe locally
3. **Only proceed to Phase 2 after successful download!**

---

### Phase 2: Update Northflank Service

#### Option A: Northflank UI (Recommended)

1. **Go to Service:**
   https://app.northflank.com/p/khmerllm-foundation/services/ml-pdf-extract

2. **Update Repository (if needed):**
   - Click "Build" tab
   - Verify: Branch = `main`, Commit = latest (will be pushed in Phase 3)
   - Click "Rebuild" after we push to GitHub

3. **Verify Volume:**
   - Click "Volumes" tab
   - Verify: `ml-pdf-extract` mounted at `/root/.cache` ✅
   - Access mode: ReadWriteMany ✅

4. **Add Port 8000 (if not exists):**
   - Click "Ports" tab
   - If port 8000 doesn't exist, click "Add Port"
   - Port: `8000`, Protocol: `HTTP`, Public: ✅

5. **Environment Variables (Optional):**
   - Click "Environment" tab
   - You can override any ENV from Dockerfile:
     - `DATA_DIR` = `data/phearun_2483` (already default)
     - `OUTPUT_DIR` = `/root/.cache/runs/phearun_2483_training` (already default)
     - `EPOCHS` = `50` (already default)
     - `BATCH_SIZE` = `16` (or higher if you want faster training)
     - `AUTO_SERVE_MODELS` = `true` (already default)

6. **Deploy:**
   - Click "Deploy" button
   - Wait for build to complete (~5 minutes)
   - Training will auto-start!

#### Option B: Northflank CLI

```bash
# Update service to rebuild from latest commit
northflank deploy service \
  --serviceId ml-pdf-extract \
  --projectId khmerllm-foundation
```

---

### Phase 3: Push Updates to GitHub

This will be done automatically in next step. Includes:
- ✅ New Dockerfile with auto-serving
- ✅ Dataset already pushed (commit 5f6ffb3)
- ✅ All training code up to date

---

## What Happens After Deploy?

### 1. Build Phase (~5 minutes)
```
- Pulls latest code from GitHub
- Builds Docker image with new Dockerfile
- Pushes to Northflank registry
```

### 2. Training Phase (~3-6 hours on A100)
```
- Loads 2,483 training pairs from data/phearun_2483/
- Trains for 50 epochs
- Saves checkpoints to /root/.cache/runs/phearun_2483_training/
- Old model in hybrid_production/ untouched! ✅
```

### 3. Auto-Serve Phase (Automatic!)
```
- Training completes
- HTTP server starts on port 8000
- You can download immediately!
- Container keeps running until you pause
```

---

## Download New Model When Ready

**After training completes (~3-6 hours):**

```bash
# Check if training finished
northflank logs service ml-pdf-extract --projectId khmerllm-foundation --tail 50

# Look for: "✅ TRAINING COMPLETED SUCCESSFULLY!"
# Then download

cd /Users/nicolas.consultant/Downloads/pdf_ocr/trained_models/
mkdir phearun_2483_50epoch

# Get public URL from Northflank (Ports tab)
curl -O https://[YOUR-URL]/phearun_2483_training/best_model.pt
curl -O https://[YOUR-URL]/phearun_2483_training/vocab.json
curl -O https://[YOUR-URL]/phearun_2483_training/training_log.csv

# Or use the script:
/Users/nicolas.consultant/Downloads/pdf_ocr/download_northflank_model.sh https://[YOUR-URL]
# Then manually navigate to /phearun_2483_training/ folder
```

---

## Monitoring Training Progress

### Via Northflank UI
- Go to: Services → ml-pdf-extract → Logs tab
- Watch real-time training output
- Look for epoch progress: `Epoch 1/50`, `Epoch 2/50`, etc.

### Via CLI
```bash
# Follow logs in real-time
northflank logs service ml-pdf-extract \
  --projectId khmerllm-foundation \
  --follow

# Check last 100 lines
northflank logs service ml-pdf-extract \
  --projectId khmerllm-foundation \
  --tail 100
```

---

## Expected Training Metrics

**Dataset Comparison:**

| Metric | Old (331 pairs) | New (2,483 pairs) | Improvement |
|--------|-----------------|-------------------|-------------|
| Training pairs | 331 | 2,483 | **7.5x more data!** |
| Validation CER (expected) | ~8-10% | ~3-5% | **~50% better** |
| Training time (A100) | ~33 min | ~3-6 hours | 10x longer |
| Model robustness | Good | **Excellent** | More diverse examples |

---

## Cost Breakdown

| Phase | Duration | GPU | Cost |
|-------|----------|-----|------|
| **Download old model** | 15 min | A100-80GB | ~$0.50 |
| **Build new image** | 5 min | None | $0 |
| **Train new model** | 3-6 hours | A100-80GB | ~$6-12 |
| **Download new model** | 15 min | A100-80GB | ~$0.50 |
| **Volume storage** | Monthly | None | $0.15/month |

**Total one-time cost:** ~$7-13 for complete new training
**Monthly cost:** ~$0.15 to keep both models on volume

---

## Comparison: Reuse vs New Service

| Aspect | Reuse Existing Service ✅ | Create New Service |
|--------|--------------------------|-------------------|
| **Setup time** | 5 min | 20-30 min |
| **Old model safety** | Same volume ✅ | Need to download first |
| **Volume cost** | $0.15/month | $0.30/month (2 volumes) |
| **Complexity** | Low | Medium |
| **Model organization** | Clean folders | Separate volumes |
| **Rollback** | Easy (same service) | Hard (different services) |

**Winner:** Reuse existing service! 🏆

---

## Rollback Plan (If Something Goes Wrong)

**If training fails:**
1. Check logs for errors
2. Old model still safe in `hybrid_production/` ✅
3. Fix issue and redeploy
4. Or revert to old commit: `git revert HEAD`

**If build fails:**
1. Check build logs in Northflank
2. Fix Dockerfile issue
3. Push to GitHub
4. Redeploy

**Old model is ALWAYS safe** because we're writing to a different folder!

---

## Summary: The Smart Strategy

1. ✅ **Download old model first** (safety!)
2. ✅ **Push improved Dockerfile** to GitHub
3. ✅ **Redeploy existing service** with new code
4. ✅ **Same volume, different folder** for new model
5. ✅ **Both models safe** and organized
6. ✅ **Auto-serving** when training completes

**Total setup:** ~10 minutes
**Total cost:** ~$7-13 for new training
**Risk:** Near zero (old model in different folder)

---

## Next Steps

1. **NOW:** Download your 50-epoch model (see MODEL_SAFETY_GUIDE.md)
2. **THEN:** We'll push Dockerfile to GitHub
3. **THEN:** Redeploy ml-pdf-extract service
4. **WAIT:** 3-6 hours for training
5. **DOWNLOAD:** New model automatically served!

Ready to proceed? 🚀
