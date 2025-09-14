# RunPod Persistent Volume Storage Setup

This guide explains how to set up and use RunPod's persistent volume storage with the Music AI API Suite for optimal performance and cost efficiency.

## 🎯 Benefits of Persistent Volume Storage

### ✅ **Faster Cold Starts**
- Models cached on persistent storage load instantly
- Reduces cold start time from 2-5 minutes to 10-30 seconds
- No need to re-download large models on each serverless invocation

### ✅ **Cost Savings**
- Avoid repeated bandwidth costs for model downloads
- Reduce compute time spent on model loading
- Pay once for storage, use many times

### ✅ **Reliability**
- Models persist across serverless worker shutdowns
- No dependency on external download speeds
- Consistent performance regardless of network conditions

## 🏗️ Volume Storage Architecture

```
/runpod-volume/                    # Persistent volume mount point
├── models/                        # AI model storage
│   ├── ace-step/                  # ACE-Step models
│   │   └── ACE-Step-v1-3.5B/      # Cached model files
│   ├── demucs/                    # Demucs models
│   │   ├── htdemucs.th            # Cached model weights
│   │   └── htdemucs_ft.th         # Additional models
│   ├── sovits/                    # so-vits-svc voice models
│   │   ├── voice_model_1/         # Custom voice models
│   │   └── voice_model_2/         # (User uploaded)
│   └── matchering/                # Matchering cache (minimal)
├── cache/                         # Temporary processing cache
└── model_metadata.json            # Model registry and metadata
```

## 🚀 RunPod Setup Instructions

### 1. Create Persistent Volume

1. **Login to RunPod Dashboard**
   - Go to https://runpod.io
   - Navigate to "Storage" → "Volumes"

2. **Create New Volume**
   ```
   Name: music-ai-models
   Size: 50GB (recommended minimum)
   Region: Same as your serverless endpoints
   ```

3. **Note the Volume ID**
   - Copy the volume ID (e.g., `vol-abc123def456`)
   - You'll need this for serverless configuration

### 2. Deploy Serverless Endpoint

1. **Create Serverless Endpoint**
   ```
   Template: Custom
   Container Image: your-registry/music-ai-suite:latest
   Container Disk: 20GB
   ```

2. **Configure Volume Mount**
   ```
   Volume ID: vol-abc123def456
   Mount Path: /runpod-volume
   ```

3. **Set Environment Variables**
   ```
   RUNPOD_VOLUME_PATH=/runpod-volume
   PYTHONPATH=/workspace:$PYTHONPATH
   ```

### 3. Initial Model Warmup

After deployment, warm up the models to cache them in persistent storage:

```bash
# Warm up all models
curl -X POST https://your-endpoint.runpod.io/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "warmup",
      "params": {
        "models": ["ace_step", "demucs"],
        "demucs_models": ["htdemucs", "htdemucs_ft"]
      }
    }
  }'
```

## 📊 Storage Management

### Check Storage Stats

```json
{
  "task": "storage_stats"
}
```

**Response:**
```json
{
  "task": "storage_stats",
  "stats": {
    "cached_models": 3,
    "total_size_mb": 2048.5,
    "free_space_mb": 45123.2,
    "total_space_mb": 51200.0,
    "volume_path": "/runpod-volume"
  },
  "cached_models": {
    "ace_step_ACE-Step-v1-3.5B": {
      "name": "ACE-Step-v1-3.5B",
      "type": "ace_step",
      "size_mb": 1024.3,
      "cached_at": "2024-01-15T10:30:00"
    }
  }
}
```

### Model Warmup Options

```json
{
  "task": "warmup",
  "params": {
    "models": ["ace_step", "demucs"],
    "demucs_models": ["htdemucs", "htdemucs_ft", "mdx_extra"]
  }
}
```

## 🔧 Advanced Configuration

### Custom Volume Path

If using a different mount path:

```dockerfile
ENV RUNPOD_VOLUME_PATH=/custom/volume/path
```

### Model-Specific Paths

Override individual model paths:

```python
# In your deployment
os.environ['ACE_STEP_MODEL_PATH'] = '/runpod-volume/models/ace-step/custom-model'
os.environ['SOVITS_MODELS_DIR'] = '/runpod-volume/models/sovits'
```

### Storage Optimization

```python
# Clean up old models (30+ days)
{
  "task": "storage_cleanup",
  "params": {
    "days_old": 30
  }
}
```

## 📈 Performance Optimization

### Cold Start Times

| Scenario | Cold Start Time | Notes |
|----------|----------------|-------|
| **No Volume Storage** | 3-5 minutes | Downloads models each time |
| **With Volume Storage** | 15-45 seconds | Models pre-cached |
| **Pre-warmed Models** | 5-15 seconds | Models already loaded |

### Storage Size Recommendations

| Model Type | Size | Recommendation |
|------------|------|----------------|
| **ACE-Step-v1-3.5B** | ~1.2GB | Essential for music generation |
| **Demucs htdemucs** | ~320MB | Most commonly used |
| **Demucs htdemucs_ft** | ~320MB | Higher quality separation |
| **so-vits-svc models** | 50-200MB each | User-specific voice models |
| **Total Recommended** | **50GB** | Allows for multiple models + cache |

## 🛠️ Troubleshooting

### Volume Not Mounting

```bash
# Check if volume is mounted
ls -la /runpod-volume/

# Check environment variable
echo $RUNPOD_VOLUME_PATH
```

### Permission Issues

```bash
# Fix permissions (run as root in container)
chown -R musicai:musicai /runpod-volume
chmod -R 755 /runpod-volume
```

### Storage Full

```bash
# Check disk usage
df -h /runpod-volume

# Clean up old models
python model_manager.py cleanup --days 30
```

### Model Download Failures

```bash
# Check network connectivity
curl -I https://huggingface.co/

# Check model manager logs
python -c "from model_manager import get_model_manager; mm = get_model_manager(); mm.download_ace_step_model()"
```

## 🔄 Model Management Workflow

### 1. Initial Setup
```bash
# Deploy with volume storage
# Run initial warmup
```

### 2. Regular Operations
```bash
# Models load from cache automatically
# New models download and cache on first use
```

### 3. Maintenance
```bash
# Periodic cleanup of old models
# Monitor storage usage
# Update models as needed
```

## 📝 Best Practices

### ✅ **Do**
- Always use persistent volume storage for production
- Warm up models after deployment
- Monitor storage usage regularly
- Use appropriate volume size (50GB+ recommended)
- Clean up old models periodically

### ❌ **Don't**
- Don't skip volume storage setup
- Don't use volumes smaller than 20GB
- Don't ignore storage full warnings
- Don't manually delete model files
- Don't use temporary storage for models

## 🎯 Cost Optimization

### Storage Costs
- **Volume Storage**: ~$0.10/GB/month
- **50GB Volume**: ~$5/month
- **Bandwidth Savings**: $10-50/month (depending on usage)

### ROI Calculation
```
Monthly Volume Cost: $5
Bandwidth Savings: $20
Performance Improvement: Priceless
Net Savings: $15/month + faster response times
```

## 🔗 Integration Examples

### Next.js Frontend
```javascript
// Warm up models on app startup
const warmupModels = async () => {
  await fetch('/api/music-ai', {
    method: 'POST',
    body: JSON.stringify({ task: 'warmup' })
  });
};
```

### Python Client
```python
import requests

# Check storage stats
response = requests.post(endpoint_url, json={
    "task": "storage_stats"
})
print(f"Storage usage: {response.json()}")
```

---

**🚀 With persistent volume storage, your Music AI API Suite will be production-ready with optimal performance and cost efficiency!**
