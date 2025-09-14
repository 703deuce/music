# API Keys & Credentials Setup Guide

This guide covers all the API keys and credentials needed for the Music AI API Suite.

## 🔑 Required API Keys

### 1. **RunPod** (Essential)
**Purpose**: Serverless GPU deployment platform

**What you need:**
- RunPod API Key
- RunPod Account with credits

**Setup:**
1. Sign up at https://runpod.io
2. Add credits to your account ($10-50 recommended to start)
3. Go to Settings → API Keys
4. Generate new API key
5. Copy the key (starts with `runpod-...`)

**Usage:**
```bash
# Environment variable
RUNPOD_AI_API_KEY=runpod-your-api-key-here

# Or use in RunPod CLI
runpod config --api-key runpod-your-api-key-here
```

### 2. **HuggingFace** (Recommended)
**Purpose**: Download ACE-Step and other AI models

**What you need:**
- HuggingFace Account (Free)
- HuggingFace Token (for private models or faster downloads)

**Setup:**
1. Sign up at https://huggingface.co
2. Go to Settings → Access Tokens
3. Create new token with "Read" permissions
4. Copy token (starts with `hf_...`)

**Usage:**
```bash
# Environment variable
HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Or login via CLI
huggingface-cli login
```

**Note:** Some models may require acceptance of license terms on HuggingFace.

## 🗄️ Storage API Keys (Choose One)

For storing/serving generated audio files, you need cloud storage:

### Option A: **AWS S3** (Most Popular)
**Setup:**
1. Create AWS account
2. Create IAM user with S3 permissions
3. Generate Access Key and Secret Key
4. Create S3 bucket for audio files

**Environment Variables:**
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=your-audio-bucket
AWS_REGION=us-east-1
```

### Option B: **Google Cloud Storage**
**Setup:**
1. Create Google Cloud account
2. Create project and enable Cloud Storage API
3. Create service account and download JSON key
4. Create storage bucket

**Environment Variables:**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_BUCKET=your-audio-bucket
```

### Option C: **Azure Blob Storage**
**Setup:**
1. Create Azure account
2. Create storage account
3. Get connection string
4. Create container for audio files

**Environment Variables:**
```bash
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_CONTAINER=your-audio-container
```

## 🎵 Model-Specific Requirements

### **ACE-Step** ✅
- **No API key required**
- Uses HuggingFace for model download (optional token for faster downloads)
- Model: `ACE-Step/ACE-Step-v1-3.5B` (publicly available)

### **Demucs** ✅
- **No API key required**
- Models download automatically from Facebook Research
- All models are open-source and public

### **so-vits-svc** ⚠️
- **No API key required for the framework**
- **Voice models**: You need to provide your own voice models
- Models are typically trained custom or downloaded from community

### **Matchering** ✅
- **No API key required**
- Pure algorithmic processing, no external services

## 🚀 RunPod Deployment Keys

### **Container Registry** (Required)
You need to push your Docker image to a registry:

#### Option A: **Docker Hub** (Free)
```bash
# Login
docker login

# Environment (automatic)
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password
```

#### Option B: **GitHub Container Registry** (Free)
```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Environment
GITHUB_TOKEN=ghp_your_token_here
```

#### Option C: **AWS ECR**
```bash
# Login
aws ecr get-login-password --region region | docker login --username AWS --password-stdin account.dkr.ecr.region.amazonaws.com

# Uses AWS credentials above
```

## 🔧 Optional API Keys

### **OpenAI** (Optional)
If you want to enhance prompts or add AI-generated descriptions:
```bash
OPENAI_API_KEY=sk-your-openai-key
```

### **Replicate** (Alternative)
If you want to use Replicate for some models instead:
```bash
REPLICATE_API_TOKEN=r8_your-replicate-token
```

## 📋 Complete Environment Variables Template

Create a `.env` file with all your keys:

```bash
# RunPod (Required)
RUNPOD_AI_API_KEY=runpod-your-api-key-here
RUNPOD_VOLUME_PATH=/runpod-volume

# HuggingFace (Recommended)
HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Cloud Storage (Choose one)
# AWS S3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=your-audio-bucket
AWS_REGION=us-east-1

# OR Google Cloud
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# GCS_BUCKET=your-audio-bucket

# OR Azure
# AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
# AZURE_CONTAINER=your-audio-container

# Model Paths (Auto-configured with volume storage)
ACE_STEP_MODEL_PATH=/runpod-volume/models/ace-step
SOVITS_MODELS_DIR=/runpod-volume/models/sovits
DEMUCS_MODELS_DIR=/runpod-volume/models/demucs

# Optional
OPENAI_API_KEY=sk-your-openai-key
REPLICATE_API_TOKEN=r8_your-replicate-token
```

## 💰 Cost Estimates

### **Required Costs:**
- **RunPod GPU**: $0.20-0.80/hour (only when running)
- **RunPod Volume Storage**: $0.10/GB/month (50GB = $5/month)
- **Cloud Storage**: $0.02-0.05/GB/month for audio files

### **Optional Costs:**
- **HuggingFace Pro**: $20/month (faster downloads, private models)
- **OpenAI API**: $0.002/1K tokens (if using prompt enhancement)

### **Total Monthly Cost:**
- **Minimum**: $5-15/month (storage + light usage)
- **Heavy Usage**: $50-200/month (frequent processing)

## 🛡️ Security Best Practices

### **Environment Variables:**
```bash
# Never commit API keys to git
echo ".env" >> .gitignore

# Use environment-specific configs
# .env.development
# .env.production
```

### **RunPod Secrets:**
Set sensitive keys in RunPod dashboard:
1. Go to your serverless endpoint
2. Settings → Environment Variables
3. Add keys as "Secret" variables

### **Key Rotation:**
- Rotate API keys every 90 days
- Use separate keys for development/production
- Monitor key usage in respective dashboards

## 🚀 Quick Start Checklist

### **Minimum Required (Free/Low Cost):**
- [ ] RunPod account + API key + $10 credits
- [ ] HuggingFace account (free)
- [ ] AWS/GCP/Azure account for storage
- [ ] Docker Hub account (free)

### **Recommended Setup:**
- [ ] RunPod Pro account
- [ ] HuggingFace Pro ($20/month)
- [ ] Dedicated cloud storage bucket
- [ ] Monitoring/logging setup

### **Production Ready:**
- [ ] All above + monitoring
- [ ] Automated key rotation
- [ ] Backup storage strategy
- [ ] Load balancing setup

## 🔍 Testing Your Keys

Use this script to test all your API keys:

```python
#!/usr/bin/env python3
"""Test all API keys and connections."""

import os
import requests
from huggingface_hub import HfApi

def test_runpod():
    api_key = os.getenv('RUNPOD_AI_API_KEY')
    if not api_key:
        return "❌ RUNPOD_AI_API_KEY not set"
    
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        response = requests.get('https://api.runpod.io/v2/pods', headers=headers)
        if response.status_code == 200:
            return "✅ RunPod API key valid"
        else:
            return f"❌ RunPod API error: {response.status_code}"
    except Exception as e:
        return f"❌ RunPod connection error: {e}"

def test_huggingface():
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if not token:
        return "⚠️ HUGGINGFACE_HUB_TOKEN not set (optional)"
    
    try:
        api = HfApi(token=token)
        user = api.whoami()
        return f"✅ HuggingFace token valid for user: {user['name']}"
    except Exception as e:
        return f"❌ HuggingFace token error: {e}"

def test_aws_s3():
    import boto3
    try:
        s3 = boto3.client('s3')
        s3.list_buckets()
        return "✅ AWS S3 credentials valid"
    except Exception as e:
        return f"❌ AWS S3 error: {e}"

if __name__ == "__main__":
    print("🔑 Testing API Keys...")
    print(test_runpod())
    print(test_huggingface())
    print(test_aws_s3())
```

## 🆘 Troubleshooting

### **Common Issues:**

1. **RunPod "Insufficient Credits"**
   - Add more credits to your account
   - Check billing settings

2. **HuggingFace Download Fails**
   - Accept model license terms on website
   - Check if model requires authentication

3. **Storage Upload Fails**
   - Verify bucket/container exists
   - Check IAM permissions
   - Ensure region matches

4. **Container Registry Push Fails**
   - Login to registry: `docker login`
   - Check repository permissions
   - Verify image name format

### **Getting Help:**
- **RunPod**: Discord community, support tickets
- **HuggingFace**: Community forums, documentation
- **Cloud Providers**: Support documentation, forums

---

**🎯 Start with the minimum required keys, then add optional ones as needed for enhanced features!**
