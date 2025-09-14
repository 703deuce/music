# RunPod Music AI API Suite Dockerfile
# Optimized for serverless GPU deployment

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libtag1-dev \
    libchromaprint-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (for better dependency resolution)
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional dependencies that might not be in requirements.txt
RUN pip install runpod

# Install models from source (correct repositories)
# Install ACE-Step from source
RUN git clone https://github.com/ace-step/ACE-Step.git /tmp/ace-step && \
    cd /tmp/ace-step && \
    pip install -e . && \
    rm -rf /tmp/ace-step

# Install so-vits-svc from source (correct repository)
RUN git clone https://github.com/voicepaw/so-vits-svc.git /tmp/so-vits-svc && \
    cd /tmp/so-vits-svc && \
    pip install -e . && \
    rm -rf /tmp/so-vits-svc

# Create directories for local temp data
RUN mkdir -p /workspace/temp

# Copy application code
COPY . /workspace/

# Set permissions
RUN chmod +x /workspace/handler.py

# Create a non-root user (optional, for security)
RUN useradd -m -u 1000 musicai && \
    chown -R musicai:musicai /workspace
USER musicai

# Set environment variables for RunPod volume storage
ENV RUNPOD_VOLUME_PATH=/runpod-volume
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Create volume mount point (RunPod will mount the persistent volume here)
USER root
RUN mkdir -p /runpod-volume && \
    chown -R musicai:musicai /runpod-volume
USER musicai

# Pre-warm models on container start (optional, can be done via API call)
# This will check if models exist in volume storage and download if needed

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# RunPod serverless handler
CMD ["python", "-u", "handler.py"]

# Alternative: If using RunPod's handler system
# CMD ["python", "-m", "runpod.serverless.start", "--handler_file=handler.py"]

# Labels for documentation
LABEL maintainer="Music AI API Suite"
LABEL description="Serverless music AI processing with ACE-Step, Demucs, so-vits-svc, and Matchering"
LABEL version="1.0.0"

# Expose port (if needed for local testing)
EXPOSE 8000

# Optional: Add model download script
COPY <<EOF /workspace/download_models.py
#!/usr/bin/env python3
"""
Download required models for the Music AI API Suite.
Run this script to pre-download models and reduce cold start times.
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_ace_step_model():
    """Download ACE-Step model."""
    try:
        from huggingface_hub import snapshot_download
        model_path = "/workspace/models/ace-step"
        
        logger.info("Downloading ACE-Step model...")
        snapshot_download(
            repo_id="ACE-Step/ACE-Step-v1-3.5B",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        logger.info(f"ACE-Step model downloaded to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download ACE-Step model: {e}")

def download_demucs_models():
    """Download Demucs models."""
    try:
        import demucs.pretrained
        
        models = ['htdemucs', 'htdemucs_ft', 'mdx_extra']
        for model in models:
            try:
                logger.info(f"Downloading Demucs model: {model}")
                demucs.pretrained.get_model(model)
                logger.info(f"Downloaded {model}")
            except Exception as e:
                logger.warning(f"Failed to download {model}: {e}")
    except Exception as e:
        logger.error(f"Failed to download Demucs models: {e}")

def setup_sovits_models():
    """Setup so-vits-svc models directory."""
    models_dir = Path("/workspace/models/sovits")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example voice model structure
    example_dir = models_dir / "example_voice"
    example_dir.mkdir(exist_ok=True)
    
    logger.info(f"so-vits-svc models directory created: {models_dir}")
    logger.info("Add your voice models to this directory")

if __name__ == "__main__":
    logger.info("Starting model downloads...")
    
    download_demucs_models()
    download_ace_step_model()
    setup_sovits_models()
    
    logger.info("Model setup complete!")
EOF

RUN chmod +x /workspace/download_models.py

# Optional: Run model download during build (uncomment if desired)
# RUN python /workspace/download_models.py
