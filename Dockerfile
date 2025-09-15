# RunPod Music AI API Suite Dockerfile
# Optimized for serverless GPU deployment
# Cache buster: 2025-09-14-rebuild-009-debug-ace-step-library-api

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install NumPy first to avoid compatibility issues
RUN pip install numpy==1.24.3

# Install PyTorch with CUDA support (force reinstall to avoid version conflicts)
RUN pip uninstall -y torch torchaudio torchvision
RUN pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir

# Copy requirements and install core dependencies only
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Demucs from GitHub source (required for proper API access)
RUN pip install git+https://github.com/facebookresearch/demucs

# Install ACE-Step from GitHub source (as per official docs)
RUN pip install git+https://github.com/ace-step/ACE-Step.git

# Models will be installed on-demand in the handler or via model manager
# This keeps the base image lightweight and avoids dependency conflicts

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
ENV PYTHONPATH=/workspace

# Default storage configuration (can be overridden by RunPod environment variables)
ENV FIREBASE_STORAGE_BUCKET=aitts-d4c6d.firebasestorage.app

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

# RunPod serverless handler with proper logging
CMD ["python", "-u", "handler.py"]

# Labels for documentation
LABEL maintainer="Music AI API Suite"
LABEL description="Serverless music AI processing with ACE-Step, Demucs, so-vits-svc, and Matchering"
LABEL version="1.0.0"

# Expose port (if needed for local testing)
EXPOSE 8000

# Models will be downloaded on-demand via model_manager.py when needed
# This keeps the Docker image lightweight and builds faster
