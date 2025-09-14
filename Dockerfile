# RunPod Music AI API Suite Dockerfile
# Optimized for serverless GPU deployment

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

# Install PyTorch with CUDA support first
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install core dependencies only
COPY requirements.txt .
RUN pip install -r requirements.txt

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

# Models will be downloaded on-demand via model_manager.py when needed
# This keeps the Docker image lightweight and builds faster
