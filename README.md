# Music AI API Suite - RunPod Serverless Handler

A comprehensive serverless music AI processing suite that combines four powerful open-source models:

- **ACE-Step-v1-3.5B**: Text-to-music generation
- **Demucs v4**: Vocal/instrument stem separation  
- **so-vits-svc**: Singing voice cloning and conversion
- **Matchering 2.0**: Automatic audio mastering

## 🎵 Features

### 🎼 Music Generation (ACE-Step)
- Generate music from text prompts
- Customizable duration and style
- High-quality audio output
- GPU-accelerated inference

### 🎤 Stem Separation (Demucs)
- Separate vocals, drums, bass, and other instruments
- Multiple model options (htdemucs, mdx, etc.)
- Support for 4-stem and 6-stem separation
- Professional-grade quality

### 🎙️ Voice Cloning (so-vits-svc)
- Clone any singing voice
- Real-time voice conversion
- Pitch shifting and fine-tuning
- Custom voice model support

### 🎚️ Audio Mastering (Matchering)
- Automatic audio mastering
- Reference-based matching
- Multiple mastering presets
- Professional loudness standards

## 🏗️ Architecture

```
/
├── handler.py              # Main RunPod serverless handler
├── utils.py               # Audio I/O and utility functions
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── ace_step/             # ACE-Step wrapper
│   ├── __init__.py
│   └── wrapper.py
├── demucs/              # Demucs wrapper
│   ├── __init__.py
│   └── wrapper.py
├── sovits/              # so-vits-svc wrapper
│   ├── __init__.py
│   └── wrapper.py
├── matchering/          # Matchering wrapper
│   ├── __init__.py
│   └── wrapper.py
└── models/              # Model storage (created at runtime)
    ├── ace-step/
    ├── sovits/
    └── demucs/
```

## 🚀 Quick Start

### 1. Setup Persistent Volume Storage (Recommended)

**🎯 Critical for Production:** Use RunPod's persistent volume storage to cache models and reduce cold start times from 5 minutes to 30 seconds!

1. **Create Persistent Volume:**
   - Go to RunPod Dashboard → Storage → Volumes
   - Create new volume: 50GB, same region as your endpoints
   - Note the Volume ID (e.g., `vol-abc123def456`)

2. **See detailed setup guide:** [RUNPOD_VOLUME_SETUP.md](RUNPOD_VOLUME_SETUP.md)

### 2. Deploy to RunPod

1. **Build the Docker image:**
   ```bash
   docker build -t music-ai-suite .
   ```

2. **Push to container registry:**
   ```bash
   docker tag music-ai-suite your-registry/music-ai-suite:latest
   docker push your-registry/music-ai-suite:latest
   ```

3. **Deploy on RunPod:**
   - Create a new serverless endpoint
   - Use your container image
   - **Mount persistent volume:** `/runpod-volume`
   - Configure GPU settings (recommended: RTX 4090 or better)
   - Set environment variables:
     ```
     RUNPOD_VOLUME_PATH=/runpod-volume
     PYTHONPATH=/workspace:$PYTHONPATH
     ```

4. **Warm up models (First time):**
   ```bash
   curl -X POST https://your-endpoint.runpod.io/run \
     -H "Content-Type: application/json" \
     -d '{"input": {"task": "warmup", "params": {"models": ["ace_step", "demucs"]}}}'
   ```

### 2. Local Setup & Testing

1. **Quick Setup:**
   ```bash
   python setup.py
   ```
   This creates `.env` file and installs dependencies.

2. **Configure API Keys:**
   ```bash
   # Edit .env file with your actual API keys
   cp env.example .env
   # Fill in: RUNPOD_AI_API_KEY, storage credentials, etc.
   ```

3. **Test API Keys:**
   ```bash
   python test_api_keys.py
   ```

4. **Deploy to RunPod:**
   ```bash
   python deploy.py
   ```

5. **Test Deployed Endpoint:**
   ```bash
   python test_endpoint.py
   ```

## 📡 API Usage

### Request Format

Send POST requests with JSON payload:

```json
{
  "task": "ace_step|separate|voice_clone|master",
  "input_url": "https://example.com/audio.wav",
  "params": {
    // Task-specific parameters
  }
}
```

### Response Format

```json
{
  "task": "ace_step",
  "audio_url": "https://storage.example.com/output.wav",
  "metadata": {
    "duration": 60.0,
    "file_size": 2048576,
    "model": "ACE-Step-v1-3.5B"
  }
}
```

## 🎯 Task Examples

### Music Generation (ACE-Step)

```json
{
  "task": "ace_step",
  "params": {
    "prompt": "Chill lo-fi hip hop beat with piano and vinyl crackle",
    "duration": 90
  }
}
```

**Response:**
```json
{
  "task": "ace_step",
  "audio_url": "https://storage.example.com/generated_music.wav",
  "metadata": {
    "prompt": "Chill lo-fi hip hop beat with piano and vinyl crackle",
    "duration_actual": 90.5,
    "sample_rate": 44100,
    "model": "ACE-Step-v1-3.5B"
  }
}
```

### Stem Separation (Demucs)

```json
{
  "task": "separate",
  "input_url": "https://example.com/song.wav",
  "params": {
    "model": "htdemucs",
    "stems": ["vocals", "drums", "bass", "other"]
  }
}
```

**Response:**
```json
{
  "task": "separate",
  "stems": {
    "vocals": "https://storage.example.com/vocals.wav",
    "drums": "https://storage.example.com/drums.wav",
    "bass": "https://storage.example.com/bass.wav",
    "other": "https://storage.example.com/other.wav"
  },
  "metadata": {
    "model": "htdemucs",
    "original_stems": ["vocals", "drums", "bass", "other"]
  }
}
```

### Voice Cloning (so-vits-svc)

```json
{
  "task": "voice_clone",
  "input_url": "https://example.com/vocals.wav",
  "params": {
    "target_voice": "singer_model_v1",
    "pitch_shift": 2.0
  }
}
```

**Response:**
```json
{
  "task": "voice_clone",
  "audio_url": "https://storage.example.com/cloned_voice.wav",
  "metadata": {
    "target_voice": "singer_model_v1",
    "pitch_shift": 2.0,
    "file_size": 1024768
  }
}
```

### Audio Mastering (Matchering)

```json
{
  "task": "master",
  "input_url": "https://example.com/raw_mix.wav",
  "params": {
    "reference_url": "https://example.com/reference.wav",
    "loudness": -14.0
  }
}
```

**Response:**
```json
{
  "task": "master",
  "audio_url": "https://storage.example.com/mastered.wav",
  "metadata": {
    "method": "reference_matching",
    "reference_used": true,
    "target_loudness": -14.0,
    "file_size": 2048576
  }
}
```

### Model Management (RunPod Volume Storage)

#### Warm Up Models
```json
{
  "task": "warmup",
  "params": {
    "models": ["ace_step", "demucs"],
    "demucs_models": ["htdemucs", "htdemucs_ft"]
  }
}
```

#### Check Storage Stats
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
    "total_space_mb": 51200.0
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

## ⚙️ Configuration

### Environment Variables

```bash
# RunPod Volume Storage (Recommended)
RUNPOD_VOLUME_PATH=/runpod-volume
PYTHONPATH=/workspace:$PYTHONPATH

# Model paths (fallback if volume storage not used)
ACE_STEP_MODEL_PATH=/workspace/models/ace-step
SOVITS_MODELS_DIR=/workspace/models/sovits
DEMUCS_MODELS_DIR=/workspace/models/demucs

# Storage configuration for audio files
S3_BUCKET=your-audio-bucket
GCS_BUCKET=your-audio-bucket
AZURE_CONTAINER=your-audio-container

# RunPod configuration
RUNPOD_AI_API_KEY=your-api-key
```

### Model Management

1. **ACE-Step Models:**
   - Download from HuggingFace: `ACE-Step/ACE-Step-v1-3.5B`
   - Place in `/workspace/models/ace-step/`

2. **Demucs Models:**
   - Auto-downloaded on first use
   - Available models: `htdemucs`, `htdemucs_ft`, `mdx_extra`, etc.

3. **so-vits-svc Models:**
   - Custom voice models in `/workspace/models/sovits/`
   - Each voice should have `.pth` and `.json` files

4. **Matchering:**
   - No additional models required
   - Uses built-in processing algorithms

## 🔧 Development

### Adding New Models

1. Create a new wrapper module in its own directory
2. Implement the required interface methods
3. Add to `handler.py` routing
4. Update `requirements.txt` with dependencies
5. Update `Dockerfile` if needed

### Testing

```bash
# Run unit tests
pytest tests/

# Test individual components
python -m ace_step.wrapper "test prompt"
python -m demucs.wrapper test_audio.wav
python -m sovits.wrapper test_vocals.wav test_voice
python -m matchering.wrapper test_mix.wav
```

### Local Development Server

```bash
# Install FastAPI for local testing
pip install fastapi uvicorn

# Run local server (create dev_server.py)
uvicorn dev_server:app --host 0.0.0.0 --port 8000
```

## 📋 Requirements

### System Requirements
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM
- 50GB+ storage for models
- Ubuntu 22.04 or compatible Linux distribution

### Python Requirements
- Python 3.9+
- PyTorch 2.0+ with CUDA
- See `requirements.txt` for complete list

## 🐳 Docker Usage

### Build Image
```bash
docker build -t music-ai-suite .
```

### Run Locally
```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  -e CUDA_VISIBLE_DEVICES=0 \
  music-ai-suite
```

### Environment Variables
```bash
docker run --gpus all \
  -e ACE_STEP_MODEL_PATH=/workspace/models/ace-step \
  -e S3_BUCKET=your-bucket \
  music-ai-suite
```

## 🎼 Model Information

### ACE-Step-v1-3.5B
- **Repository:** https://github.com/ace-step/ACE-Step
- **Model:** https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B
- **Purpose:** Text-to-music generation
- **Input:** Text prompt, duration
- **Output:** Generated music audio

### Demucs v4
- **Repository:** https://github.com/facebookresearch/demucs
- **Purpose:** Source separation (vocals, instruments)
- **Input:** Mixed audio
- **Output:** Separated audio stems

### so-vits-svc
- **Repository:** https://github.com/voicepaw/so-vits-svc
- **Purpose:** Singing voice cloning/conversion
- **Input:** Source vocals + target voice model
- **Output:** Converted vocals

### Matchering 2.0
- **Repository:** https://github.com/matchering/matchering
- **Purpose:** Automatic audio mastering
- **Input:** Raw mix (+ optional reference)
- **Output:** Mastered audio

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size or audio duration
   - Use CPU fallback for less critical tasks
   - Monitor GPU memory usage

2. **Model Loading Errors:**
   - Check model file paths and permissions
   - Ensure models are downloaded completely
   - Verify model compatibility

3. **Audio Format Issues:**
   - Ensure input audio is in supported format
   - Check sample rate compatibility
   - Use FFmpeg for format conversion

4. **Slow Cold Starts:**
   - Pre-download models during container build
   - Use model caching strategies
   - Consider keeping containers warm

### Performance Optimization

1. **GPU Utilization:**
   - Use mixed precision training
   - Batch multiple requests when possible
   - Monitor GPU memory usage

2. **Storage:**
   - Use fast SSD storage for models
   - Implement audio file caching
   - Clean up temporary files

3. **Network:**
   - Use CDN for model downloads
   - Implement audio streaming
   - Optimize file upload/download

## 📄 License

This project combines multiple open-source models, each with their own licenses:

- **ACE-Step:** Check repository for license terms
- **Demucs:** MIT License
- **so-vits-svc:** Check repository for license terms  
- **Matchering:** GPL v3.0

Please review individual model licenses before commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

- GitHub Issues: Report bugs and feature requests
- Documentation: Check individual model repositories
- Community: Join discussions in model-specific communities

## 🔮 Roadmap

- [ ] Add more music generation models
- [ ] Implement real-time audio processing
- [ ] Add audio effects and filters
- [ ] Support for longer audio generation
- [ ] Batch processing capabilities
- [ ] WebSocket support for streaming
- [ ] Model fine-tuning endpoints
- [ ] Audio analysis and metadata extraction

---

**Built for RunPod Serverless** 🚀

*Combining the power of ACE-Step, Demucs, so-vits-svc, and Matchering in a single, scalable API.*
