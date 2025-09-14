"""
Utility functions for audio I/O, storage, and common operations.
"""

import os
import logging
import requests
import tempfile
import uuid
from typing import Optional, Union
from urllib.parse import urlparse
import soundfile as sf
import numpy as np


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def download_audio(url: str, tmpdir: str, filename: Optional[str] = None) -> str:
    """
    Download audio file from URL to local temporary directory.
    
    Args:
        url: URL to download from (http/https/s3/gcs)
        tmpdir: Temporary directory to save file
        filename: Optional filename, auto-generated if None
        
    Returns:
        Local path to downloaded file
    """
    if not filename:
        # Generate filename from URL or use UUID
        parsed = urlparse(url)
        if parsed.path:
            filename = os.path.basename(parsed.path)
            if not filename or '.' not in filename:
                filename = f"input_{uuid.uuid4().hex[:8]}.wav"
        else:
            filename = f"input_{uuid.uuid4().hex[:8]}.wav"
    
    local_path = os.path.join(tmpdir, filename)
    
    try:
        logger.info(f"Downloading {url} to {local_path}")
        
        # Handle different URL schemes
        if url.startswith(('http://', 'https://')):
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        elif url.startswith('s3://'):
            # AWS S3 download
            import boto3
            s3 = boto3.client('s3')
            bucket, key = url[5:].split('/', 1)
            s3.download_file(bucket, key, local_path)
            
        elif url.startswith('gs://'):
            # Google Cloud Storage download
            from google.cloud import storage
            client = storage.Client()
            bucket_name, blob_name = url[5:].split('/', 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            
        else:
            raise ValueError(f"Unsupported URL scheme: {url}")
            
        # Verify file was downloaded
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise ValueError(f"Failed to download or empty file: {url}")
            
        logger.info(f"Successfully downloaded {os.path.getsize(local_path)} bytes")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise


def upload_audio(local_path: str, public: bool = True) -> str:
    """
    Upload audio file to storage and return public URL.
    
    Supports Firebase Storage, AWS S3, Google Cloud Storage, and Azure Blob.
    
    Args:
        local_path: Local file path to upload
        public: Whether to make file publicly accessible
        
    Returns:
        Public URL to uploaded file
    """
    if not os.path.exists(local_path):
        raise ValueError(f"File does not exist: {local_path}")
        
    file_size = os.path.getsize(local_path)
    logger.info(f"Uploading {local_path} ({file_size} bytes)")
    
    filename = os.path.basename(local_path)
    file_id = uuid.uuid4().hex[:12]
    
    # Try Firebase Storage first (if configured)
    firebase_config = {
        'api_key': os.getenv('FIREBASE_API_KEY'),
        'project_id': os.getenv('FIREBASE_PROJECT_ID'),
        'storage_bucket': os.getenv('FIREBASE_STORAGE_BUCKET')
    }
    
    # Log configuration status for debugging
    logger.info(f"Firebase config: api_key={'***' if firebase_config['api_key'] else None}, "
               f"project_id={firebase_config['project_id']}, "
               f"bucket={firebase_config['storage_bucket']}")
    
    if all(firebase_config.values()):
        try:
            return _upload_to_firebase(local_path, filename, file_id, firebase_config)
        except Exception as e:
            logger.warning(f"Firebase upload failed, trying alternatives: {e}")
    
    # Try AWS S3
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        try:
            return _upload_to_s3(local_path, filename, file_id)
        except Exception as e:
            logger.warning(f"S3 upload failed, trying alternatives: {e}")
    
    # Try Google Cloud Storage
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or os.getenv('GCS_BUCKET'):
        try:
            return _upload_to_gcs(local_path, filename, file_id)
        except Exception as e:
            logger.warning(f"GCS upload failed, trying alternatives: {e}")
    
    # Try Azure Blob Storage
    if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
        try:
            return _upload_to_azure(local_path, filename, file_id)
        except Exception as e:
            logger.warning(f"Azure upload failed: {e}")
    
    # Fallback: return mock URL for development
    mock_url = f"https://storage.example.com/outputs/{file_id}/{filename}"
    logger.warning(f"No storage configured, using mock URL: {mock_url}")
    return mock_url


def _upload_to_firebase(local_path: str, filename: str, file_id: str, config: dict) -> str:
    """Upload file to Firebase Storage."""
    import requests
    
    # Generate unique path
    upload_path = f"music-ai-outputs/{file_id}/{filename}"
    
    # Firebase Storage REST API upload
    bucket_name = config['storage_bucket']
    upload_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o?name={upload_path}"
    
    with open(local_path, 'rb') as f:
        response = requests.post(
            upload_url,
            data=f,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=300
        )
    
    if not response.ok:
        raise Exception(f"Firebase upload failed: {response.status_code} {response.text}")
    
    # Generate public download URL
    download_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{upload_path}?alt=media"
    
    logger.info(f"Uploaded to Firebase Storage: {download_url}")
    return download_url


def _upload_to_s3(local_path: str, filename: str, file_id: str) -> str:
    """Upload file to AWS S3."""
    import boto3
    
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET', 'music-ai-outputs')
    key = f"outputs/{file_id}/{filename}"
    
    s3.upload_file(local_path, bucket, key)
    
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    logger.info(f"Uploaded to S3: {url}")
    return url


def _upload_to_gcs(local_path: str, filename: str, file_id: str) -> str:
    """Upload file to Google Cloud Storage."""
    from google.cloud import storage
    
    client = storage.Client()
    bucket_name = os.getenv('GCS_BUCKET', 'music-ai-outputs')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"outputs/{file_id}/{filename}")
    
    blob.upload_from_filename(local_path)
    blob.make_public()
    
    logger.info(f"Uploaded to GCS: {blob.public_url}")
    return blob.public_url


def _upload_to_azure(local_path: str, filename: str, file_id: str) -> str:
    """Upload file to Azure Blob Storage."""
    from azure.storage.blob import BlobServiceClient
    
    client = BlobServiceClient.from_connection_string(
        os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    )
    container = os.getenv('AZURE_CONTAINER', 'music-ai-outputs')
    blob_name = f"outputs/{file_id}/{filename}"
    
    with open(local_path, 'rb') as f:
        client.get_blob_client(container=container, blob=blob_name).upload_blob(f)
    
    url = f"https://{client.account_name}.blob.core.windows.net/{container}/{blob_name}"
    logger.info(f"Uploaded to Azure: {url}")
    return url


def validate_audio_file(file_path: str) -> dict:
    """
    Validate and get metadata for audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio metadata
    """
    try:
        info = sf.info(file_path)
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'frames': info.frames
        }
    except Exception as e:
        logger.error(f"Failed to validate audio file {file_path}: {str(e)}")
        raise ValueError(f"Invalid audio file: {str(e)}")


def convert_audio_format(input_path: str, output_path: str, 
                        target_sr: int = 44100, target_channels: int = 2) -> str:
    """
    Convert audio to target format/sample rate.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_sr: Target sample rate
        target_channels: Target number of channels
        
    Returns:
        Path to converted file
    """
    try:
        # Read audio
        audio, sr = sf.read(input_path)
        
        # Convert to target channels
        if audio.ndim == 1 and target_channels == 2:
            # Mono to stereo
            audio = np.column_stack([audio, audio])
        elif audio.ndim == 2 and target_channels == 1:
            # Stereo to mono
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Write converted audio
        sf.write(output_path, audio, target_sr)
        
        logger.info(f"Converted audio: {input_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to convert audio: {str(e)}")
        raise


def get_audio_duration(file_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        logger.error(f"Failed to get duration for {file_path}: {str(e)}")
        return 0.0


def ensure_audio_format(file_path: str, required_format: str = 'wav') -> str:
    """
    Ensure audio file is in required format, convert if needed.
    
    Args:
        file_path: Input audio file path
        required_format: Required format ('wav', 'mp3', etc.)
        
    Returns:
        Path to file in required format (may be same as input)
    """
    current_format = os.path.splitext(file_path)[1][1:].lower()
    
    if current_format == required_format.lower():
        return file_path
    
    # Convert to required format
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}.{required_format}"
    
    return convert_audio_format(file_path, output_path)


def cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {str(e)}")


# Audio processing utilities
def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    # Calculate RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms > 0:
        # Convert target dB to linear scale
        target_rms = 10 ** (target_db / 20)
        # Apply gain
        gain = target_rms / rms
        audio = audio * gain
    
    # Prevent clipping
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def apply_fade(audio: np.ndarray, sample_rate: int, 
               fade_in: float = 0.1, fade_out: float = 0.1) -> np.ndarray:
    """Apply fade in/out to audio."""
    fade_in_samples = int(fade_in * sample_rate)
    fade_out_samples = int(fade_out * sample_rate)
    
    if len(audio.shape) == 1:
        # Mono
        if fade_in_samples > 0:
            fade_curve = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_curve
            
        if fade_out_samples > 0:
            fade_curve = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_curve
    else:
        # Stereo
        if fade_in_samples > 0:
            fade_curve = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_curve[:, np.newaxis]
            
        if fade_out_samples > 0:
            fade_curve = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_curve[:, np.newaxis]
    
    return audio
