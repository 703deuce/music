"""
RunPod Serverless Handler for Music AI API Suite

Supports:
- ACE-Step: Music generation from text prompts
- Demucs v4: Vocal/instrument separation
- so-vits-svc: Singing voice cloning
- Matchering: Automatic audio mastering

All operations are stateless - models loaded per request.
"""

import json
import tempfile
import traceback
import os
from typing import Dict, Any, Optional

from utils import download_audio, upload_audio, setup_logging
from ace_step.wrapper import generate_music
from demucs.wrapper import separate_stems
from sovits.wrapper import clone_voice
from matchering.wrapper import master_track
from model_manager import get_model_manager

logger = setup_logging()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler for music AI operations.
    
    Expected JSON payload:
    {
        'task': 'ace_step'|'separate'|'voice_clone'|'master',
        'input_url': 'http/s3/gcs path to input file' (optional for ace_step),
        'params': {...}   # task-specific options
    }
    
    Task-specific params:
    - ace_step: {'prompt': str, 'duration': int (default 60)}
    - separate: {'model': str (default 'htdemucs'), 'stems': list (default all)}
    - voice_clone: {'target_voice': str, 'pitch_shift': float (default 0)}
    - master: {'reference_url': str (optional), 'loudness': float (default -16)}
    
    Returns:
        JSON with output URLs and metadata
    """
    try:
        # Parse input
        if isinstance(event, dict) and 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
            
        task = body.get('task')
        if not task:
            raise ValueError("Missing required 'task' parameter")
            
        params = body.get('params', {})
        input_url = body.get('input_url')
        
        logger.info(f"Processing task: {task} with params: {params}")
        
        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Created temporary directory: {tmpdir}")
            
            # Download input file if provided
            local_input = None
            if input_url:
                logger.info(f"Downloading input from: {input_url}")
                local_input = download_audio(input_url, tmpdir)
                logger.info(f"Downloaded to: {local_input}")
            
            # Route to appropriate handler
            if task == 'ace_step':
                return handle_ace_step(params, tmpdir)
                
            elif task == 'separate':
                if not local_input:
                    raise ValueError("input_url required for stem separation")
                return handle_separation(local_input, params, tmpdir)
                
            elif task == 'voice_clone':
                if not local_input:
                    raise ValueError("input_url required for voice cloning")
                return handle_voice_clone(local_input, params, tmpdir)
                
            elif task == 'master':
                if not local_input:
                    raise ValueError("input_url required for mastering")
                return handle_mastering(local_input, params, tmpdir)
                
            elif task == 'warmup':
                return handle_warmup(params)
                
            elif task == 'storage_stats':
                return handle_storage_stats()
                
            else:
                raise ValueError(f"Unknown task: {task}")
                
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def handle_ace_step(params: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Handle ACE-Step music generation."""
    prompt = params.get('prompt')
    if not prompt:
        raise ValueError("Missing required 'prompt' parameter for ACE-Step")
        
    duration = params.get('duration', 60)
    output_path = os.path.join(tmpdir, 'generated_music.wav')
    
    logger.info(f"Generating music: prompt='{prompt}', duration={duration}s")
    
    # Generate music
    metadata = generate_music(prompt, duration, output_path)
    
    # Upload result
    audio_url = upload_audio(output_path)
    
    return {
        "task": "ace_step",
        "audio_url": audio_url,
        "metadata": metadata,
        "params": {
            "prompt": prompt,
            "duration": duration
        }
    }


def handle_separation(input_path: str, params: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Handle Demucs stem separation."""
    model = params.get('model', 'htdemucs')
    stems = params.get('stems', ['vocals', 'drums', 'bass', 'other'])
    
    stems_dir = os.path.join(tmpdir, 'stems')
    os.makedirs(stems_dir, exist_ok=True)
    
    logger.info(f"Separating stems: model={model}, stems={stems}")
    
    # Separate stems
    stem_files = separate_stems(input_path, stems_dir, model, stems)
    
    # Upload each stem
    stem_urls = {}
    for stem_name, stem_path in stem_files.items():
        if os.path.exists(stem_path):
            stem_urls[stem_name] = upload_audio(stem_path)
    
    return {
        "task": "separate",
        "stems": stem_urls,
        "metadata": {
            "model": model,
            "original_stems": list(stem_files.keys())
        }
    }


def handle_voice_clone(input_path: str, params: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Handle so-vits-svc voice cloning."""
    target_voice = params.get('target_voice')
    if not target_voice:
        raise ValueError("Missing required 'target_voice' parameter")
        
    pitch_shift = params.get('pitch_shift', 0.0)
    output_path = os.path.join(tmpdir, 'cloned_voice.wav')
    
    logger.info(f"Cloning voice: target={target_voice}, pitch_shift={pitch_shift}")
    
    # Clone voice
    metadata = clone_voice(input_path, target_voice, output_path, pitch_shift)
    
    # Upload result
    audio_url = upload_audio(output_path)
    
    return {
        "task": "voice_clone",
        "audio_url": audio_url,
        "metadata": metadata,
        "params": {
            "target_voice": target_voice,
            "pitch_shift": pitch_shift
        }
    }


def handle_mastering(input_path: str, params: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Handle Matchering audio mastering."""
    reference_url = params.get('reference_url')
    loudness = params.get('loudness', -16.0)
    output_path = os.path.join(tmpdir, 'mastered.wav')
    
    # Download reference if provided
    reference_path = None
    if reference_url:
        logger.info(f"Downloading reference track: {reference_url}")
        reference_path = download_audio(reference_url, tmpdir, filename='reference.wav')
    
    logger.info(f"Mastering audio: loudness={loudness}dB, reference={bool(reference_path)}")
    
    # Master track
    metadata = master_track(input_path, output_path, reference_path, loudness)
    
    # Upload result
    audio_url = upload_audio(output_path)
    
    return {
        "task": "master",
        "audio_url": audio_url,
        "metadata": metadata,
        "params": {
            "loudness": loudness,
            "reference_used": bool(reference_path)
        }
    }


def handle_warmup(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle model warmup to pre-cache models in persistent storage."""
    try:
        model_manager = get_model_manager()
        
        # Get list of models to warmup (default: all)
        models_to_warmup = params.get('models', ['ace_step', 'demucs'])
        
        logger.info(f"Starting model warmup: {models_to_warmup}")
        
        results = {}
        
        if 'ace_step' in models_to_warmup:
            try:
                model_manager.download_ace_step_model()
                results['ace_step'] = {'status': 'success', 'cached': True}
            except Exception as e:
                results['ace_step'] = {'status': 'error', 'error': str(e)}
        
        if 'demucs' in models_to_warmup:
            try:
                demucs_models = params.get('demucs_models', ['htdemucs'])
                cached_models = model_manager.download_demucs_models(demucs_models)
                results['demucs'] = {'status': 'success', 'cached_models': list(cached_models.keys())}
            except Exception as e:
                results['demucs'] = {'status': 'error', 'error': str(e)}
        
        return {
            'task': 'warmup',
            'results': results,
            'storage_stats': model_manager.get_storage_stats()
        }
        
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return {
            'task': 'warmup',
            'error': str(e)
        }


def handle_storage_stats() -> Dict[str, Any]:
    """Handle storage statistics request."""
    try:
        model_manager = get_model_manager()
        
        stats = model_manager.get_storage_stats()
        cached_models = model_manager.list_cached_models()
        
        return {
            'task': 'storage_stats',
            'stats': stats,
            'cached_models': {
                model_key: {
                    'name': info['name'],
                    'type': info['type'],
                    'size_mb': info['size_mb'],
                    'cached_at': info['cached_at']
                }
                for model_key, info in cached_models.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Storage stats failed: {str(e)}")
        return {
            'task': 'storage_stats',
            'error': str(e)
        }


# RunPod serverless entry point
if __name__ == "__main__":
    # For local testing
    import sys
    if len(sys.argv) > 1:
        test_event = json.loads(sys.argv[1])
        result = handler(test_event)
        print(json.dumps(result, indent=2))
