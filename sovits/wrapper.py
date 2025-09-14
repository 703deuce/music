"""
so-vits-svc Voice Cloning Wrapper

Wrapper for so-vits-svc model for singing voice cloning and conversion.
Repository: https://github.com/voicepaw/so-vits-svc
"""

import os
import logging
import subprocess
import tempfile
import json
import shutil
from typing import Dict, Any, Optional, List
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class SoVitsSVCProcessor:
    """so-vits-svc voice cloning processor."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use RunPod persistent volume for model storage
        from model_manager import get_model_manager
        self.model_manager = get_model_manager()
        self.models_dir = self.model_manager.sovits_path
        self.config_template = {
            "model": {
                "ssl_dim": 256,
                "n_speakers": 200,
                "sampling_rate": 44100
            },
            "inference": {
                "auto_predict_f0": True,
                "cluster_model_path": "",
                "f0_predictor": "crepe",
                "feature_retrieval": True,
                "noise_scale": 0.4,
                "noise_scale_w": 0.8
            }
        }
    
    def list_available_voices(self) -> List[str]:
        """List available voice models."""
        voices = []
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, item)
                if os.path.isdir(model_path):
                    # Check for required model files
                    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
                    config_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
                    
                    if pth_files and config_files:
                        voices.append(item)
        
        return voices
    
    def clone_voice_cli(self, input_path: str, target_voice: str, output_path: str,
                       pitch_shift: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Clone voice using so-vits-svc CLI.
        
        Args:
            input_path: Path to input audio file
            target_voice: Target voice model name
            output_path: Path to save cloned audio
            pitch_shift: Pitch shift in semitones
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with cloning metadata
        """
        try:
            # Validate voice model
            available_voices = self.list_available_voices()
            if target_voice not in available_voices:
                logger.warning(f"Voice {target_voice} not found. Available: {available_voices}")
                if available_voices:
                    target_voice = available_voices[0]
                    logger.info(f"Using fallback voice: {target_voice}")
                else:
                    raise ValueError("No voice models available")
            
            model_path = os.path.join(self.models_dir, target_voice)
            
            # Find model files (so-vits-svc uses .pth for model and .json for config)
            pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
            config_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
            
            if not pth_files or not config_files:
                raise ValueError(f"Invalid voice model directory: {model_path}")
            
            model_file = os.path.join(model_path, pth_files[0])
            config_file = os.path.join(model_path, config_files[0])
            
            # Build command using the correct so-vits-svc interface
            cmd = [
                'python', 'inference_main.py',  # so-vits-svc uses this script
                '-m', model_file,
                '-c', config_file,
                '-s', target_voice,  # speaker name
                '-n', os.path.basename(input_path),  # input file name
                '-t', str(int(pitch_shift)),  # transpose (pitch shift)
                '-i', input_path,
                '-o', output_path
            ]
            
            # Add optional parameters specific to so-vits-svc
            if 'cluster_model_path' in kwargs:
                cmd.extend(['-cm', kwargs['cluster_model_path']])
            if 'cluster_infer_ratio' in kwargs:
                cmd.extend(['-cr', str(kwargs['cluster_infer_ratio'])])
            if 'noise_scale' in kwargs:
                cmd.extend(['-ns', str(kwargs['noise_scale'])])
            if 'f0_predictor' in kwargs:
                cmd.extend(['-f0p', kwargs['f0_predictor']])
            
            # Add device specification
            if self.device.type == 'cuda':
                cmd.extend(['-d', 'cuda'])
            else:
                cmd.extend(['-d', 'cpu'])
            
            logger.info(f"Running voice cloning: {' '.join(cmd)}")
            
            # Run cloning
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.path.dirname(model_file)  # Run from model directory
            )
            
            if result.returncode != 0:
                logger.error(f"so-vits-svc failed: {result.stderr}")
                raise RuntimeError(f"Voice cloning failed: {result.stderr}")
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError("Output file was not created")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Voice cloning completed: {file_size} bytes")
            
            return {
                'target_voice': target_voice,
                'pitch_shift': pitch_shift,
                'file_size': file_size,
                'model_path': model_file,
                'config_path': config_file
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Voice cloning timed out")
            raise RuntimeError("Voice cloning timed out")
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            raise
    
    def clone_voice_python(self, input_path: str, target_voice: str, output_path: str,
                          pitch_shift: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Clone voice using so-vits-svc Python API.
        
        Args:
            input_path: Path to input audio file
            target_voice: Target voice model name
            output_path: Path to save cloned audio
            pitch_shift: Pitch shift in semitones
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with cloning metadata
        """
        try:
            # Import so-vits-svc modules (correct import path)
            from so_vits_svc.inference.infer_tool import Svc
            
            # Validate voice model
            available_voices = self.list_available_voices()
            if target_voice not in available_voices:
                if available_voices:
                    target_voice = available_voices[0]
                else:
                    raise ValueError("No voice models available")
            
            model_path = os.path.join(self.models_dir, target_voice)
            
            # Find model files
            pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
            config_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
            
            model_file = os.path.join(model_path, pth_files[0])
            config_file = os.path.join(model_path, config_files[0])
            
            logger.info(f"Loading so-vits-svc model: {model_file}")
            
            # Initialize model with correct parameters
            svc = Svc(
                net_g_path=model_file,
                config_path=config_file,
                device=self.device.type,
                cluster_model_path=kwargs.get('cluster_model_path', None)
            )
            
            # Process audio
            logger.info("Processing voice conversion...")
            
            # Convert voice using the correct interface
            converted_audio = svc.slice_inference(
                raw_audio_path=input_path,
                spk=target_voice,
                tran=int(pitch_shift),
                slice_db=kwargs.get('slice_db', -40),
                cluster_infer_ratio=kwargs.get('cluster_infer_ratio', 0.0),
                auto_predict_f0=kwargs.get('auto_predict_f0', True),
                noice_scale=kwargs.get('noise_scale', 0.4),
                f0_predictor=kwargs.get('f0_predictor', 'pm')
            )
            
            # Save output
            if isinstance(converted_audio, tuple):
                # Some versions return (audio, sample_rate)
                audio_data, sample_rate = converted_audio
            else:
                audio_data = converted_audio
                sample_rate = 44100  # Default sample rate
            
            # Ensure audio is in correct format
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)
            
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)  # Add channel dimension
            
            torchaudio.save(output_path, audio_data, sample_rate)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Voice cloning completed: {file_size} bytes")
            
            return {
                'target_voice': target_voice,
                'pitch_shift': pitch_shift,
                'file_size': file_size,
                'model_path': model_file,
                'config_path': config_file,
                'sample_rate': sample_rate
            }
            
        except ImportError as e:
            logger.error(f"so-vits-svc import failed: {str(e)}")
            raise RuntimeError("so-vits-svc not properly installed")
        except Exception as e:
            logger.error(f"Python API voice cloning failed: {str(e)}")
            raise
    
    def download_voice_model(self, voice_name: str, model_url: str) -> bool:
        """
        Download a voice model from URL.
        
        Args:
            voice_name: Name for the voice model
            model_url: URL to download model from
            
        Returns:
            True if successful
        """
        try:
            import requests
            
            voice_dir = os.path.join(self.models_dir, voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            logger.info(f"Downloading voice model: {voice_name}")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Determine file type and save
            if model_url.endswith('.zip'):
                # Extract zip file
                import zipfile
                zip_path = os.path.join(voice_dir, 'model.zip')
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(voice_dir)
                
                os.remove(zip_path)
                
            else:
                # Direct file download
                filename = os.path.basename(model_url)
                file_path = os.path.join(voice_dir, filename)
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Voice model downloaded: {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download voice model: {str(e)}")
            return False


# Global processor instance
_processor = None


def get_processor() -> SoVitsSVCProcessor:
    """Get or create global processor instance."""
    global _processor
    if _processor is None:
        _processor = SoVitsSVCProcessor()
    return _processor


def clone_voice(input_path: str, target_voice: str, output_path: str, 
               pitch_shift: float = 0.0, use_cli: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Clone/convert voice using so-vits-svc.
    
    Args:
        input_path: Path to input audio file (vocals)
        target_voice: Target voice model name
        output_path: Path to save converted audio
        pitch_shift: Pitch shift in semitones
        use_cli: Whether to use CLI or Python API
        **kwargs: Additional parameters (noise_scale, noise_scale_w, etc.)
        
    Returns:
        Dictionary with conversion metadata
    """
    try:
        processor = get_processor()
        
        # Validate input
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use CLI or Python API
        if use_cli:
            metadata = processor.clone_voice_cli(
                input_path, target_voice, output_path, pitch_shift, **kwargs
            )
        else:
            metadata = processor.clone_voice_python(
                input_path, target_voice, output_path, pitch_shift, **kwargs
            )
        
        return metadata
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {str(e)}")
        
        # Create silent audio as fallback
        try:
            # Copy input as fallback
            shutil.copy2(input_path, output_path)
            return {
                'target_voice': target_voice,
                'pitch_shift': pitch_shift,
                'file_size': os.path.getsize(output_path),
                'error': str(e),
                'fallback': True
            }
        except:
            # Create silence if copy fails
            silent_audio = torch.zeros(1, 44100 * 10)  # 10 seconds of silence
            torchaudio.save(output_path, silent_audio, 44100)
            return {
                'target_voice': target_voice,
                'pitch_shift': pitch_shift,
                'file_size': os.path.getsize(output_path),
                'error': str(e),
                'fallback': True
            }


def get_available_voices() -> List[str]:
    """Get list of available voice models."""
    processor = get_processor()
    return processor.list_available_voices()


def download_voice_model(voice_name: str, model_url: str) -> bool:
    """Download a voice model."""
    processor = get_processor()
    return processor.download_voice_model(voice_name, model_url)


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clone voice with so-vits-svc")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("voice", help="Target voice model name")
    parser.add_argument("--output", default="cloned_voice.wav", help="Output file path")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="Pitch shift in semitones")
    parser.add_argument("--python-api", action="store_true", help="Use Python API instead of CLI")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    
    args = parser.parse_args()
    
    if args.list_voices:
        voices = get_available_voices()
        print(f"Available voices: {voices}")
    else:
        metadata = clone_voice(
            args.input,
            args.voice,
            args.output,
            args.pitch_shift,
            use_cli=not args.python_api
        )
        
        print(f"Voice cloning result: {metadata}")
