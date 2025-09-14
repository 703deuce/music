"""
Demucs v4 Stem Separation Wrapper

Wrapper for Demucs v4 model for vocal/instrument separation.
Repository: https://github.com/facebookresearch/demucs
"""

import os
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class DemucsProcessor:
    """Demucs stem separation processor."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use RunPod persistent volume for model storage
        from model_manager import get_model_manager
        self.model_manager = get_model_manager()
        
        self.available_models = [
            'htdemucs',        # Default hybrid model
            'htdemucs_ft',     # Fine-tuned version
            'htdemucs_6s',     # 6-source separation
            'hdemucs_mmi',     # MMI version
            'mdx',             # MDX model
            'mdx_extra',       # Extra MDX model
            'mdx_q',           # Quantized MDX
            'mdx_extra_q'      # Quantized extra MDX
        ]
        self.stem_names = {
            'htdemucs': ['drums', 'bass', 'other', 'vocals'],
            'htdemucs_ft': ['drums', 'bass', 'other', 'vocals'],
            'htdemucs_6s': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
            'hdemucs_mmi': ['drums', 'bass', 'other', 'vocals'],
            'mdx': ['drums', 'bass', 'other', 'vocals'],
            'mdx_extra': ['drums', 'bass', 'other', 'vocals'],
            'mdx_q': ['drums', 'bass', 'other', 'vocals'],
            'mdx_extra_q': ['drums', 'bass', 'other', 'vocals']
        }
    
    def separate_cli(self, input_path: str, output_dir: str, model: str = 'htdemucs') -> Dict[str, str]:
        """
        Separate stems using Demucs CLI (recommended approach).
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated stems
            model: Demucs model to use
            
        Returns:
            Dictionary mapping stem names to file paths
        """
        if model not in self.available_models:
            logger.warning(f"Unknown model {model}, using htdemucs")
            model = 'htdemucs'
        
        logger.info(f"Separating stems with {model}")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Build command
            cmd = [
                'python', '-m', 'demucs.separate',
                '--model', model,
                '--out', output_dir,
                '--filename', '{track}/{stem}.{ext}',  # Organized output
                '--mp3',  # Output format
                input_path
            ]
            
            # Add device specification
            if self.device.type == 'cuda':
                cmd.extend(['--device', 'cuda'])
            else:
                cmd.extend(['--device', 'cpu'])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run separation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Demucs failed: {result.stderr}")
                raise RuntimeError(f"Demucs separation failed: {result.stderr}")
            
            # Find output files
            track_name = os.path.splitext(os.path.basename(input_path))[0]
            track_dir = os.path.join(output_dir, track_name)
            
            stem_files = {}
            expected_stems = self.stem_names.get(model, ['drums', 'bass', 'other', 'vocals'])
            
            for stem in expected_stems:
                stem_path = os.path.join(track_dir, f"{stem}.mp3")
                if os.path.exists(stem_path):
                    stem_files[stem] = stem_path
                else:
                    logger.warning(f"Expected stem file not found: {stem_path}")
            
            logger.info(f"Separated {len(stem_files)} stems")
            return stem_files
            
        except subprocess.TimeoutExpired:
            logger.error("Demucs separation timed out")
            raise RuntimeError("Stem separation timed out")
        except Exception as e:
            logger.error(f"Stem separation failed: {str(e)}")
            raise
    
    def separate_python(self, input_path: str, output_dir: str, model: str = 'htdemucs') -> Dict[str, str]:
        """
        Separate stems using Demucs Python API.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated stems
            model: Demucs model to use
            
        Returns:
            Dictionary mapping stem names to file paths
        """
        try:
            # Import Demucs modules (correct imports for current version)
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            from demucs import audio
            
            logger.info(f"Loading Demucs model: {model}")
            
            # Load model
            demucs_model = get_model(model)
            demucs_model.to(self.device)
            demucs_model.eval()
            
            # Load audio using Demucs audio module
            wav = audio.read_audio(input_path, demucs_model.samplerate, channels=demucs_model.audio_channels)
            wav = wav.to(self.device)
            
            # Apply model - correct format for apply_model
            logger.info("Applying separation model...")
            with torch.no_grad():
                # apply_model expects (batch, channels, time)
                if wav.dim() == 2:
                    wav = wav.unsqueeze(0)  # Add batch dimension
                sources = apply_model(demucs_model, wav, device=self.device)[0]
            
            # Save stems
            os.makedirs(output_dir, exist_ok=True)
            stem_files = {}
            
            expected_stems = self.stem_names.get(model, ['drums', 'bass', 'other', 'vocals'])
            
            for i, stem_name in enumerate(expected_stems):
                if i < sources.shape[0]:
                    stem_path = os.path.join(output_dir, f"{stem_name}.wav")
                    
                    # Save stem audio using Demucs audio module
                    stem_audio = sources[i].cpu()
                    audio.save_audio(stem_audio, stem_path, demucs_model.samplerate)
                    
                    stem_files[stem_name] = stem_path
                    logger.info(f"Saved {stem_name} stem to {stem_path}")
            
            return stem_files
            
        except ImportError as e:
            logger.error(f"Demucs import failed: {str(e)}")
            raise RuntimeError("Demucs not properly installed")
        except Exception as e:
            logger.error(f"Python API separation failed: {str(e)}")
            raise


# Global processor instance
_processor = None


def get_processor() -> DemucsProcessor:
    """Get or create global processor instance."""
    global _processor
    if _processor is None:
        _processor = DemucsProcessor()
    return _processor


def separate_stems(input_path: str, output_dir: str, model: str = 'htdemucs', 
                  stems: Optional[List[str]] = None, use_cli: bool = True) -> Dict[str, str]:
    """
    Separate audio into stems (vocals, drums, bass, other).
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated stems
        model: Demucs model to use ('htdemucs', 'htdemucs_ft', etc.)
        stems: List of specific stems to extract (None for all)
        use_cli: Whether to use CLI or Python API
        
    Returns:
        Dictionary mapping stem names to file paths
    """
    try:
        processor = get_processor()
        
        # Validate input
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        # Try Python API first, fallback to CLI if needed
        try:
            stem_files = processor.separate_python(input_path, output_dir, model)
        except Exception as e:
            logger.warning(f"Python API failed, trying CLI: {str(e)}")
            stem_files = processor.separate_cli(input_path, output_dir, model)
        
        # Filter requested stems
        if stems:
            stem_files = {k: v for k, v in stem_files.items() if k in stems}
        
        # Validate outputs
        for stem_name, stem_path in stem_files.items():
            if not os.path.exists(stem_path):
                logger.warning(f"Stem file missing: {stem_path}")
            else:
                file_size = os.path.getsize(stem_path)
                logger.info(f"Stem {stem_name}: {file_size} bytes")
        
        return stem_files
        
    except Exception as e:
        logger.error(f"Stem separation failed: {str(e)}")
        
        # Return empty dict on failure
        return {}


def get_available_models() -> List[str]:
    """Get list of available Demucs models."""
    processor = get_processor()
    return processor.available_models


def get_model_stems(model: str) -> List[str]:
    """Get list of stems for a given model."""
    processor = get_processor()
    return processor.stem_names.get(model, ['drums', 'bass', 'other', 'vocals'])


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Separate audio stems with Demucs")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--output", default="./stems", help="Output directory")
    parser.add_argument("--model", default="htdemucs", choices=get_available_models(),
                       help="Demucs model to use")
    parser.add_argument("--stems", nargs='+', help="Specific stems to extract")
    parser.add_argument("--python-api", action="store_true", help="Use Python API instead of CLI")
    
    args = parser.parse_args()
    
    stem_files = separate_stems(
        args.input,
        args.output,
        args.model,
        args.stems,
        use_cli=not args.python_api
    )
    
    print(f"Separated stems: {list(stem_files.keys())}")
    for stem, path in stem_files.items():
        print(f"  {stem}: {path}")
