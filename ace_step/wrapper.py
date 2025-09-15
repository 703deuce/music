"""
ACE-Step Music Generation Wrapper

Uses the official ACE-Step CLI for music generation.
Repository: https://github.com/ace-step/ACE-Step
"""

import os
import json
import logging
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional
import torch
import torchaudio

logger = logging.getLogger(__name__)


class ACEStepGenerator:
    """ACE-Step music generation using official CLI."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use RunPod persistent volume for model storage
        from model_manager import get_model_manager
        self.model_manager = get_model_manager()
        
        # ACE-Step checkpoint path
        self.checkpoint_path = os.path.join(
            os.getenv('RUNPOD_VOLUME_PATH', '/runpod-volume'),
            'models', 'ace-step', 'checkpoints'
        )
        
        self.ace_step_installed = False
        self._check_installation()
    
    def _check_installation(self):
        """Check if ACE-Step is properly installed."""
        try:
            result = subprocess.run(
                ['python', '-c', 'import acestep; print("ACE-Step installed")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("ACE-Step is properly installed")
                self.ace_step_installed = True
            else:
                logger.warning(f"ACE-Step import failed: {result.stderr}")
                self._install_ace_step()
                
        except Exception as e:
            logger.warning(f"ACE-Step check failed: {e}")
            self._install_ace_step()
    
    def _install_ace_step(self):
        """Install ACE-Step if not available."""
        try:
            logger.info("Installing ACE-Step from GitHub...")
            
            result = subprocess.run(
                ['pip', 'install', 'git+https://github.com/ace-step/ACE-Step.git'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                logger.info("ACE-Step installed successfully")
                self.ace_step_installed = True
            else:
                logger.error(f"ACE-Step installation failed: {result.stderr}")
                self.ace_step_installed = False
                
        except Exception as e:
            logger.error(f"Failed to install ACE-Step: {e}")
            self.ace_step_installed = False
    
    def _ensure_checkpoint_path(self):
        """Ensure checkpoint directory exists."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        logger.info(f"Checkpoint path: {self.checkpoint_path}")
    
    def generate_music_cli(self, prompt: str, duration: int = 60, output_path: str = None) -> Dict[str, Any]:
        """
        Generate music using ACE-Step CLI.
        
        Args:
            prompt: Text description of desired music
            duration: Duration in seconds
            output_path: Path to save generated audio
            
        Returns:
            Dictionary with generation results and metadata
        """
        if not self.ace_step_installed:
            raise RuntimeError("ACE-Step is not properly installed")
        
        self._ensure_checkpoint_path()
        
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"ace_step_output_{int(time.time())}.wav")
        
        try:
            # Create a temporary script to run ACE-Step generation
            script_content = f'''
import torch
import torchaudio
from acestep import generate_music

# Generate music
result = generate_music(
    prompt="{prompt}",
    duration={duration},
    checkpoint_path="{self.checkpoint_path}",
    device_id=0 if torch.cuda.is_available() else -1,
    bf16=True if torch.cuda.is_available() else False
)

# Save the generated audio
if result and "audio" in result:
    audio_tensor = result["audio"]
    sample_rate = result.get("sample_rate", 44100)
    
    # Ensure audio is in the right format
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    
    torchaudio.save("{output_path}", audio_tensor.cpu(), sample_rate)
    print("SUCCESS: Audio saved to {output_path}")
else:
    print("ERROR: No audio generated")
'''
            
            # Write the script to a temporary file
            script_path = os.path.join(tempfile.gettempdir(), f"ace_step_script_{int(time.time())}.py")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run the script
            logger.info(f"Generating music with prompt: '{prompt}', duration: {duration}s")
            
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=tempfile.gettempdir()
            )
            
            # Clean up script
            try:
                os.remove(script_path)
            except:
                pass
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                logger.info("Music generated successfully")
                
                # Get audio metadata
                if os.path.exists(output_path):
                    audio_info = torchaudio.info(output_path)
                    
                    return {
                        "success": True,
                        "output_path": output_path,
                        "duration_actual": audio_info.num_frames / audio_info.sample_rate,
                        "sample_rate": audio_info.sample_rate,
                        "channels": audio_info.num_channels,
                        "prompt": prompt,
                        "method": "ace_step_cli"
                    }
                else:
                    raise FileNotFoundError("Generated audio file not found")
            else:
                logger.error(f"ACE-Step generation failed: {result.stderr}")
                raise RuntimeError(f"ACE-Step CLI failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("ACE-Step generation timed out")
            raise RuntimeError("Music generation timed out")
        except Exception as e:
            logger.error(f"ACE-Step generation error: {e}")
            raise
    
    def generate_music_fallback(self, prompt: str, duration: int = 60, output_path: str = None) -> Dict[str, Any]:
        """
        Fallback music generator when ACE-Step is not available.
        
        Args:
            prompt: Text description (used for logging)
            duration: Duration in seconds
            output_path: Path to save generated audio
            
        Returns:
            Dictionary with generation results and metadata
        """
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"fallback_music_{int(time.time())}.wav")
        
        logger.warning(f"Using fallback generator for prompt: '{prompt}'")
        
        sample_rate = 44100
        samples = int(duration * sample_rate)
        
        # Generate more sophisticated musical content
        t = torch.linspace(0, duration, samples)
        
        # Create a musical progression
        frequencies = [220, 277, 330, 440, 554, 659]  # A-major scale notes
        audio = torch.zeros(samples)
        
        # Generate multiple harmonics with time-varying amplitude
        for i, freq in enumerate(frequencies):
            # Phase shift for harmony
            phase = i * torch.pi / 3
            
            # Time-varying envelope (fade in/out with some modulation)
            envelope = torch.sin(torch.pi * t / duration) * (1 + 0.3 * torch.sin(2 * torch.pi * t * 0.5))
            
            # Generate the tone
            tone = torch.sin(2 * torch.pi * freq * t + phase) * envelope * (0.1 / len(frequencies))
            audio += tone
        
        # Add some bass notes
        bass_freq = 110  # A2
        bass_envelope = torch.sin(torch.pi * t / duration) * 0.5
        bass = torch.sin(2 * torch.pi * bass_freq * t) * bass_envelope * 0.15
        audio += bass
        
        # Add rhythmic elements
        beat_pattern = torch.sin(2 * torch.pi * t * 2) > 0.5  # 2 Hz beat
        audio *= (0.7 + 0.3 * beat_pattern.float())
        
        # Normalize and add stereo
        audio = audio / torch.max(torch.abs(audio)) * 0.7
        stereo_audio = torch.stack([audio, audio])  # Stereo
        
        # Save the audio
        torchaudio.save(output_path, stereo_audio, sample_rate)
        
        return {
            "success": True,
            "output_path": output_path,
            "duration_actual": duration,
            "sample_rate": sample_rate,
            "channels": 2,
            "prompt": prompt,
            "method": "fallback_generator",
            "note": "Generated using fallback - ACE-Step not available"
        }


def generate_music(prompt: str, duration: int = 60, output_path: str = None) -> Dict[str, Any]:
    """
    Main function to generate music using ACE-Step.
    
    Args:
        prompt: Text description of desired music
        duration: Duration in seconds (default: 60)
        output_path: Path to save the generated audio file
        
    Returns:
        Dictionary containing:
        - success: Whether generation succeeded
        - output_path: Path to generated audio file
        - duration_actual: Actual duration of generated audio
        - sample_rate: Sample rate of generated audio
        - channels: Number of audio channels
        - prompt: Original prompt used
        - method: Generation method used
        - error: Error message if generation failed
    """
    generator = ACEStepGenerator()
    
    try:
        if generator.ace_step_installed:
            return generator.generate_music_cli(prompt, duration, output_path)
        else:
            logger.warning("ACE-Step not available, using fallback generator")
            return generator.generate_music_fallback(prompt, duration, output_path)
            
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        
        # Try fallback generator as last resort
        try:
            logger.info("Attempting fallback generation...")
            return generator.generate_music_fallback(prompt, duration, output_path)
        except Exception as fallback_error:
            logger.error(f"Fallback generation also failed: {fallback_error}")
            
            return {
                "success": False,
                "error": str(e),
                "fallback_error": str(fallback_error),
                "prompt": prompt,
                "method": "failed"
            }


if __name__ == "__main__":
    # Test the wrapper
    import sys
    
    if len(sys.argv) > 1:
        test_prompt = sys.argv[1]
    else:
        test_prompt = "Upbeat electronic music with synthesizers"
    
    print(f"Testing ACE-Step wrapper with prompt: '{test_prompt}'")
    
    result = generate_music(test_prompt, duration=10)
    print(f"Result: {json.dumps(result, indent=2)}")