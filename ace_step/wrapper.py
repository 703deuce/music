"""
ACE-Step Music Generation Wrapper

Uses the official ACE-Step CLI via subprocess.
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


def generate_music(prompt: str, duration: int = 60, output_path: str = None) -> Dict[str, Any]:
    """
    Generate music using ACE-Step CLI via subprocess.
    
    Args:
        prompt: Text description of desired music
        duration: Duration in seconds (default: 60)
        output_path: Path to save the generated audio file
        
    Returns:
        Dictionary containing generation results and metadata
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"ace_step_output_{int(time.time())}.wav")
    
    # Get checkpoint path from persistent volume
    checkpoint_path = os.path.join(
        os.getenv('RUNPOD_VOLUME_PATH', '/runpod-volume'),
        'models', 'ace-step', 'checkpoints'
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_path, exist_ok=True)
    
    logger.info(f"Generating music with ACE-Step CLI")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Duration: {duration}s")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Try to use ACE-Step as a Python library (API Usage approach)
        script_content = f'''
import sys
import os
import torch

# Set checkpoint path
checkpoint_path = "{checkpoint_path}"
os.makedirs(checkpoint_path, exist_ok=True)

try:
    # Import ACE-Step library
    import acestep
    print("ACE-Step library imported successfully")
    
    # Try to find the correct API for generation
    print("Available ACE-Step attributes:")
    print([attr for attr in dir(acestep) if not attr.startswith('_')])
    
    # This is exploratory - we need to find the right API
    print("SUCCESS: ACE-Step library exploration completed")
    
except ImportError as e:
    print(f"IMPORT_ERROR: ACE-Step not available - {{e}}")
    sys.exit(1)
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        # Write and execute the exploration script
        script_path = os.path.join(tempfile.gettempdir(), f"ace_step_explore_{int(time.time())}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info("Exploring ACE-Step library API...")
        
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for exploration
            cwd=tempfile.gettempdir()
        )
        
        # Clean up script
        try:
            os.remove(script_path)
        except:
            pass
        
        logger.info(f"ACE-Step CLI output: {result.stdout}")
        if result.stderr:
            logger.warning(f"ACE-Step CLI errors: {result.stderr}")
        
        # Check if generation was successful
        if result.returncode == 0:
            logger.info("ACE-Step generation completed successfully")
            
            # Verify output file exists
            if os.path.exists(output_path):
                # Get audio metadata
                try:
                    audio_info = torchaudio.info(output_path)
                    
                    return {
                        "success": True,
                        "output_path": output_path,
                        "duration_actual": audio_info.num_frames / audio_info.sample_rate,
                        "sample_rate": audio_info.sample_rate,
                        "channels": audio_info.num_channels,
                        "prompt": prompt,
                        "method": "ace_step_cli",
                        "file_size": os.path.getsize(output_path)
                    }
                except Exception as info_error:
                    logger.warning(f"Could not get audio info: {info_error}")
                    
                    return {
                        "success": True,
                        "output_path": output_path,
                        "duration_actual": duration,
                        "sample_rate": 44100,  # Default
                        "channels": 2,  # Default
                        "prompt": prompt,
                        "method": "ace_step_cli",
                        "file_size": os.path.getsize(output_path)
                    }
            else:
                raise FileNotFoundError(f"Generated audio file not found at {output_path}")
        else:
            # Command failed
            error_msg = f"ACE-Step CLI failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            elif result.stdout:
                error_msg += f": {result.stdout}"
            
            raise RuntimeError(error_msg)
            
    except subprocess.TimeoutExpired:
        logger.error("ACE-Step generation timed out")
        raise RuntimeError("Music generation timed out after 15 minutes")
        
    except Exception as e:
        logger.error(f"ACE-Step generation failed: {e}")
        raise RuntimeError(f"Music generation failed: {str(e)}")


if __name__ == "__main__":
    # Test the wrapper
    import sys
    
    if len(sys.argv) > 1:
        test_prompt = sys.argv[1]
    else:
        test_prompt = "Upbeat electronic music with synthesizers"
    
    print(f"Testing ACE-Step CLI wrapper with prompt: '{test_prompt}'")
    
    try:
        result = generate_music(test_prompt, duration=10)
        print(f"Success: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()