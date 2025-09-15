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
        # Build the ACE-Step CLI command
        device_id = 0 if torch.cuda.is_available() else -1
        bf16 = torch.cuda.is_available()  # Use bf16 only if CUDA available
        
        # Create a temporary Python script that uses ACE-Step programmatically
        # This is better than trying to automate the interactive CLI
        script_content = f'''
import os
import sys
import tempfile

# Set up environment
os.environ['CUDA_VISIBLE_DEVICES'] = '{0 if torch.cuda.is_available() else ""}'

try:
    # Import ACE-Step modules
    from acestep.models.text2music import Text2MusicModel
    from acestep.utils.audio_utils import save_audio
    import torch
    
    print("ACE-Step modules imported successfully")
    
    # Initialize model
    model = Text2MusicModel(
        checkpoint_path="{checkpoint_path}",
        device_id={device_id},
        bf16={str(bf16).lower()}
    )
    
    print("Model initialized successfully")
    
    # Generate music
    audio_data = model.generate(
        prompt="{prompt}",
        duration={duration}
    )
    
    print("Music generated successfully")
    
    # Save audio
    save_audio("{output_path}", audio_data, sample_rate=44100)
    
    print("SUCCESS: Audio saved to {output_path}")
    
except ImportError as e:
    print(f"IMPORT_ERROR: {{e}}")
    
    # Try alternative approach using CLI
    import subprocess
    import os
    
    try:
        # Use acestep CLI directly
        cmd = [
            "acestep",
            "--checkpoint_path", "{checkpoint_path}",
            "--device_id", "{device_id}",
            "--bf16", "{str(bf16).lower()}",
            "--prompt", "{prompt}",
            "--duration", "{duration}",
            "--output", "{output_path}"
        ]
        
        print(f"Running ACE-Step CLI: {{' '.join(cmd)}}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print("SUCCESS: ACE-Step CLI completed")
        else:
            print(f"CLI_ERROR: {{result.stderr}}")
            
    except Exception as cli_error:
        print(f"CLI_ERROR: {{cli_error}}")
        
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
        
        # Write the script to a temporary file
        script_path = os.path.join(tempfile.gettempdir(), f"ace_step_gen_{int(time.time())}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created generation script: {script_path}")
        
        # Execute the script
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes timeout for model download + generation
            cwd=tempfile.gettempdir()
        )
        
        # Clean up script
        try:
            os.remove(script_path)
        except:
            pass
        
        logger.info(f"ACE-Step script output: {result.stdout}")
        if result.stderr:
            logger.warning(f"ACE-Step script errors: {result.stderr}")
        
        # Check if generation was successful
        if result.returncode == 0 and "SUCCESS" in result.stdout:
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
            # Parse the error from the output
            error_msg = "ACE-Step generation failed"
            if "IMPORT_ERROR" in result.stdout:
                error_msg = "ACE-Step import failed - module not properly installed"
            elif "CLI_ERROR" in result.stdout:
                error_msg = "ACE-Step CLI execution failed"
            elif result.stderr:
                error_msg = f"Script error: {result.stderr}"
            
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