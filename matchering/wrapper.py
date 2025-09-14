"""
Matchering Audio Mastering Wrapper

Wrapper for Matchering 2.0 for automatic audio mastering.
Repository: https://github.com/matchering/matchering
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class MatcheringProcessor:
    """Matchering audio mastering processor."""
    
    def __init__(self):
        self.default_reference = None
        self.mastering_presets = {
            'gentle': {
                'loudness_target': -16.0,
                'true_peak_target': -1.0,
                'dynamics_factor': 0.8
            },
            'medium': {
                'loudness_target': -14.0,
                'true_peak_target': -0.5,
                'dynamics_factor': 0.6
            },
            'aggressive': {
                'loudness_target': -12.0,
                'true_peak_target': -0.1,
                'dynamics_factor': 0.4
            },
            'streaming': {
                'loudness_target': -16.0,
                'true_peak_target': -1.0,
                'dynamics_factor': 0.7
            },
            'cd': {
                'loudness_target': -12.0,
                'true_peak_target': -0.3,
                'dynamics_factor': 0.5
            }
        }
    
    def master_with_matchering(self, input_path: str, output_path: str,
                              reference_path: Optional[str] = None,
                              loudness: float = -16.0, **kwargs) -> Dict[str, Any]:
        """
        Master audio using Matchering library.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save mastered audio
            reference_path: Optional reference track for matching
            loudness: Target loudness in LUFS
            **kwargs: Additional mastering parameters
            
        Returns:
            Dictionary with mastering metadata
        """
        try:
            # Import Matchering
            import matchering as mg
            
            logger.info(f"Mastering audio: target loudness = {loudness} LUFS")
            
            if reference_path and os.path.exists(reference_path):
                # Master with reference using Matchering's correct API
                logger.info(f"Using reference track: {reference_path}")
                
                # Apply matchering with reference - correct API call
                mg.process(
                    target=input_path,
                    reference=reference_path,
                    results=[
                        mg.pcm16(output_path),  # 16-bit PCM output to file
                        mg.pcm24(output_path.replace('.wav', '_24bit.wav')),  # Optional 24-bit
                    ]
                )
                
                metadata = {
                    'method': 'reference_matching',
                    'reference_used': True,
                    'reference_path': reference_path
                }
                
            else:
                # Master without reference (use basic mastering)
                logger.info("Mastering without reference track")
                
                # Load target audio for basic processing
                target_audio, target_sr = sf.read(input_path)
                
                # Apply basic mastering
                mastered_audio = self._apply_basic_mastering_chain(
                    target_audio, target_sr, loudness, **kwargs
                )
                
                # Save mastered audio
                sf.write(output_path, mastered_audio, target_sr)
                
                metadata = {
                    'method': 'preset_mastering',
                    'reference_used': False,
                    'target_loudness': loudness
                }
            
            # Calculate metadata
            file_size = os.path.getsize(output_path)
            
            # Get duration from the output file
            try:
                info = sf.info(output_path)
                duration = info.duration
                sample_rate = info.samplerate
                channels = info.channels
            except:
                # Fallback if file info can't be read
                duration = 0.0
                sample_rate = 44100
                channels = 2
            
            metadata.update({
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels
            })
            
            logger.info(f"Mastering completed: {file_size} bytes")
            return metadata
            
        except ImportError as e:
            logger.error(f"Matchering import failed: {str(e)}")
            # Fallback to basic mastering
            return self._master_fallback(input_path, output_path, loudness, **kwargs)
        except Exception as e:
            logger.error(f"Matchering failed: {str(e)}")
            # Fallback to basic mastering
            return self._master_fallback(input_path, output_path, loudness, **kwargs)
    
    def _master_without_reference(self, audio: np.ndarray, sample_rate: int,
                                 loudness: float = -16.0, **kwargs) -> Dict[str, np.ndarray]:
        """
        Basic mastering without reference track.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            loudness: Target loudness in LUFS
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with mastered audio
        """
        try:
            import matchering as mg
            
            # Create a synthetic reference or use preset
            preset = kwargs.get('preset', 'medium')
            if preset in self.mastering_presets:
                preset_params = self.mastering_presets[preset]
                loudness = preset_params['loudness_target']
            
            # Apply basic processing chain
            processed_audio = self._apply_basic_mastering_chain(
                audio, sample_rate, loudness, **kwargs
            )
            
            return {mg.pcm16: processed_audio}
            
        except ImportError:
            # Pure NumPy fallback
            processed_audio = self._apply_basic_mastering_chain(
                audio, sample_rate, loudness, **kwargs
            )
            return {'pcm16': processed_audio}
    
    def _apply_basic_mastering_chain(self, audio: np.ndarray, sample_rate: int,
                                   loudness: float = -16.0, **kwargs) -> np.ndarray:
        """
        Apply basic mastering processing chain.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            loudness: Target loudness in LUFS
            **kwargs: Additional parameters
            
        Returns:
            Processed audio array
        """
        # Ensure stereo
        if len(audio.shape) == 1:
            audio = np.column_stack([audio, audio])
        
        # Basic EQ (high-pass filter)
        audio = self._apply_highpass_filter(audio, sample_rate, cutoff=40)
        
        # Compression
        compression_ratio = kwargs.get('compression_ratio', 3.0)
        threshold = kwargs.get('compression_threshold', -20.0)
        audio = self._apply_compression(audio, threshold, compression_ratio)
        
        # Limiting
        audio = self._apply_limiter(audio, threshold=-0.1)
        
        # Loudness normalization
        audio = self._normalize_loudness(audio, target_lufs=loudness)
        
        return audio
    
    def _apply_highpass_filter(self, audio: np.ndarray, sample_rate: int, cutoff: float = 40) -> np.ndarray:
        """Apply simple high-pass filter."""
        try:
            from scipy import signal
            
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            b, a = signal.butter(2, normalized_cutoff, btype='high')
            
            if len(audio.shape) == 2:
                filtered = np.zeros_like(audio)
                for ch in range(audio.shape[1]):
                    filtered[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            else:
                filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except ImportError:
            logger.warning("SciPy not available, skipping high-pass filter")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, threshold: float = -20.0, ratio: float = 3.0) -> np.ndarray:
        """Apply basic compression."""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Apply compression
        compressed_db = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        gain = 10 ** ((compressed_db - audio_db) / 20)
        compressed = audio * gain
        
        return compressed
    
    def _apply_limiter(self, audio: np.ndarray, threshold: float = -0.1) -> np.ndarray:
        """Apply basic limiter."""
        linear_threshold = 10 ** (threshold / 20)
        
        # Simple hard limiting
        limited = np.clip(audio, -linear_threshold, linear_threshold)
        
        return limited
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
        """Basic loudness normalization."""
        # Simple RMS-based normalization (approximation of LUFS)
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0:
            # Convert target LUFS to approximate linear scale
            target_rms = 10 ** (target_lufs / 20)
            gain = target_rms / rms
            
            # Apply gain with safety limiting
            normalized = audio * min(gain, 10.0)  # Max 20dB gain
            
            # Final safety clip
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
        
        return audio
    
    def _master_fallback(self, input_path: str, output_path: str, loudness: float = -16.0, **kwargs) -> Dict[str, Any]:
        """
        Fallback mastering using basic audio processing.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            loudness: Target loudness in LUFS
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with mastering metadata
        """
        try:
            logger.info("Using fallback mastering")
            
            # Load audio
            audio, sample_rate = sf.read(input_path)
            
            # Apply basic mastering
            mastered_audio = self._apply_basic_mastering_chain(
                audio, sample_rate, loudness, **kwargs
            )
            
            # Save audio
            sf.write(output_path, mastered_audio, sample_rate)
            
            file_size = os.path.getsize(output_path)
            duration = len(mastered_audio) / sample_rate
            
            return {
                'method': 'fallback_mastering',
                'reference_used': False,
                'target_loudness': loudness,
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': mastered_audio.shape[1] if len(mastered_audio.shape) > 1 else 1
            }
            
        except Exception as e:
            logger.error(f"Fallback mastering failed: {str(e)}")
            raise


# Global processor instance
_processor = None


def get_processor() -> MatcheringProcessor:
    """Get or create global processor instance."""
    global _processor
    if _processor is None:
        _processor = MatcheringProcessor()
    return _processor


def master_track(input_path: str, output_path: str, reference_path: Optional[str] = None,
                loudness: float = -16.0, preset: str = 'medium', **kwargs) -> Dict[str, Any]:
    """
    Master audio track using Matchering or fallback processing.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save mastered audio
        reference_path: Optional reference track for matching
        loudness: Target loudness in LUFS
        preset: Mastering preset ('gentle', 'medium', 'aggressive', 'streaming', 'cd')
        **kwargs: Additional mastering parameters
        
    Returns:
        Dictionary with mastering metadata
    """
    try:
        processor = get_processor()
        
        # Validate input
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Apply preset if specified
        if preset in processor.mastering_presets:
            preset_params = processor.mastering_presets[preset]
            loudness = preset_params['loudness_target']
            kwargs.update({
                'compression_ratio': 1.0 / preset_params['dynamics_factor'] + 1.0,
                'compression_threshold': -20.0 + (preset_params['dynamics_factor'] * 10)
            })
        
        # Master audio
        metadata = processor.master_with_matchering(
            input_path, output_path, reference_path, loudness, preset=preset, **kwargs
        )
        
        metadata['preset'] = preset
        return metadata
        
    except Exception as e:
        logger.error(f"Audio mastering failed: {str(e)}")
        
        # Copy input as fallback
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            
            return {
                'method': 'copy_fallback',
                'reference_used': False,
                'target_loudness': loudness,
                'preset': preset,
                'file_size': os.path.getsize(output_path),
                'error': str(e)
            }
        except:
            raise


def get_mastering_presets() -> Dict[str, Dict[str, float]]:
    """Get available mastering presets."""
    processor = get_processor()
    return processor.mastering_presets


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Master audio with Matchering")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--output", default="mastered.wav", help="Output file path")
    parser.add_argument("--reference", help="Reference track for matching")
    parser.add_argument("--loudness", type=float, default=-16.0, help="Target loudness in LUFS")
    parser.add_argument("--preset", choices=['gentle', 'medium', 'aggressive', 'streaming', 'cd'],
                       default='medium', help="Mastering preset")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        presets = get_mastering_presets()
        print("Available presets:")
        for name, params in presets.items():
            print(f"  {name}: {params}")
    else:
        metadata = master_track(
            args.input,
            args.output,
            args.reference,
            args.loudness,
            args.preset
        )
        
        print(f"Mastering result: {metadata}")
