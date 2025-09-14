#!/usr/bin/env python3
"""
Example client for the Music AI API Suite.
Demonstrates how to call the different AI models.
"""

import json
import requests
import time
from typing import Dict, Any

class MusicAIClient:
    """Client for the Music AI API Suite."""
    
    def __init__(self, endpoint_url: str, api_key: str = None):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the API."""
        try:
            response = self.session.post(
                f"{self.endpoint_url}/",
                json=payload,
                timeout=300  # 5 minutes
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise
    
    def generate_music(self, prompt: str, duration: int = 60) -> Dict[str, Any]:
        """Generate music from text prompt using ACE-Step."""
        payload = {
            "task": "ace_step",
            "params": {
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"🎼 Generating music: '{prompt}' ({duration}s)")
        result = self._make_request(payload)
        print(f"✅ Music generated: {result.get('audio_url', 'No URL')}")
        return result
    
    def separate_stems(self, audio_url: str, model: str = "htdemucs", 
                      stems: list = None) -> Dict[str, Any]:
        """Separate audio into stems using Demucs."""
        if stems is None:
            stems = ["vocals", "drums", "bass", "other"]
        
        payload = {
            "task": "separate",
            "input_url": audio_url,
            "params": {
                "model": model,
                "stems": stems
            }
        }
        
        print(f"🎤 Separating stems from: {audio_url}")
        result = self._make_request(payload)
        print(f"✅ Stems separated: {list(result.get('stems', {}).keys())}")
        return result
    
    def clone_voice(self, vocal_url: str, target_voice: str, 
                   pitch_shift: float = 0.0) -> Dict[str, Any]:
        """Clone voice using so-vits-svc."""
        payload = {
            "task": "voice_clone",
            "input_url": vocal_url,
            "params": {
                "target_voice": target_voice,
                "pitch_shift": pitch_shift
            }
        }
        
        print(f"🎙️ Cloning voice: {target_voice} (pitch: {pitch_shift})")
        result = self._make_request(payload)
        print(f"✅ Voice cloned: {result.get('audio_url', 'No URL')}")
        return result
    
    def master_audio(self, audio_url: str, reference_url: str = None, 
                    loudness: float = -16.0, preset: str = "medium") -> Dict[str, Any]:
        """Master audio using Matchering."""
        payload = {
            "task": "master",
            "input_url": audio_url,
            "params": {
                "loudness": loudness,
                "preset": preset
            }
        }
        
        if reference_url:
            payload["params"]["reference_url"] = reference_url
        
        print(f"🎚️ Mastering audio with preset: {preset}")
        result = self._make_request(payload)
        print(f"✅ Audio mastered: {result.get('audio_url', 'No URL')}")
        return result
    
    def warmup_models(self, models: List[str] = None, demucs_models: List[str] = None) -> Dict[str, Any]:
        """Warm up models to cache them in persistent storage."""
        if models is None:
            models = ["ace_step", "demucs"]
        
        payload = {
            "task": "warmup",
            "params": {
                "models": models
            }
        }
        
        if demucs_models:
            payload["params"]["demucs_models"] = demucs_models
        
        print(f"🔥 Warming up models: {models}")
        result = self._make_request(payload)
        print(f"✅ Warmup completed: {result.get('results', {})}")
        return result
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for persistent volume."""
        payload = {"task": "storage_stats"}
        
        print("📊 Getting storage statistics...")
        result = self._make_request(payload)
        
        if 'stats' in result:
            stats = result['stats']
            print(f"✅ Storage: {stats['cached_models']} models, "
                  f"{stats['total_size_mb']}MB used, "
                  f"{stats['free_space_mb']}MB free")
        
        return result


def example_workflow():
    """Example workflow using all four AI models."""
    
    # Initialize client
    # Replace with your actual RunPod endpoint URL
    client = MusicAIClient(
        endpoint_url="https://your-runpod-endpoint.com",
        api_key="your-api-key"  # Optional
    )
    
    print("🎵 Music AI API Suite - Example Workflow")
    print("=" * 50)
    
    try:
        # Step 1: Generate music with ACE-Step
        print("\n1️⃣ Generating base music track...")
        music_result = client.generate_music(
            prompt="Energetic rock song with guitar and drums",
            duration=30
        )
        generated_music_url = music_result.get('audio_url')
        
        if not generated_music_url:
            print("❌ Failed to generate music")
            return
        
        # Step 2: Separate stems with Demucs
        print("\n2️⃣ Separating audio stems...")
        stems_result = client.separate_stems(
            audio_url=generated_music_url,
            model="htdemucs"
        )
        
        stems = stems_result.get('stems', {})
        vocals_url = stems.get('vocals')
        
        # Step 3: Clone voice (if vocals are available)
        if vocals_url:
            print("\n3️⃣ Cloning vocals...")
            try:
                cloned_result = client.clone_voice(
                    vocal_url=vocals_url,
                    target_voice="example_voice",  # Replace with actual voice model
                    pitch_shift=1.0
                )
                cloned_vocals_url = cloned_result.get('audio_url')
                print(f"Cloned vocals: {cloned_vocals_url}")
            except Exception as e:
                print(f"Voice cloning skipped: {e}")
        
        # Step 4: Master the final audio
        print("\n4️⃣ Mastering final audio...")
        mastered_result = client.master_audio(
            audio_url=generated_music_url,
            preset="medium",
            loudness=-14.0
        )
        
        final_url = mastered_result.get('audio_url')
        print(f"\n🎉 Final mastered track: {final_url}")
        
        # Summary
        print("\n" + "=" * 50)
        print("Workflow Complete! 🎊")
        print("Files created:")
        print(f"  • Original: {generated_music_url}")
        for stem_name, stem_url in stems.items():
            print(f"  • {stem_name.title()}: {stem_url}")
        print(f"  • Mastered: {final_url}")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")


def test_individual_features():
    """Test individual features."""
    
    client = MusicAIClient("https://your-runpod-endpoint.com")
    
    print("🧪 Testing Individual Features")
    print("=" * 40)
    
    # Test music generation
    print("\n🎼 Testing Music Generation...")
    try:
        result = client.generate_music("Calm ambient music", 15)
        print(f"Success: {result.get('metadata', {}).get('duration_actual', 0)}s generated")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test stem separation (requires audio URL)
    print("\n🎤 Testing Stem Separation...")
    test_audio_url = "https://example.com/test-song.wav"  # Replace with real URL
    try:
        result = client.separate_stems(test_audio_url)
        print(f"Success: {len(result.get('stems', {}))} stems separated")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test voice cloning (requires vocal URL and voice model)
    print("\n🎙️ Testing Voice Cloning...")
    test_vocal_url = "https://example.com/test-vocal.wav"  # Replace with real URL
    try:
        result = client.clone_voice(test_vocal_url, "test_voice")
        print(f"Success: Voice cloned to {result.get('audio_url', 'unknown')}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test audio mastering
    print("\n🎚️ Testing Audio Mastering...")
    try:
        result = client.master_audio(test_audio_url, preset="gentle")
        print(f"Success: Audio mastered with {result.get('metadata', {}).get('method', 'unknown')} method")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_individual_features()
    else:
        example_workflow()
