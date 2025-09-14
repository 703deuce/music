"""
Model Manager for RunPod Persistent Volume Storage

Handles downloading, caching, and managing AI models on persistent storage
to reduce cold start times and bandwidth usage in serverless environments.
"""

import os
import logging
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AI models on RunPod persistent volume storage."""
    
    def __init__(self, volume_path: str = "/runpod-volume"):
        """
        Initialize ModelManager with persistent volume path.
        
        Args:
            volume_path: Path to RunPod persistent volume (default: /runpod-volume)
        """
        self.volume_path = Path(volume_path)
        self.models_path = self.volume_path / "models"
        self.cache_path = self.volume_path / "cache"
        self.metadata_file = self.volume_path / "model_metadata.json"
        
        # Create directory structure
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Model paths for each AI system
        self.ace_step_path = self.models_path / "ace-step"
        self.demucs_path = self.models_path / "demucs"
        self.sovits_path = self.models_path / "sovits"
        self.matchering_path = self.models_path / "matchering"
        
        # Create model-specific directories
        for path in [self.ace_step_path, self.demucs_path, self.sovits_path, self.matchering_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"ModelManager initialized with volume: {self.volume_path}")
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from persistent storage."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {
            "models": {},
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def _save_metadata(self):
        """Save model metadata to persistent storage."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def is_model_cached(self, model_name: str, model_type: str) -> bool:
        """
        Check if a model is already cached in persistent storage.
        
        Args:
            model_name: Name of the model (e.g., 'ACE-Step-v1-3.5B')
            model_type: Type of model ('ace_step', 'demucs', 'sovits', 'matchering')
            
        Returns:
            True if model is cached and valid
        """
        model_key = f"{model_type}_{model_name}"
        
        # Check metadata
        if model_key not in self.metadata["models"]:
            return False
        
        model_info = self.metadata["models"][model_key]
        model_path = Path(model_info["path"])
        
        # Check if path exists
        if not model_path.exists():
            logger.warning(f"Model path missing: {model_path}")
            return False
        
        # For directories, check if they contain expected files
        if model_path.is_dir():
            expected_files = model_info.get("expected_files", [])
            for expected_file in expected_files:
                if not (model_path / expected_file).exists():
                    logger.warning(f"Missing model file: {model_path / expected_file}")
                    return False
        
        logger.info(f"Model {model_name} found in cache: {model_path}")
        return True
    
    def get_model_path(self, model_name: str, model_type: str) -> Optional[Path]:
        """
        Get the path to a cached model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Path to model if cached, None otherwise
        """
        if not self.is_model_cached(model_name, model_type):
            return None
        
        model_key = f"{model_type}_{model_name}"
        return Path(self.metadata["models"][model_key]["path"])
    
    def download_ace_step_model(self, model_name: str = "ACE-Step-v1-3.5B") -> Path:
        """
        Download and cache ACE-Step model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Path to cached model
        """
        model_path = self.ace_step_path / model_name
        
        if self.is_model_cached(model_name, "ace_step"):
            return self.get_model_path(model_name, "ace_step")
        
        logger.info(f"Downloading ACE-Step model: {model_name}")
        
        try:
            # Use huggingface_hub to download
            from huggingface_hub import snapshot_download
            
            downloaded_path = snapshot_download(
                repo_id=f"ACE-Step/{model_name}",
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # Register in metadata
            self._register_model(
                model_name=model_name,
                model_type="ace_step",
                model_path=model_path,
                expected_files=["config.json", "pytorch_model.bin"],
                download_source=f"ACE-Step/{model_name}"
            )
            
            logger.info(f"ACE-Step model cached: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download ACE-Step model: {e}")
            # Fallback: try git clone
            try:
                subprocess.run([
                    "git", "clone", 
                    f"https://huggingface.co/ACE-Step/{model_name}",
                    str(model_path)
                ], check=True)
                
                self._register_model(
                    model_name=model_name,
                    model_type="ace_step", 
                    model_path=model_path,
                    expected_files=["config.json"],
                    download_source=f"git_clone_ACE-Step/{model_name}"
                )
                
                return model_path
            except Exception as git_error:
                logger.error(f"Git clone also failed: {git_error}")
                raise
    
    def download_demucs_models(self, models: List[str] = None) -> Dict[str, Path]:
        """
        Download and cache Demucs models.
        
        Args:
            models: List of model names to download (default: common models)
            
        Returns:
            Dictionary mapping model names to paths
        """
        if models is None:
            models = ["htdemucs", "htdemucs_ft", "mdx_extra"]
        
        cached_models = {}
        
        for model_name in models:
            if self.is_model_cached(model_name, "demucs"):
                cached_models[model_name] = self.get_model_path(model_name, "demucs")
                continue
            
            logger.info(f"Downloading Demucs model: {model_name}")
            
            try:
                # Import demucs and download model
                from demucs.pretrained import get_model
                
                # This will download to demucs default location first
                model = get_model(model_name)
                
                # Copy to our persistent storage
                import torch
                model_path = self.demucs_path / f"{model_name}.th"
                torch.save(model.state_dict(), model_path)
                
                self._register_model(
                    model_name=model_name,
                    model_type="demucs",
                    model_path=model_path,
                    expected_files=[f"{model_name}.th"],
                    download_source="demucs_pretrained"
                )
                
                cached_models[model_name] = model_path
                logger.info(f"Demucs model cached: {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to download Demucs model {model_name}: {e}")
        
        return cached_models
    
    def register_sovits_model(self, model_name: str, model_files: Dict[str, str]) -> Path:
        """
        Register a so-vits-svc model in persistent storage.
        
        Args:
            model_name: Name of the voice model
            model_files: Dictionary of file types to file paths
                        e.g., {"model": "model.pth", "config": "config.json"}
            
        Returns:
            Path to model directory
        """
        model_path = self.sovits_path / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files to persistent storage
        expected_files = []
        for file_type, file_path in model_files.items():
            if os.path.exists(file_path):
                dest_path = model_path / os.path.basename(file_path)
                shutil.copy2(file_path, dest_path)
                expected_files.append(os.path.basename(file_path))
                logger.info(f"Copied {file_type} file: {dest_path}")
        
        self._register_model(
            model_name=model_name,
            model_type="sovits",
            model_path=model_path,
            expected_files=expected_files,
            download_source="manual_registration"
        )
        
        return model_path
    
    def _register_model(self, model_name: str, model_type: str, model_path: Path,
                       expected_files: List[str], download_source: str):
        """Register a model in metadata."""
        model_key = f"{model_type}_{model_name}"
        
        self.metadata["models"][model_key] = {
            "name": model_name,
            "type": model_type,
            "path": str(model_path),
            "expected_files": expected_files,
            "download_source": download_source,
            "cached_at": datetime.now().isoformat(),
            "size_mb": self._get_directory_size(model_path)
        }
        
        self._save_metadata()
        logger.info(f"Registered model: {model_key}")
    
    def _get_directory_size(self, path: Path) -> float:
        """Get directory size in MB."""
        try:
            total_size = 0
            if path.is_file():
                total_size = path.stat().st_size
            elif path.is_dir():
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
            
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
        except Exception:
            return 0.0
    
    def list_cached_models(self) -> Dict[str, Dict]:
        """List all cached models with their metadata."""
        return self.metadata.get("models", {})
    
    def cleanup_old_models(self, days_old: int = 30) -> List[str]:
        """
        Clean up models older than specified days.
        
        Args:
            days_old: Remove models cached more than this many days ago
            
        Returns:
            List of removed model keys
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_models = []
        
        for model_key, model_info in list(self.metadata["models"].items()):
            try:
                cached_date = datetime.fromisoformat(model_info["cached_at"])
                if cached_date < cutoff_date:
                    model_path = Path(model_info["path"])
                    if model_path.exists():
                        if model_path.is_dir():
                            shutil.rmtree(model_path)
                        else:
                            model_path.unlink()
                    
                    del self.metadata["models"][model_key]
                    removed_models.append(model_key)
                    logger.info(f"Removed old model: {model_key}")
            except Exception as e:
                logger.warning(f"Failed to remove model {model_key}: {e}")
        
        if removed_models:
            self._save_metadata()
        
        return removed_models
    
    def get_storage_stats(self) -> Dict:
        """Get storage usage statistics."""
        total_size = 0
        model_count = 0
        
        for model_info in self.metadata.get("models", {}).values():
            total_size += model_info.get("size_mb", 0)
            model_count += 1
        
        # Get volume free space
        try:
            statvfs = os.statvfs(self.volume_path)
            free_space_mb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 * 1024)
            total_space_mb = (statvfs.f_frsize * statvfs.f_blocks) / (1024 * 1024)
        except Exception:
            free_space_mb = 0
            total_space_mb = 0
        
        return {
            "cached_models": model_count,
            "total_size_mb": round(total_size, 2),
            "free_space_mb": round(free_space_mb, 2),
            "total_space_mb": round(total_space_mb, 2),
            "volume_path": str(self.volume_path)
        }
    
    def warmup_models(self) -> Dict[str, bool]:
        """
        Download commonly used models for faster cold starts.
        
        Returns:
            Dictionary of model download results
        """
        results = {}
        
        logger.info("Starting model warmup...")
        
        # Download ACE-Step model
        try:
            self.download_ace_step_model()
            results["ace_step"] = True
        except Exception as e:
            logger.error(f"Failed to warmup ACE-Step: {e}")
            results["ace_step"] = False
        
        # Download common Demucs models
        try:
            self.download_demucs_models(["htdemucs"])
            results["demucs"] = True
        except Exception as e:
            logger.error(f"Failed to warmup Demucs: {e}")
            results["demucs"] = False
        
        logger.info(f"Model warmup completed: {results}")
        return results


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        volume_path = os.getenv('RUNPOD_VOLUME_PATH', '/runpod-volume')
        _model_manager = ModelManager(volume_path)
    return _model_manager


# CLI interface for model management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage models on RunPod persistent volume")
    parser.add_argument("command", choices=["warmup", "list", "cleanup", "stats"],
                       help="Command to execute")
    parser.add_argument("--days", type=int, default=30,
                       help="Days for cleanup command")
    
    args = parser.parse_args()
    
    manager = get_model_manager()
    
    if args.command == "warmup":
        results = manager.warmup_models()
        print(f"Warmup results: {results}")
    
    elif args.command == "list":
        models = manager.list_cached_models()
        print(f"Cached models ({len(models)}):")
        for key, info in models.items():
            print(f"  {key}: {info['size_mb']}MB ({info['cached_at']})")
    
    elif args.command == "cleanup":
        removed = manager.cleanup_old_models(args.days)
        print(f"Removed {len(removed)} old models")
    
    elif args.command == "stats":
        stats = manager.get_storage_stats()
        print(f"Storage stats: {json.dumps(stats, indent=2)}")
