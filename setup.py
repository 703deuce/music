#!/usr/bin/env python3
"""
Setup script for Music AI API Suite.
Creates .env file from template and installs dependencies.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def create_env_file():
    """Create .env file from template."""
    env_example = Path('env.example')
    env_file = Path('.env')
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("   Overwrite? (y/N): ").lower()
        if response != 'y':
            print("   Skipping .env creation")
            return False
    
    if not env_example.exists():
        print("❌ env.example file not found!")
        return False
    
    shutil.copy2(env_example, env_file)
    print("✅ Created .env file from template")
    print("   📝 Edit .env file with your actual API keys")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found!")
        print("   Install Docker from: https://docs.docker.com/get-docker/")
        return False

def main():
    """Main setup function."""
    print("🎵 Music AI API Suite - Setup")
    print("=" * 40)
    
    # Create .env file
    env_created = create_env_file()
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    # Check Docker
    docker_available = check_docker()
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"   .env file: {'✅' if env_created else '⚠️ '}")
    print(f"   Dependencies: {'✅' if deps_installed else '❌'}")
    print(f"   Docker: {'✅' if docker_available else '❌'}")
    
    if env_created:
        print("\n🔑 Next Steps:")
        print("1. Edit .env file with your API keys:")
        print("   - RUNPOD_AI_API_KEY (required)")
        print("   - AWS/GCP/Azure storage credentials (required)")
        print("   - DOCKER_USERNAME/DOCKER_PASSWORD (for deployment)")
        print("   - HUGGINGFACE_HUB_TOKEN (recommended)")
        print()
        print("2. Test your API keys:")
        print("   python test_api_keys.py")
        print()
        print("3. Deploy to RunPod:")
        print("   python deploy.py")
        print()
        print("4. Test your endpoint:")
        print("   python test_endpoint.py")
    
    success = deps_installed and docker_available
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
