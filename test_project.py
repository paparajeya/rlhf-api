#!/usr/bin/env python3
"""
Comprehensive test script for the RLHF project.
This script tests the library, API, and frontend components.
"""

import os
import sys
import subprocess
import requests
import time
import json
from pathlib import Path

def print_status(message, status="INFO"):
    """Print status message with color coding."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
    }
    color = colors.get(status, colors["INFO"])
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {message}")

def check_dependencies():
    """Check if required dependencies are installed."""
    print_status("Checking dependencies...", "INFO")
    
    required_packages = [
        "torch", "transformers", "fastapi", "uvicorn",
        "sqlalchemy", "redis", "celery", "pydantic"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"✓ {package}", "SUCCESS")
        except ImportError:
            print_status(f"✗ {package} - missing", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        return False
    
    return True

def test_rlhf_library():
    """Test the RLHF library functionality."""
    print_status("Testing RLHF library...", "INFO")
    
    try:
        # Test imports
        sys.path.append('rlhf_lib')
        from rlhf import RLHFTrainer, PPOConfig, DPOConfig
        from rlhf.models import GPT2Policy, GPT2Value
        from rlhf.data import PreferenceDataset
        from rlhf.algorithms import PPO, DPO, A2C
        
        print_status("✓ RLHF library imports successful", "SUCCESS")
        
        # Test configuration
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=4,
            max_grad_norm=1.0,
            target_kl=0.1
        )
        print_status("✓ Configuration creation successful", "SUCCESS")
        
        # Test model creation (without downloading)
        try:
            policy_model = GPT2Policy("gpt2")
            print_status("✓ Model creation successful", "SUCCESS")
        except Exception as e:
            print_status(f"⚠ Model creation failed (expected without internet): {e}", "WARNING")
        
        return True
        
    except Exception as e:
        print_status(f"✗ RLHF library test failed: {e}", "ERROR")
        return False

def test_backend_api():
    """Test the FastAPI backend."""
    print_status("Testing backend API...", "INFO")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print_status("✓ Backend API is running", "SUCCESS")
            return True
        else:
            print_status(f"✗ Backend API returned status {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("✗ Backend API is not running", "ERROR")
        print_status("Start the backend with: cd backend && uvicorn app.main:app --reload", "INFO")
        return False

def test_frontend():
    """Test the React frontend."""
    print_status("Testing frontend...", "INFO")
    
    # Check if frontend is running
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print_status("✓ Frontend is running", "SUCCESS")
            return True
        else:
            print_status(f"✗ Frontend returned status {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("✗ Frontend is not running", "ERROR")
        print_status("Start the frontend with: cd frontend && npm start", "INFO")
        return False

def test_docker():
    """Test Docker setup."""
    print_status("Testing Docker setup...", "INFO")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("✓ Docker is installed", "SUCCESS")
        else:
            print_status("✗ Docker is not installed", "ERROR")
            return False
    except FileNotFoundError:
        print_status("✗ Docker is not installed", "ERROR")
        return False
    
    # Check if docker-compose is available
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("✓ Docker Compose is available", "SUCCESS")
        else:
            print_status("✗ Docker Compose is not available", "ERROR")
            return False
    except FileNotFoundError:
        print_status("✗ Docker Compose is not available", "ERROR")
        return False
    
    return True

def check_project_structure():
    """Check if the project structure is correct."""
    print_status("Checking project structure...", "INFO")
    
    required_dirs = [
        "rlhf_lib",
        "backend",
        "frontend",
        "backend/app",
        "backend/app/api",
        "backend/app/core",
        "frontend/src",
        "frontend/src/components",
        "frontend/src/pages",
    ]
    
    required_files = [
        "README.md",
        "docker-compose.yml",
        "env.example",
        "rlhf_lib/setup.py",
        "rlhf_lib/rlhf/__init__.py",
        "backend/requirements.txt",
        "backend/app/main.py",
        "frontend/package.json",
        "frontend/src/App.tsx",
    ]
    
    missing_dirs = []
    missing_files = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
        else:
            print_status(f"✓ Directory: {directory}", "SUCCESS")
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print_status(f"✓ File: {file}", "SUCCESS")
    
    if missing_dirs:
        print_status(f"Missing directories: {', '.join(missing_dirs)}", "ERROR")
    
    if missing_files:
        print_status(f"Missing files: {', '.join(missing_files)}", "ERROR")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_installation():
    """Test the installation process."""
    print_status("Testing installation...", "INFO")
    
    # Test RLHF library installation
    try:
        os.chdir("rlhf_lib")
        result = subprocess.run([sys.executable, "setup.py", "develop"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_status("✓ RLHF library installation successful", "SUCCESS")
        else:
            print_status(f"✗ RLHF library installation failed: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        print_status(f"✗ RLHF library installation failed: {e}", "ERROR")
        return False
    finally:
        os.chdir("..")
    
    # Test backend dependencies
    try:
        os.chdir("backend")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_status("✓ Backend dependencies installation successful", "SUCCESS")
        else:
            print_status(f"✗ Backend dependencies installation failed: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        print_status(f"✗ Backend dependencies installation failed: {e}", "ERROR")
        return False
    finally:
        os.chdir("..")
    
    return True

def main():
    """Main test function."""
    print_status("Starting comprehensive project test...", "INFO")
    print("=" * 60)
    
    tests = [
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("RLHF Library", test_rlhf_library),
        ("Docker Setup", test_docker),
        ("Installation", test_installation),
        ("Backend API", test_backend_api),
        ("Frontend", test_frontend),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Test failed with exception: {e}", "ERROR")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print_status("TEST SUMMARY", "INFO")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print_status("🎉 All tests passed! The project is ready to use.", "SUCCESS")
        print_status("\nNext steps:", "INFO")
        print_status("1. Copy env.example to .env and configure your settings", "INFO")
        print_status("2. Start the services: docker-compose up -d", "INFO")
        print_status("3. Access the application at http://localhost:3000", "INFO")
    else:
        print_status("⚠ Some tests failed. Please check the errors above.", "WARNING")

if __name__ == "__main__":
    main() 