#!/usr/bin/env python3
"""
Setup script for the RLHF project.
This script sets up the environment and starts the services.
"""

import os
import sys
import subprocess
import time
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

def create_env_file():
    """Create .env file with default configuration."""
    env_content = """# Database Configuration
DATABASE_URL=sqlite:///./rlhf.db
DATABASE_TEST_URL=sqlite:///./rlhf_test.db

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Security Configuration
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Upload Configuration
UPLOAD_DIR=./uploads
MODEL_STORAGE_PATH=./models
MAX_FILE_SIZE=104857600

# Training Configuration
DEFAULT_LEARNING_RATE=0.00001
DEFAULT_EPOCHS=3
DEFAULT_ALGORITHM=ppo

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# External APIs (optional)
OPENAI_API_KEY=
HUGGINGFACE_TOKEN=
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print_status("✓ Created .env file", "SUCCESS")

def create_directories():
    """Create necessary directories."""
    directories = [
        "uploads",
        "uploads/datasets",
        "models",
        "logs",
        "backend/app/static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_status(f"✓ Created directory: {directory}", "SUCCESS")

def install_dependencies():
    """Install Python dependencies."""
    print_status("Installing Python dependencies...", "INFO")
    
    # Install backend dependencies
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "fastapi", "uvicorn", "sqlalchemy", "asyncpg", 
            "pydantic", "pydantic-settings", "python-multipart",
            "aiofiles", "redis", "celery"
        ], check=True)
        print_status("✓ Backend dependencies installed", "SUCCESS")
    except subprocess.CalledProcessError as e:
        print_status(f"✗ Failed to install backend dependencies: {e}", "ERROR")
        return False
    
    # Install frontend dependencies
    try:
        os.chdir("frontend")
        subprocess.run(["npm", "install"], check=True)
        os.chdir("..")
        print_status("✓ Frontend dependencies installed", "SUCCESS")
    except subprocess.CalledProcessError as e:
        print_status(f"✗ Failed to install frontend dependencies: {e}", "ERROR")
        return False
    
    return True

def start_services():
    """Start the services."""
    print_status("Starting services...", "INFO")
    
    # Start backend
    try:
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        print_status("✓ Backend started on http://localhost:8000", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Failed to start backend: {e}", "ERROR")
        return False
    
    # Start frontend
    try:
        os.chdir("frontend")
        frontend_process = subprocess.Popen(["npm", "start"])
        os.chdir("..")
        print_status("✓ Frontend started on http://localhost:3000", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Failed to start frontend: {e}", "ERROR")
        return False
    
    return True

def main():
    """Main setup function."""
    print_status("Setting up RLHF project...", "INFO")
    print("=" * 60)
    
    # Create .env file
    if not os.path.exists(".env"):
        create_env_file()
    else:
        print_status("✓ .env file already exists", "SUCCESS")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print_status("Setup failed due to dependency installation errors", "ERROR")
        return
    
    print("\n" + "=" * 60)
    print_status("SETUP COMPLETE", "SUCCESS")
    print("=" * 60)
    
    print_status("Next steps:", "INFO")
    print_status("1. Start the backend: cd backend && python -m uvicorn app.main:app --reload", "INFO")
    print_status("2. Start the frontend: cd frontend && npm start", "INFO")
    print_status("3. Access the application at http://localhost:3000", "INFO")
    print_status("4. API documentation at http://localhost:8000/docs", "INFO")
    
    # Ask if user wants to start services now
    response = input("\nWould you like to start the services now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        start_services()
        print_status("\nServices are starting...", "INFO")
        print_status("Backend: http://localhost:8000", "INFO")
        print_status("Frontend: http://localhost:3000", "INFO")
        print_status("API Docs: http://localhost:8000/docs", "INFO")
        print_status("\nPress Ctrl+C to stop the services", "INFO")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print_status("\nStopping services...", "INFO")
            print_status("Setup complete!", "SUCCESS")

if __name__ == "__main__":
    main() 