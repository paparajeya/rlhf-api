#!/bin/bash

# Development setup for RLHF project (without Docker)

set -e

echo "🚀 Setting up RLHF development environment..."

# Function to print colored output
print_status() {
    local message=$1
    local status=${2:-"INFO"}
    case $status in
        "SUCCESS") echo -e "\033[92m✓ $message\033[0m" ;;
        "ERROR") echo -e "\033[91m✗ $message\033[0m" ;;
        "WARNING") echo -e "\033[93m⚠ $message\033[0m" ;;
        *) echo -e "\033[94mℹ $message\033[0m" ;;
    esac
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_status "Python 3 is not installed. Please install Python 3 and try again." "ERROR"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_status "Node.js is not installed. Please install Node.js and try again." "ERROR"
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p uploads/datasets models logs backend/app/static

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# Database Configuration
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
EOF
    print_status "Created .env file" "SUCCESS"
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install fastapi uvicorn sqlalchemy asyncpg pydantic pydantic-settings python-multipart aiofiles redis celery

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..

print_status "Setup complete!" "SUCCESS"
echo ""
print_status "To start the services:" "INFO"
print_status "1. Start Redis (if not running): redis-server" "INFO"
print_status "2. Start backend: cd backend && python3 -m uvicorn app.main:app --reload" "INFO"
print_status "3. Start frontend: cd frontend && npm start" "INFO"
echo ""
print_status "Or use the quick start script: ./quick_start.sh" "INFO" 