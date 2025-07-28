#!/bin/bash

# Quick start script for RLHF development

set -e

echo "🚀 Starting RLHF services..."

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

# Check if .env exists
if [ ! -f .env ]; then
    print_status "Running setup first..." "WARNING"
    ./dev_setup.sh
fi

# Create directories if they don't exist
mkdir -p uploads/datasets models logs backend/app/static

# Function to cleanup background processes
cleanup() {
    print_status "Stopping services..." "INFO"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend
print_status "Starting backend server..." "INFO"
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
print_status "Starting frontend server..." "INFO"
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait for services to be ready
print_status "Waiting for services to be ready..." "INFO"
sleep 5

# Check if services are running
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "Backend is running at http://localhost:8000" "SUCCESS"
else
    print_status "Backend failed to start" "ERROR"
    cleanup
    exit 1
fi

print_status "Frontend is starting at http://localhost:3000" "SUCCESS"
echo ""
print_status "Services are running!" "SUCCESS"
print_status "Frontend: http://localhost:3000" "INFO"
print_status "Backend API: http://localhost:8000" "INFO"
print_status "API Documentation: http://localhost:8000/docs" "INFO"
echo ""
print_status "Press Ctrl+C to stop all services" "INFO"

# Wait for user to stop
wait 