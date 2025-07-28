#!/bin/bash

# Build and run RLHF Docker containers

set -e

echo "🚀 Building and running RLHF Docker containers..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_status "Docker is not running. Please start Docker and try again." "ERROR"
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p uploads/datasets models logs

# Build and run containers
print_status "Building Docker containers..."
docker-compose build

print_status "Starting services..."
docker-compose up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check if services are running
print_status "Checking service status..."
if docker-compose ps | grep -q "Up"; then
    print_status "All services are running!" "SUCCESS"
    echo ""
    print_status "Access the application at:" "INFO"
    print_status "Frontend: http://localhost:3000" "INFO"
    print_status "Backend API: http://localhost:8000" "INFO"
    print_status "API Documentation: http://localhost:8000/docs" "INFO"
    print_status "Celery Flower: http://localhost:5555" "INFO"
    echo ""
    print_status "To view logs: docker-compose logs -f" "INFO"
    print_status "To stop services: docker-compose down" "INFO"
else
    print_status "Some services failed to start. Check logs with: docker-compose logs" "ERROR"
    exit 1
fi 