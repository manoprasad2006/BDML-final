#!/bin/bash

echo "🚀 Deploying Gesture-Based YouTube Control with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if camera is available
if [ ! -e /dev/video0 ]; then
    echo "⚠️  Warning: Camera device /dev/video0 not found."
    echo "   Make sure your camera is connected and accessible."
fi

# Check if model files exist
if [ ! -f "models/model.pth" ]; then
    echo "❌ Model file models/model.pth not found."
    echo "   Please ensure all model files are present."
    exit 1
fi

if [ ! -f "models/shape_predictor_68_face_landmarks.dat" ]; then
    echo "❌ Shape predictor model not found."
    echo "   Please ensure the dlib model file is present."
    exit 1
fi

echo "✅ Prerequisites check passed."

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting the application..."
docker-compose up -d

echo "⏳ Waiting for application to start..."
sleep 5

# Check if the application is running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Application is running!"
    echo ""
    echo "🌐 Access the application at: http://localhost:5000"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
    echo ""
    echo "📱 You can also access from other devices on your network:"
    echo "   http://$(hostname -I | awk '{print $1}'):5000"
else
    echo "❌ Failed to start the application."
    echo "📋 Check the logs: docker-compose logs"
    exit 1
fi
