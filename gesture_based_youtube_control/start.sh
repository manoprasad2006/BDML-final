#!/bin/bash

# Start the gesture control application
echo "Starting Gesture-Based YouTube Control..."

# Check if camera is available
if [ ! -e /dev/video0 ]; then
    echo "Warning: Camera device not found. Make sure camera is connected."
fi

# Start the Flask application
python backend/app.py
