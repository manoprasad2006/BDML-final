# 🐳 Docker Deployment for Gesture-Based YouTube Control

This guide will help you deploy the gesture-based YouTube control system using Docker.

## 📋 Prerequisites

- Docker and Docker Compose installed
- Camera connected to your system
- At least 4GB RAM available

## 🚀 Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 2. Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

### 3. Using the Application

1. Click "Start Camera" to begin gesture recognition
2. Position your hands in front of the camera
3. Make gestures to control YouTube:
   - **Right Hand**: Volume control, mouse movement, play/pause
   - **Left Hand**: Navigation (forward/backward), fullscreen, captions

## 🛠️ Manual Docker Build

If you prefer to build manually:

```bash
# Build the Docker image
docker build -t gesture-control .

# Run the container
docker run -p 5000:5000 \
  --device /dev/video0:/dev/video0 \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  gesture-control
```

## 📁 Project Structure

```
gesture_based_youtube_control/
├── backend/
│   ├── app.py              # Flask backend API
│   └── templates/
│       └── index.html      # Web frontend
├── models/
│   ├── model.pth           # Trained gesture model
│   └── shape_predictor_68_face_landmarks.dat
├── data/
│   ├── gestures.csv        # Training data
│   └── label.csv           # Gesture labels
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── requirements-docker.txt # Python dependencies
└── start.sh               # Startup script
```

## 🔧 Configuration

### Environment Variables

You can customize the application using environment variables:

```bash
# In docker-compose.yml
environment:
  - FLASK_ENV=production
  - PYTHONPATH=/app
  - CONFIDENCE_THRESHOLD=0.9
```

### Camera Settings

The application uses the default camera (`/dev/video0`). If you have multiple cameras, you can specify a different one:

```yaml
devices:
  - /dev/video1:/dev/video0  # Use second camera
```

## 🐛 Troubleshooting

### Camera Not Working

1. **Check camera permissions:**
   ```bash
   ls -la /dev/video*
   ```

2. **Add user to video group:**
   ```bash
   sudo usermod -a -G video $USER
   ```

3. **Restart Docker service:**
   ```bash
   sudo systemctl restart docker
   ```

### Port Already in Use

If port 5000 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # Use port 8080 instead
```

### Model Files Missing

Make sure all model files are present:
- `models/model.pth`
- `models/shape_predictor_68_face_landmarks.dat`
- `data/gestures.csv`
- `data/label.csv`

## 📊 Monitoring

### View Logs

```bash
# View application logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f gesture-control
```

### Container Status

```bash
# Check running containers
docker-compose ps

# Check container resources
docker stats
```

## 🔄 Updates

To update the application:

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build
```

## 🛑 Stopping the Application

```bash
# Stop and remove containers
docker-compose down

# Stop and remove everything (including volumes)
docker-compose down -v
```

## 🌐 Network Access

To access from other devices on your network:

1. Find your machine's IP address:
   ```bash
   ip addr show
   ```

2. Access from other devices:
   ```
   http://YOUR_IP_ADDRESS:5000
   ```

## 📱 Mobile Access

The web interface is responsive and works on mobile devices. You can access it from your phone or tablet on the same network.

## 🔒 Security Notes

- The application runs with privileged access for camera functionality
- Only use on trusted networks
- Consider adding authentication for production use

## 🆘 Support

If you encounter issues:

1. Check the logs: `docker-compose logs`
2. Verify camera access: `ls /dev/video*`
3. Ensure all model files are present
4. Check system resources: `docker stats`

## 🎯 Features

- **Real-time Gesture Recognition**: Uses MediaPipe and PyTorch
- **Left/Right Hand Differentiation**: Different controls for each hand
- **Web Interface**: Modern, responsive design
- **Docker Deployment**: Easy setup and deployment
- **Camera Integration**: Works with USB cameras
- **YouTube Control**: Volume, playback, navigation
