# Gesture-Based YouTube Control

A real-time hand gesture recognition system that allows you to control YouTube playback using only hand gestures. This project uses computer vision and machine learning to detect hand gestures and translate them into YouTube keyboard controls.

## Features

- **Real-time Gesture Recognition**: Uses MediaPipe for hand detection and a custom PyTorch model for gesture classification
- **Left/Right Hand Differentiation**: Different controls for left and right hands
- **Web Interface**: Modern, responsive web interface with live camera feed
- **Multiple Gesture Controls**:
  - Right Hand: Volume control, mouse movement, play/pause
  - Left Hand: Navigation (forward/backward), fullscreen, captions
- **Mouse Control**: Natural mouse movement using hand position
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Technology Stack

- **Computer Vision**: MediaPipe, OpenCV
- **Machine Learning**: PyTorch
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Automation**: PyAutoGUI

## Requirements

- Python 3.8+
- Webcam/Camera
- Windows/Linux/macOS

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gesture_based_youtube_control
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are present:
- `models/model.pth` - Trained gesture model
- `data/label.csv` - Gesture labels

## Usage

### Web Interface

1. Start the Flask server:
```bash
python backend/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Click "Start Camera" to begin gesture recognition
4. Position your hands in front of the camera
5. Make gestures to control YouTube

### Command Line Interface

Run the main script directly:
```bash
python main.py
```

Press 'q' to quit.

## Supported Gestures

### Right Hand Controls
- **Play/Pause**: Play or pause video playback
- **Volume Up**: Increase YouTube volume
- **Volume Down**: Decrease YouTube volume
- **Move Mouse**: Control mouse cursor with hand position
- **Left/Right Click**: Click interactions

### Left Hand Controls
- **Play/Pause**: Play or pause video playback
- **Forward**: Skip forward in video
- **Backward**: Skip backward in video
- **Fullscreen**: Toggle fullscreen mode
- **Captions/Subtitles**: Toggle closed captions

## Project Structure

```
gesture_based_youtube_control/
├── backend/
│   ├── app.py              # Flask backend API
│   └── templates/
│       └── index.html      # Web frontend
├── models/
│   ├── model.pth           # Trained gesture model
│   └── model_architecture.py
├── data/
│   ├── gestures.csv        # Training data
│   └── label.csv           # Gesture labels
├── utils.py                # Utility functions
├── main.py                 # Command-line interface
└── requirements.txt        # Python dependencies
```

## Docker Deployment

See `README-Docker.md` for detailed Docker deployment instructions.

## Configuration

- **Confidence Threshold**: Adjust `CONF_THRESH` in `backend/app.py` or `main.py` to change gesture detection sensitivity
- **Camera Resolution**: Modify `WIDTH` and `HEIGHT` variables
- **Mouse Smoothing**: Adjust `SMOOTH_FACTOR` for mouse movement sensitivity

## Training Your Own Model

Use the included `train.ipynb` notebook to train custom gesture recognition models with your own data.

## Troubleshooting

### Camera Not Detected
- Ensure your camera is connected and permissions are granted
- Check if other applications are using the camera
- On Linux, verify camera access: `ls /dev/video*`

### Low Gesture Recognition Accuracy
- Adjust the confidence threshold
- Ensure good lighting conditions
- Keep hands visible and clearly positioned
- Retrain the model with your own data if needed

### Model Files Missing
- Ensure `models/model.pth` exists
- Download required model files
- Check that `data/label.csv` is present

## License

This project is open source and available for personal and educational use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

Built with MediaPipe, PyTorch, and OpenCV for computer vision and machine learning capabilities.

