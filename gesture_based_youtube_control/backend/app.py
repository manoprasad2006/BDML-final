import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import torch
import pandas as pd
from models.model_architecture import model
from utils import *
import threading
import time
import json
import base64
from io import BytesIO
from PIL import Image
from collections import deque

# Handle pyautogui import for headless environments
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    HAS_DISPLAY = True
except Exception as e:
    print(f"Warning: pyautogui not available in headless environment: {e}")
    HAS_DISPLAY = False
    # Mock pyautogui for headless environments
    class MockPyAutoGUI:
        @staticmethod
        def size():
            return (1920, 1080)  # Default screen size
        @staticmethod
        def moveTo(x, y):
            print(f"Mock: Move to ({x}, {y})")
        @staticmethod
        def click():
            print("Mock: Click")
        @staticmethod
        def press(key):
            print(f"Mock: Press {key}")
        @staticmethod
        def keyDown(key):
            print(f"Mock: Key down {key}")
        @staticmethod
        def keyUp(key):
            print(f"Mock: Key up {key}")
    
    pyautogui = MockPyAutoGUI()

app = Flask(__name__)
CORS(app)

# Global variables for gesture recognition
gesture_data = {
    'current_gesture': 'None',
    'confidence': 0.0,
    'hand_label': 'None',
    'is_active': False
}

# Initialize MediaPipe with MIRRORED output disabled for correct hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    static_image_mode=False,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model.pth')
    labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'label.csv')
    
    # Load model and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()  # Set to evaluation mode
    labels = pd.read_csv(labels_path, header=None).values.flatten().tolist()
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Labels loaded from: {labels_path}")
    print(f"✅ Model set to evaluation mode")
    print(f"✅ Available labels ({len(labels)}): {labels}")
except Exception as e:
    print(f"❌ Error loading model or labels: {e}")
    labels = []

# Camera settings
WIDTH = 1028//2
HEIGHT = 720//2
CONF_THRESH = 0.75  # Adjusted confidence threshold

# Mouse movement variables
SMOOTH_FACTOR = 6
PLOCX, PLOCY = 0, 0
CLOX, CLOXY = 0, 0
GEN_COUNTER = 0
GESTURE_HISTORY = deque(maxlen=10)  # Limit history size

# Gesture stabilization
GESTURE_BUFFER = deque(maxlen=5)  # Buffer to stabilize gesture detection
MIN_CONSISTENT_FRAMES = 3  # Minimum frames with same gesture to confirm

def get_corrected_hand_label(handedness_obj, frame_width):
    """
    Fix the left/right hand detection issue.
    MediaPipe returns mirrored labels when processing webcam feed.
    """
    # Get the label from MediaPipe
    mp_label = handedness_obj.classification[0].label
    confidence = handedness_obj.classification[0].score
    
    # MediaPipe's label is already correct for front-facing cameras
    # But if you're getting reversed results, uncomment the line below:
    # corrected_label = 'Left' if mp_label == 'Right' else 'Right'
    
    # For most webcam scenarios, MediaPipe is correct, so we use as-is:
    corrected_label = mp_label
    
    print(f"👋 Hand detected - MediaPipe: {mp_label}, Using: {corrected_label}, Confidence: {confidence:.2f}")
    
    return corrected_label, confidence

def stabilize_gesture(gesture, confidence):
    """
    Stabilize gesture detection by requiring consistent detection
    across multiple frames.
    """
    GESTURE_BUFFER.append((gesture, confidence))
    
    if len(GESTURE_BUFFER) < MIN_CONSISTENT_FRAMES:
        return None, 0.0
    
    # Count occurrences of each gesture in buffer
    gesture_counts = {}
    total_confidence = {}
    
    for g, c in GESTURE_BUFFER:
        if g not in gesture_counts:
            gesture_counts[g] = 0
            total_confidence[g] = 0.0
        gesture_counts[g] += 1
        total_confidence[g] += c
    
    # Find most common gesture
    most_common = max(gesture_counts.items(), key=lambda x: x[1])
    
    if most_common[1] >= MIN_CONSISTENT_FRAMES:
        avg_confidence = total_confidence[most_common[0]] / most_common[1]
        return most_common[0], avg_confidence
    
    return None, 0.0

def process_gesture(frame_rgb):
    """Process gesture recognition on the frame with enhanced debugging"""
    global gesture_data, GEN_COUNTER, GESTURE_HISTORY, PLOCX, PLOCY, CLOX, CLOXY
    
    try:
        # Flip the frame horizontally for natural mirror view
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_height, frame_width = frame_rgb.shape[:2]
        
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            print(f"\n{'='*60}")
            print(f"🖐️  Detected {len(results.multi_hand_landmarks)} hand(s)")
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Get CORRECTED hand label
                    handedness = results.multi_handedness[idx]
                    hand_label, hand_confidence = get_corrected_hand_label(handedness, frame_width)
                    
                    # Get landmarks coordinates
                    coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
                    
                    # DEBUG: Print detection info
                    print(f"\n--- Hand #{idx + 1} ---")
                    print(f"📍 Total landmarks detected: {len(coordinates_list)}")
                    print(f"👋 Hand: {hand_label} (confidence: {hand_confidence:.3f})")
                    
                    # Determine which keypoints to use (check your training configuration)
                    # Option 1: Only fingertips and wrist (6 points)
                    TRAINING_KEYPOINTS = [0, 4, 8, 12, 16, 20]  # wrist + 5 fingertips
                    
                    # Option 2: All landmarks (21 points) - uncomment if trained on all
                    # TRAINING_KEYPOINTS = list(range(21))
                    
                    important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]
                    print(f"🔑 Using keypoints: {TRAINING_KEYPOINTS} ({len(important_points)} points)")
                    
                    # Preprocess landmarks
                    preprocessed = pre_process_landmark(important_points)
                    print(f"📊 Preprocessed shape: {preprocessed.shape}")
                    
                    # Calculate distances
                    d0 = calc_distance(coordinates_list[0], coordinates_list[5])
                    pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
                    distances = normalize_distances(d0, get_all_distances(pts_for_distances))
                    print(f"📏 Distances shape: {distances.shape}")
                    
                    # Combine features
                    features = np.concatenate([preprocessed, distances])
                    print(f"🔍 Final features shape: {features.shape}")
                    print(f"🔍 Features - min: {features.min():.3f}, max: {features.max():.3f}")
                    print(f"🔍 Features - mean: {features.mean():.3f}, std: {features.std():.3f}")
                    
                    # Predict gesture with no_grad for evaluation
                    with torch.no_grad():
                        conf, pred = predict(features, model)
                    
                    gesture = labels[pred] if pred < len(labels) else "Unknown"
                    
                    print(f"🎯 Raw Prediction: {gesture} (index: {pred})")
                    print(f"🎯 Confidence: {conf:.3f} (threshold: {CONF_THRESH})")
                    
                    # Stabilize gesture detection
                    stable_gesture, stable_conf = stabilize_gesture(gesture, conf)
                    
                    if stable_gesture and stable_conf >= CONF_THRESH:
                        print(f"✅ STABLE GESTURE: {stable_gesture} (avg conf: {stable_conf:.3f})")
                        
                        # Update global gesture data
                        gesture_data.update({
                            'current_gesture': stable_gesture,
                            'confidence': float(stable_conf),
                            'hand_label': hand_label,
                            'is_active': True
                        })
                        
                        # Execute gesture commands
                        execute_gesture_command(stable_gesture, hand_label, coordinates_list)
                        
                        return True
                    elif conf >= CONF_THRESH:
                        print(f"⏳ Gesture detected but waiting for stability: {gesture}")
                    else:
                        print(f"⚠️  Low confidence: {conf:.3f} < {CONF_THRESH}")
                        
                except Exception as e:
                    print(f"❌ Error processing hand #{idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"{'='*60}\n")
        else:
            # No hands detected
            if results.multi_hand_landmarks is None:
                print("👁️  No hands detected in frame")
        
        # No valid gesture detected
        gesture_data.update({
            'current_gesture': 'None',
            'confidence': 0.0,
            'hand_label': 'None',
            'is_active': False
        })
        return False
        
    except Exception as e:
        print(f"❌ CRITICAL Error in process_gesture: {e}")
        import traceback
        traceback.print_exc()
        return False

def execute_gesture_command(gesture, hand_label, coordinates_list):
    """Execute the appropriate command based on gesture and hand"""
    global GEN_COUNTER, GESTURE_HISTORY, PLOCX, PLOCY, CLOX, CLOXY
    
    print(f"⚡ Executing: {gesture} with {hand_label} hand")
    
    # Track command history
    GESTURE_HISTORY.append(gesture)
    
    # Get previous gesture
    before_last = GESTURE_HISTORY[-2] if len(GESTURE_HISTORY) >= 2 else gesture
    
    # Mouse gestures (work with either hand)
    if gesture == 'Move_mouse':
        try:
            # Convert hand position to screen coordinates
            screen_size = pyautogui.size()
            screen_width, screen_height = screen_size
            
            # Use index finger tip (landmark 8) for mouse control
            hand_x, hand_y = coordinates_list[8]
            x = np.interp(hand_x, (0, WIDTH), (0, screen_width))
            y = np.interp(hand_y, (0, HEIGHT), (0, screen_height))
            
            # Smooth mouse movements
            CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
            CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
            pyautogui.moveTo(CLOX, CLOXY)
            PLOCX, PLOCY = CLOX, CLOXY
            print(f"🖱️  Mouse moved to ({int(CLOX)}, {int(CLOXY)})")
        except Exception as e:
            print(f"❌ Mouse movement error: {e}")
    
    elif gesture == 'Right_click' and before_last != 'Right_click':
        pyautogui.rightClick()
        print("🖱️  Right click executed")
    
    elif gesture == 'Left_click' and before_last != 'Left_click':
        pyautogui.click()
        print("🖱️  Left click executed")
    
    # Right hand controls volume and playback
    elif hand_label == 'Right':
        if gesture == 'Play_Pause' and before_last != 'Play_Pause':
            pyautogui.press('space')
            print("⏯️  Play/Pause (Right hand)")
        elif gesture == 'Vol_up_gen':
            pyautogui.press('volumeup')
            print("🔊 Volume up")
        elif gesture == 'Vol_down_gen':
            pyautogui.press('volumedown')
            print("🔉 Volume down")
        elif gesture == 'Vol_up_ytb':
            GEN_COUNTER += 1
            if GEN_COUNTER % 2 == 0:
                pyautogui.press('up')
                pyautogui.press('volumeup')
                print("🔊 YouTube volume up")
        elif gesture == 'Vol_down_ytb':
            GEN_COUNTER += 1
            if GEN_COUNTER % 2 == 0:
                pyautogui.press('down')
                pyautogui.press('volumedown')
                print("🔉 YouTube volume down")
    
    # Left hand controls navigation
    elif hand_label == 'Left':
        if gesture == 'Play_Pause' and before_last != 'Play_Pause':
            pyautogui.press('space')
            print("⏯️  Play/Pause (Left hand)")
        elif gesture == 'Forward':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                pyautogui.press('right')
                print("⏩ Forward")
        elif gesture == 'Backward':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                pyautogui.press('left')
                print("⏪ Backward")
        elif gesture == 'fullscreen' and before_last != 'fullscreen':
            pyautogui.press('f')
            print("⛶ Fullscreen toggle")
        elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
            pyautogui.press('c')
            print("📝 Captions/Subtitles toggle")
    
    # Reset counter on neutral gesture
    if gesture == 'Neutral':
        GEN_COUNTER = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/gesture', methods=['GET'])
def get_gesture():
    """Get current gesture data"""
    return jsonify(gesture_data)

@app.route('/api/gesture', methods=['POST'])
def process_frame():
    """Process a frame for gesture recognition"""
    try:
        # Get frame data from request
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process gesture
        gesture_detected = process_gesture(frame_rgb)
        
        return jsonify({
            'success': True,
            'gesture_detected': gesture_detected,
            'gesture_data': gesture_data
        })
        
    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'gesture_data': gesture_data,
        'model_loaded': len(labels) > 0,
        'labels': labels,
        'confidence_threshold': CONF_THRESH,
        'gesture_buffer_size': len(GESTURE_BUFFER),
        'gesture_history_size': len(GESTURE_HISTORY)
    })

@app.route('/api/debug')
def debug():
    """Debug endpoint to check model and system status"""
    import torch
    return jsonify({
        'model_loaded': len(labels) > 0,
        'labels_count': len(labels),
        'labels': labels,
        'confidence_threshold': CONF_THRESH,
        'current_gesture_data': gesture_data,
        'mediapipe_hands_initialized': hands is not None,
        'pytorch_version': torch.__version__,
        'gesture_buffer_size': len(GESTURE_BUFFER),
        'gesture_history_size': len(GESTURE_HISTORY),
        'smooth_factor': SMOOTH_FACTOR,
        'resolution': f"{WIDTH}x{HEIGHT}"
    })

@app.route('/api/clear_buffer', methods=['POST'])
def clear_buffer():
    """Clear gesture buffers (useful for testing)"""
    global GESTURE_BUFFER, GESTURE_HISTORY, GEN_COUNTER
    GESTURE_BUFFER.clear()
    GESTURE_HISTORY.clear()
    GEN_COUNTER = 0
    return jsonify({'success': True, 'message': 'Buffers cleared'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Gesture Recognition Server Starting")
    print("="*60)
    print(f"📊 Model: {'✅ Loaded' if len(labels) > 0 else '❌ Not loaded'}")
    print(f"🏷️  Labels: {len(labels)} gestures")
    print(f"🎯 Confidence Threshold: {CONF_THRESH}")
    print(f"📹 Resolution: {WIDTH}x{HEIGHT}")
    print(f"🔄 Gesture stabilization: {MIN_CONSISTENT_FRAMES} frames")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)