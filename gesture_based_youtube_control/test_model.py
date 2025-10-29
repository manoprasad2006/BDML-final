#!/usr/bin/env python3
"""
Test script to verify the gesture model is working correctly
"""

import sys
import os
import numpy as np
import torch
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

def test_model_loading():
    """Test if the model loads correctly"""
    print("🧪 Testing model loading...")
    
    try:
        from models.model_architecture import model
        
        # Load model
        model_path = 'models/model.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("✅ Model loaded successfully")
        
        # Test with dummy data
        dummy_features = np.random.randn(1, 13).astype(np.float32)
        dummy_tensor = torch.from_numpy(dummy_features)
        
        with torch.no_grad():
            output = model(dummy_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.softmax(output, dim=1)).item()
            
        print(f"✅ Model prediction test: {prediction}, confidence: {confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_labels():
    """Test if labels are loaded correctly"""
    print("\n🧪 Testing labels...")
    
    try:
        labels_path = 'data/label.csv'
        if not os.path.exists(labels_path):
            print(f"❌ Labels file not found: {labels_path}")
            return False
            
        labels = pd.read_csv(labels_path, header=None).values.flatten().tolist()
        print(f"✅ Labels loaded: {labels}")
        print(f"✅ Number of labels: {len(labels)}")
        return True
        
    except Exception as e:
        print(f"❌ Labels loading error: {e}")
        return False

def test_utils():
    """Test if utility functions work"""
    print("\n🧪 Testing utility functions...")
    
    try:
        from utils import pre_process_landmark, calc_distance, normalize_distances, get_all_distances
        
        # Test with dummy data
        dummy_points = [(0.5, 0.5) for _ in range(6)]
        preprocessed = pre_process_landmark(dummy_points)
        print(f"✅ pre_process_landmark works: shape {preprocessed.shape}")
        
        # Test distance calculation
        d0 = calc_distance(dummy_points[0], dummy_points[1])
        print(f"✅ calc_distance works: {d0}")
        
        # Test distance normalization
        distances = get_all_distances(dummy_points[:3])
        normalized = normalize_distances(d0, distances)
        print(f"✅ normalize_distances works: shape {normalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction pipeline"""
    print("\n🧪 Testing feature extraction...")
    
    try:
        from utils import pre_process_landmark, calc_distance, normalize_distances, get_all_distances
        
        # Simulate hand landmarks (21 points)
        dummy_landmarks = [(0.5 + 0.1 * np.random.randn(), 0.5 + 0.1 * np.random.randn()) for _ in range(21)]
        
        # Extract important points (same as in training)
        TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]
        important_points = [dummy_landmarks[i] for i in TRAINING_KEYPOINTS]
        
        # Preprocess landmarks
        preprocessed = pre_process_landmark(important_points)
        
        # Calculate distances
        d0 = calc_distance(dummy_landmarks[0], dummy_landmarks[5])
        pts_for_distances = [dummy_landmarks[i] for i in [4, 8, 12]]
        distances = normalize_distances(d0, get_all_distances(pts_for_distances))
        
        # Combine features
        features = np.concatenate([preprocessed, distances])
        
        print(f"✅ Feature extraction works:")
        print(f"   - Important points: {len(important_points)}")
        print(f"   - Preprocessed shape: {preprocessed.shape}")
        print(f"   - Distances shape: {distances.shape}")
        print(f"   - Final features shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Gesture Model")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_labels,
        test_utils,
        test_feature_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
