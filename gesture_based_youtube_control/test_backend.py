#!/usr/bin/env python3
"""
Test script to verify the backend can be imported and run
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Add current directory to path
        sys.path.append(os.getcwd())
        
        # Test basic imports
        print("  ✓ Testing basic imports...")
        import cv2
        import numpy as np
        import mediapipe as mp
        import torch
        import pandas as pd
        import flask
        import pyautogui
        
        # Test project imports
        print("  ✓ Testing project imports...")
        from models.model_architecture import model
        from utils import *
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if model and data files can be loaded"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Check if files exist
        model_path = 'models/model.pth'
        labels_path = 'data/label.csv'
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        if not os.path.exists(labels_path):
            print(f"❌ Labels file not found: {labels_path}")
            return False
        
        # Try to load model
        import torch
        model = torch.load(model_path)
        print(f"  ✓ Model loaded: {type(model)}")
        
        # Try to load labels
        import pandas as pd
        labels = pd.read_csv(labels_path, header=None).values.flatten().tolist()
        print(f"  ✓ Labels loaded: {len(labels)} labels")
        
        print("✅ Model loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def test_backend_import():
    """Test if backend can be imported"""
    print("\n🧪 Testing backend import...")
    
    try:
        # Change to backend directory
        backend_dir = os.path.join(os.getcwd(), 'backend')
        sys.path.insert(0, backend_dir)
        
        # Try to import the backend
        import app
        print("  ✓ Backend app imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Backend import error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Gesture Control Backend")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_imports,
        test_model_loading,
        test_backend_import
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
        print("🎉 All tests passed! Backend is ready to run.")
        print("\nTo start the backend:")
        print("  python backend/app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
