#!/usr/bin/env python3
"""
Test script to verify model loading functionality
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🧪 Testing model loading...")
    
    try:
        # Import the model loader
        from utils.model_loader import initialize_model, is_model_loaded, get_prediction, verify_model_architecture
        
        # Try to initialize the model
        print("🔄 Initializing model...")
        model = initialize_model()
        
        # Check if model is loaded
        if is_model_loaded():
            print("✅ Model loaded successfully!")
            
            # Verify architecture
            print("🔍 Verifying model architecture...")
            verify_model_architecture(model)
            
            # Test prediction with dummy data
            print("🧪 Testing prediction with dummy data...")
            dummy_input = np.random.random((1, 150, 150, 3))
            
            try:
                prediction = get_prediction(dummy_input)
                print("✅ Prediction successful!")
                print(f"📊 Prediction result: {prediction['class']} ({prediction['confidence']:.2%})")
                print(f"📊 All scores: {[f'{s:.3f}' for s in prediction['all_scores']]}")
                return True
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                return False
        else:
            print("❌ Model not loaded properly")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam():
    """Test if Grad-CAM can be generated"""
    print("\n🧪 Testing Grad-CAM generation...")
    
    try:
        from utils.grad_cam import generate_gradcam
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        dummy_image_path = "test_dummy_image.jpg"
        
        # Save dummy image
        from PIL import Image
        Image.fromarray(dummy_image).save(dummy_image_path)
        
        # Test Grad-CAM generation
        gradcam_path = generate_gradcam(dummy_image_path, class_index=0)
        
        # Clean up
        os.remove(dummy_image_path)
        if os.path.exists(gradcam_path):
            os.remove(gradcam_path)
        
        print("✅ Grad-CAM generation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting model loading tests...\n")
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test Grad-CAM
    gradcam_ok = test_gradcam()
    
    print("\n" + "="*50)
    print("📋 Test Results:")
    print(f"   Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"   Grad-CAM:      {'✅ PASS' if gradcam_ok else '❌ FAIL'}")
    
    if model_ok and gradcam_ok:
        print("\n🎉 All tests passed! The application should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    print("="*50) 