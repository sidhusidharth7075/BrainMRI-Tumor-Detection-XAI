#!/usr/bin/env python3
"""
Test script for Brain Tumor Detection Application
"""

import os
import sys
import requests
import json
from PIL import Image
import numpy as np

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🧪 Testing model loading...")
    
    try:
        from app import model, class_labels
        print(f"✅ Model loaded successfully!")
        print(f"✅ Class labels: {class_labels}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be started"""
    print("\n🧪 Testing Flask application...")
    
    try:
        from app import app
        print("✅ Flask app imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        'app.py',
        'best_model.h5',
        'requirements.txt',
        'templates/index.html',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def create_test_image():
    """Create a simple test image for testing"""
    print("\n🧪 Creating test image...")
    
    # Create a simple 150x150 test image
    test_image = Image.new('RGB', (150, 150), color='gray')
    test_image_path = 'test_image.jpg'
    test_image.save(test_image_path)
    print(f"✅ Test image created: {test_image_path}")
    return test_image_path

def test_image_processing():
    """Test image preprocessing functions"""
    print("\n🧪 Testing image processing...")
    
    try:
        from app import preprocess_image
        
        # Create test image
        test_image_path = create_test_image()
        
        # Test preprocessing
        img_array, original_img = preprocess_image(test_image_path)
        
        print(f"✅ Image preprocessing successful!")
        print(f"✅ Image array shape: {img_array.shape}")
        print(f"✅ Original image size: {original_img.size}")
        
        # Clean up
        os.remove(test_image_path)
        
        return True
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def test_pdf_generation():
    """Test PDF report generation"""
    print("\n🧪 Testing PDF generation...")
    
    try:
        from app import create_pdf_report
        import tempfile
        
        # Test data
        patient_data = {
            'name': 'Test Patient',
            'id': 'TEST001',
            'date': '2024-01-01'
        }
        
        prediction_result = {
            'class': 'notumor',
            'confidence': 0.95
        }
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
            test_img1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f2:
            test_img2 = f2.name
        
        # Create simple test images
        Image.new('RGB', (100, 100), color='white').save(test_img1)
        Image.new('RGB', (100, 100), color='black').save(test_img2)
        
        # Test PDF generation
        pdf_path = create_pdf_report(patient_data, prediction_result, test_img1, test_img2)
        
        if os.path.exists(pdf_path):
            print("✅ PDF generation successful!")
            os.remove(pdf_path)  # Clean up
        else:
            print("❌ PDF file not created")
            return False
        
        # Clean up test files
        os.remove(test_img1)
        os.remove(test_img2)
        
        return True
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Brain Tumor Detection - Application Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Flask App", test_flask_app),
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing),
        ("PDF Generation", test_pdf_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   python run.py")
        print("   or")
        print("   python app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 