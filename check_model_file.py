#!/usr/bin/env python3
"""
Script to check the model file directly
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def check_model_file():
    """Check if the model file exists and can be loaded"""
    print("🔍 Checking model file...")
    
    # Check possible paths
    possible_paths = [
        'model/best_model.h5',
        'best_model.h5',
        './model/best_model.h5',
        './best_model.h5'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model at: {path}")
            break
    
    if model_path is None:
        print("❌ Model file not found in any expected location")
        print("Expected locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
    
    # Try to load the model
    print("🔄 Attempting to load model...")
    try:
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
        
        # Print model info
        print(f"📊 Model information:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Number of layers: {len(model.layers)}")
        print(f"   - Total parameters: {model.count_params():,}")
        
        # Print first few layers
        print(f"📋 First 5 layers:")
        for i, layer in enumerate(model.layers[:5]):
            print(f"   {i+1}. {type(layer).__name__}: {layer.output_shape}")
        
        # Try to compile
        print("🔄 Compiling model...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting model file check...\n")
    success = check_model_file()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Model file check passed!")
    else:
        print("❌ Model file check failed!")
    print("="*50) 