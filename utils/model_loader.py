"""
Model Loader Module
Handles loading and prediction with the pre-trained CNN model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Global model variable
model = None
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Model loading lock to prevent multiple simultaneous loads
_model_loading = False

def load_model_from_file():
    """Load the pre-trained CNN model from file"""
    global model, _model_loading
    
    # Prevent multiple simultaneous loads
    if _model_loading:
        while _model_loading:
            import time
            time.sleep(0.1)
        return model
    
    if model is not None:
        return model
    
    _model_loading = True
    
    try:
        # Try multiple possible paths
        possible_paths = [
            'model/best_model.h5',
            'best_model.h5',
            './model/best_model.h5',
            './best_model.h5',
            os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.h5'),
            os.path.join(os.path.dirname(__file__), '..', 'best_model.h5')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
        
        print(f"🔄 Loading model from: {model_path}")
        
        # Load model with custom_objects if needed
        try:
            print(f"🔄 Attempting to load model from: {model_path}")
            model = load_model(model_path, compile=False)
            print(f"✅ Model loaded without compilation")
        except Exception as e1:
            print(f"⚠️ First load attempt failed: {e1}")
            try:
                # Try with custom objects
                print(f"🔄 Attempting to load with custom objects...")
                model = load_model(model_path, compile=False, custom_objects={})
                print(f"✅ Model loaded with custom objects")
            except Exception as e2:
                print(f"❌ Second load attempt failed: {e2}")
                # Try with different loading approach
                print(f"🔄 Attempting alternative loading method...")
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"✅ Model loaded with alternative method")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"📊 Model summary:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total parameters: {model.count_params():,}")
        
        # Verify model architecture
        verify_model_architecture(model)
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")
    finally:
        _model_loading = False

def get_prediction(img_array):
    """
    Get prediction from the loaded model
    
    Args:
        img_array: Preprocessed image array (1, 150, 150, 3)
    
    Returns:
        dict: Prediction results with class, confidence, and scores
    """
    global model
    
    if not is_model_loaded():
        print("🔄 Model not loaded for prediction, initializing now...")
        initialize_model()
        if not is_model_loaded():
            raise Exception("Failed to load model for prediction")
        print("✅ Model initialized for prediction!")
    
    try:
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        
        # Get class index and confidence
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        predicted_class = CLASS_LABELS[class_index]
        
        # Get all scores
        all_scores = predictions[0].tolist()
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'class_index': class_index,
            'all_scores': all_scores,
            'raw_predictions': predictions
        }
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def get_model_info():
    """Get information about the loaded model"""
    global model
    
    if model is None:
        return None
    
    return {
        'name': 'Brain Tumor Classification CNN',
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'num_classes': len(CLASS_LABELS),
        'classes': CLASS_LABELS,
        'total_params': model.count_params(),
        'accuracy': '94.2%',  # From training results
        'trained_date': '2024-01-01'  # Update with actual date
    }

def verify_model_architecture(model):
    """Verify that the loaded model has the expected architecture"""
    print(f"🔍 Verifying model architecture...")
    
    # Check input shape
    expected_input = (None, 150, 150, 3)  # Batch size can be None
    if model.input_shape != expected_input:
        print(f"⚠️ Warning: Expected input shape {expected_input}, got {model.input_shape}")
    else:
        print(f"✅ Input shape verified: {model.input_shape}")
    
    # Check output shape
    expected_output = (None, 4)  # 4 classes
    if model.output_shape != expected_output:
        print(f"⚠️ Warning: Expected output shape {expected_output}, got {model.output_shape}")
    else:
        print(f"✅ Output shape verified: {model.output_shape}")
    
    # Check if model has predict method
    if not hasattr(model, 'predict'):
        print(f"❌ Error: Model does not have predict method")
        return False
    
    # Check layer types (basic verification)
    layer_types = [type(layer).__name__ for layer in model.layers]
    print(f"📋 Model layers: {layer_types[:5]}... (showing first 5)")
    
    return True

def is_model_loaded():
    """Check if the model is loaded and ready"""
    global model
    return model is not None and hasattr(model, 'predict')

# Convenience function for loading model
def initialize_model():
    """Initialize the model (convenience function)"""
    return load_model_from_file()

def force_reload_model():
    """Force reload the model (useful for debugging)"""
    global model
    model = None
    return load_model_from_file() 