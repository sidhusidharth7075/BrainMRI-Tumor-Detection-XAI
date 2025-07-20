#!/usr/bin/env python3
"""
Quick test to verify model loading works
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test of model loading"""
    print("🚀 Quick model loading test...")
    
    try:
        # Import and test model loading
        from utils.model_loader import initialize_model, is_model_loaded
        
        print("🔄 Initializing model...")
        model = initialize_model()
        
        if is_model_loaded():
            print("✅ Model loaded successfully!")
            
            # Test basic prediction
            import numpy as np
            dummy_input = np.random.random((1, 150, 150, 3))
            
            from utils.model_loader import get_prediction
            prediction = get_prediction(dummy_input)
            
            print(f"✅ Prediction test successful!")
            print(f"📊 Predicted class: {prediction['class']}")
            print(f"📊 Confidence: {prediction['confidence']:.2%}")
            
            return True
        else:
            print("❌ Model not loaded properly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    print("\n" + "="*40)
    if success:
        print("🎉 Quick test passed! Model should work in app.")
    else:
        print("❌ Quick test failed! Check model loading.")
    print("="*40) 