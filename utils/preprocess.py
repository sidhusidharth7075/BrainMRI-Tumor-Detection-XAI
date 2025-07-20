"""
Image Preprocessing Module
Handles image preprocessing for the CNN model
"""

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Preprocess image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (width, height)
    
    Returns:
        tuple: (preprocessed_image_array, original_image)
    """
    try:
        # Load image using PIL
        original_img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Resize image to target size
        resized_img = original_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = keras_image.img_to_array(resized_img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_img
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def validate_image(image_path):
    """
    Validate image file
    
    Args:
        image_path: Path to the image file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Check if image can be opened
            img.verify()
        
        # Try to load again for format check
        with Image.open(image_path) as img:
            # Check if it's a valid image format
            if img.format not in ['JPEG', 'JPG', 'PNG']:
                return False
            
            # Check image dimensions
            width, height = img.size
            if width < 50 or height < 50:
                return False
            
            return True
    
    except Exception:
        return False

def get_image_info(image_path):
    """
    Get basic information about the image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        dict: Image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height
            }
    except Exception as e:
        raise Exception(f"Error getting image info: {str(e)}")

def enhance_image(image_path, output_path=None):
    """
    Enhance image for better visualization
    
    Args:
        image_path: Path to the input image
        output_path: Path to save enhanced image (optional)
    
    Returns:
        PIL.Image: Enhanced image
    """
    try:
        from PIL import ImageEnhance
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced_img)
            enhanced_img = enhancer.enhance(1.1)
            
            # Save if output path provided
            if output_path:
                enhanced_img.save(output_path, quality=95)
            
            return enhanced_img
    
    except Exception as e:
        raise Exception(f"Error enhancing image: {str(e)}") 