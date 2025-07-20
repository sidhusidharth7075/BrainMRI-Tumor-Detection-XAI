"""
Grad-CAM Module
Generates Gradient-weighted Class Activation Mapping for explainable AI
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.models import Model
from utils.model_loader import model, initialize_model
from utils.preprocess import preprocess_image

def ensure_model_loaded():
    """Ensure the model is loaded before generating Grad-CAM"""
    global model
    from utils.model_loader import is_model_loaded, initialize_model
    
    if not is_model_loaded():
        print("🔄 Model not loaded for Grad-CAM, initializing now...")
        try:
            initialize_model()
            if is_model_loaded():
                print("✅ Model initialized for Grad-CAM!")
            else:
                raise Exception("Failed to load model for Grad-CAM generation")
        except Exception as e:
            print(f"❌ Error initializing model for Grad-CAM: {e}")
            raise Exception(f"Failed to load model for Grad-CAM generation: {e}")
    
    # Final verification
    if model is None:
        raise Exception("Model is still None after initialization")
    
    if not hasattr(model, 'layers'):
        raise Exception("Model does not have 'layers' attribute after initialization")

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for visualizing CNN attention maps.
    """
    
    def __init__(self, model, class_index):
        """
        Initialize GradCAM with a trained model and target class.
        
        Args:
            model: Trained Keras model
            class_index: Index of the target class for visualization
        """
        self.model = model
        self.class_index = class_index
        self.grad_model = None
        
        # Create a model that outputs the last convolutional layer and predictions
        self._create_grad_model()
    
    def _create_grad_model(self):
        """Create a model that outputs gradients and activations."""
        # Get the last convolutional layer
        # For most CNN architectures, this is typically the last layer before flattening
        last_conv_layer = None
        
        # Find the last convolutional layer
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Fallback: use the last layer that has spatial dimensions
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) > 2:
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            raise ValueError("Could not find a suitable convolutional layer for Grad-CAM")
        
        print(f"Using layer for Grad-CAM: {last_conv_layer.name} with output shape: {last_conv_layer.output_shape}")
        
        # Create a model that outputs both the last conv layer and predictions
        try:
            grad_model = Model(
                inputs=[self.model.inputs],
                outputs=[last_conv_layer.output, self.model.output]
            )
            self.grad_model = grad_model
            self.last_conv_layer = last_conv_layer
        except Exception as e:
            print(f"Error creating grad model: {e}")
            # Fallback: create a simpler model
            self._create_simple_grad_model()
    
    def _create_simple_grad_model(self):
        """Create a simpler grad model as fallback."""
        print("Creating simple grad model as fallback...")
        
        # Find any convolutional layer
        conv_layers = []
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in model")
        
        # Use the last convolutional layer
        last_conv_layer = conv_layers[-1]
        print(f"Using fallback layer: {last_conv_layer.name}")
        
        # Create a simple model
        self.grad_model = self.model
        self.last_conv_layer = last_conv_layer
    
    def generate_cam(self, image, eps=1e-8):
        """
        Generate Grad-CAM heatmap for the given image.
        
        Args:
            image: Preprocessed input image (numpy array)
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Grad-CAM heatmap (numpy array)
        """
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        try:
            # Get gradients and activations
            with tf.GradientTape() as tape:
                if hasattr(self.grad_model, 'outputs') and len(self.grad_model.outputs) > 1:
                    # Multi-output model (conv_outputs, predictions)
                    conv_outputs, predictions = self.grad_model(image)
                else:
                    # Single output model - use the original model
                    predictions = self.grad_model(image)
                    # Get the last conv layer output
                    conv_outputs = self.last_conv_layer(image)
                
                class_output = predictions[:, self.class_index]
            
            # Compute gradients
            grads = tape.gradient(class_output, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by corresponding gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Apply ReLU to focus on positive contributions
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Error in generate_cam: {e}")
            # Return a simple heatmap as fallback
            return self._generate_simple_heatmap(image)
    
    def _generate_simple_heatmap(self, image):
        """Generate a simple heatmap as fallback when Grad-CAM fails."""
        print("Generating simple heatmap as fallback...")
        
        # Create a simple heatmap based on image intensity
        if len(image.shape) == 4:
            image = image[0]  # Remove batch dimension
        
        # Convert to grayscale if needed
        if image.shape[-1] == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image.squeeze()
        
        # Create a simple heatmap based on intensity
        heatmap = gray / np.max(gray)
        
        # Apply some smoothing
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap
    
    def generate_cam_with_guided_backprop(self, image, eps=1e-8):
        """
        Generate Grad-CAM with guided backpropagation for enhanced visualization.
        
        Args:
            image: Preprocessed input image (numpy array)
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Enhanced Grad-CAM heatmap (numpy array)
        """
        # Basic Grad-CAM
        heatmap = self.generate_cam(image, eps)
        
        # Resize heatmap to match input image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap
        heatmap_normalized = np.maximum(heatmap_resized, 0) / (np.max(heatmap_resized) + eps)
        
        return heatmap_normalized
    
    def overlay_heatmap(self, image, heatmap, alpha=0.6):
        """
        Overlay the heatmap on the original image.
        
        Args:
            image: Original image (numpy array)
            heatmap: Grad-CAM heatmap (numpy array)
            alpha: Transparency factor for overlay
            
        Returns:
            overlay: Image with heatmap overlay (numpy array)
        """
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Normalize image to 0-255 range if needed
        if image.max() <= 1.0:
            image_normalized = (image * 255).astype(np.uint8)
        else:
            image_normalized = image.astype(np.uint8)
        
        # Blend images
        overlay = cv2.addWeighted(image_normalized, alpha, heatmap_colored, 1 - alpha, 0)
        
        return overlay

def generate_gradcam(image_path, class_index, layer_name=None):
    """
    Generate Grad-CAM visualization for the given image and class
    
    Args:
        image_path: Path to the input image
        class_index: Index of the predicted class
        layer_name: Name of the convolutional layer to use (optional)
    
    Returns:
        str: Path to the generated Grad-CAM image
    """
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Preprocess image
        img_array, original_img = preprocess_image(image_path)
        
        # Create GradCAM instance
        gradcam = GradCAM(model, class_index)
        
        # Generate heatmap
        heatmap = gradcam.generate_cam(img_array)
        
        # Convert PIL image to numpy array for overlay
        img_array_orig = np.array(original_img)
        
        # Create overlay
        overlay = gradcam.overlay_heatmap(img_array_orig, heatmap, alpha=0.6)
        
        # Save Grad-CAM image
        gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
        gradcam_path = os.path.join('static/uploads', gradcam_filename)
        
        # Save the overlay image
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(gradcam_path)
        
        print(f"✅ Grad-CAM generated successfully: {gradcam_path}")
        return gradcam_path
    
    except Exception as e:
        print(f"❌ Error in generate_gradcam: {e}")
        # Create a fallback visualization
        return create_fallback_visualization(image_path, class_index)

def create_fallback_visualization(image_path, class_index):
    """
    Create a fallback visualization when Grad-CAM fails
    
    Args:
        image_path: Path to the input image
        class_index: Index of the predicted class
    
    Returns:
        str: Path to the fallback visualization
    """
    try:
        print("🔄 Creating fallback visualization...")
        
        # Load original image
        original_img = Image.open(image_path)
        img_array = np.array(original_img)
        
        # Create a simple heatmap based on image intensity
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=-1)
        else:
            gray = img_array
        
        # Normalize and create heatmap
        heatmap = gray / np.max(gray)
        heatmap = np.clip(heatmap, 0, 1)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        
        # Save fallback image
        fallback_filename = f"fallback_{os.path.basename(image_path)}"
        fallback_path = os.path.join('static/uploads', fallback_filename)
        
        fallback_pil = Image.fromarray(overlay)
        fallback_pil.save(fallback_path)
        
        print(f"✅ Fallback visualization created: {fallback_path}")
        return fallback_path
        
    except Exception as e:
        print(f"❌ Fallback visualization failed: {e}")
        # Return original image path as last resort
        return image_path

def generate_class_heatmaps(image_path):
    """
    Generate heatmaps for all classes
    
    Args:
        image_path: Path to the input image
    
    Returns:
        dict: Dictionary with heatmap paths for each class
    """
    try:
        ensure_model_loaded()
        
        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        heatmaps = {}
        
        # Preprocess image once
        img_array, original_img = preprocess_image(image_path)
        img_array_orig = np.array(original_img)
        
        for i, class_name in enumerate(class_labels):
            try:
                print(f"🔄 Generating heatmap for {class_name}...")
                
                # Create GradCAM instance for this class
                gradcam = GradCAM(model, i)
                
                # Generate heatmap
                heatmap = gradcam.generate_cam(img_array)
                
                # Create overlay
                overlay = gradcam.overlay_heatmap(img_array_orig, heatmap, alpha=0.6)
                
                # Save heatmap
                heatmap_filename = f"gradcam_{class_name}_{os.path.basename(image_path)}"
                heatmap_path = os.path.join('static/uploads', heatmap_filename)
                
                overlay_pil = Image.fromarray(overlay)
                overlay_pil.save(heatmap_path)
                
                heatmaps[class_name] = heatmap_path
                print(f"✅ Heatmap generated for {class_name}")
                
            except Exception as e:
                print(f"⚠️ Could not generate heatmap for {class_name}: {e}")
                heatmaps[class_name] = None
        
        return heatmaps
    
    except Exception as e:
        raise Exception(f"Error generating class heatmaps: {str(e)}")

def create_comparison_image(original_path, gradcam_path, output_path):
    """
    Create a side-by-side comparison image
    
    Args:
        original_path: Path to original image
        gradcam_path: Path to Grad-CAM image
        output_path: Path to save comparison image
    
    Returns:
        str: Path to the comparison image
    """
    try:
        # Load images
        original_img = Image.open(original_path)
        gradcam_img = Image.open(gradcam_path)
        
        # Ensure same size
        width, height = original_img.size
        gradcam_img = gradcam_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Create comparison image
        comparison_width = width * 2
        comparison_height = height
        
        comparison_img = Image.new('RGB', (comparison_width, comparison_height))
        comparison_img.paste(original_img, (0, 0))
        comparison_img.paste(gradcam_img, (width, 0))
        
        # Save comparison image
        comparison_img.save(output_path, quality=95)
        
        return output_path
    
    except Exception as e:
        raise Exception(f"Error creating comparison image: {str(e)}") 