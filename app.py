"""
Brain Tumor Classification Web Application
A Flask-based web app for MRI brain scan tumor classification using CNN
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import base64
import io

# Import custom modules
from utils.preprocess import preprocess_image
from utils.grad_cam import generate_gradcam
from utils.pdf_generator import generate_pdf_report
from utils.model_loader import initialize_model, get_prediction, is_model_loaded

# Initialize model at module level
print("🧠 Loading brain tumor classification model...")
try:
    initialize_model()
    if is_model_loaded():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model loading may have failed, will retry on first request")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🔄 Will attempt to load model on first request")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'brain-tumor-classification-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add custom Jinja2 filters
@app.template_filter('enumerate')
def enumerate_filter(iterable):
    """Custom filter to enumerate items in Jinja2 templates"""
    return enumerate(iterable)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class labels
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Validate file type
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload JPG or PNG images only.', 'error')
            return redirect(request.url)
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Process image and get prediction
        prediction_result = process_image(filepath)
        
        # Store result in session for PDF generation
        session_data = {
            'image_path': filepath,
            'prediction': prediction_result,
            'timestamp': datetime.now().isoformat(),
            'patient_id': request.form.get('patient_id', f"PT{uuid.uuid4().hex[:8].upper()}"),
            'patient_name': request.form.get('patient_name', 'Anonymous'),
            'patient_email': request.form.get('patient_email', '')
        }
        
        # Store in session (in production, use Redis or database)
        app.config['SESSION_DATA'] = session_data
        
        return render_template('result.html', 
                             prediction=prediction_result, 
                             original_image=os.path.basename(filepath),
                             gradcam_image=os.path.basename(prediction_result['gradcam_path']),
                             patient_info={
                                 'name': session_data['patient_name'],
                                 'id': session_data['patient_id'],
                                 'email': session_data['patient_email']
                             },
                             session_data=session_data)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))

def ensure_model_loaded():
    """Ensure the model is loaded before making predictions"""
    from utils.model_loader import model, initialize_model, is_model_loaded
    if not is_model_loaded():
        print("🔄 Model not loaded, initializing now...")
        initialize_model()
        if is_model_loaded():
            print("✅ Model initialized successfully!")
        else:
            raise Exception("Failed to load model after initialization attempt")

def process_image(image_path):
    """Process uploaded image and return prediction results"""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Double-check model is loaded
        from utils.model_loader import is_model_loaded
        if not is_model_loaded():
            raise Exception("Model failed to load during image processing")
        
        # Load and preprocess image
        img_array, original_img = preprocess_image(image_path)
        
        # Get model prediction
        predictions = get_prediction(img_array)
        
        # Generate Grad-CAM
        try:
            gradcam_path = generate_gradcam(image_path, predictions['class_index'])
        except Exception as gradcam_error:
            print(f"⚠️ Grad-CAM generation failed: {gradcam_error}")
            # Create a fallback - copy original image as gradcam
            import shutil
            gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
            gradcam_path = os.path.join('static/uploads', gradcam_filename)
            shutil.copy2(image_path, gradcam_path)
            print(f"📋 Using original image as fallback for Grad-CAM")
        
        # Prepare result
        result = {
            'class': predictions['class'],
            'confidence': predictions['confidence'],
            'all_scores': predictions['all_scores'],
            'class_index': predictions['class_index'],
            'gradcam_path': gradcam_path,
            'original_path': image_path
        }
        
        return result
    
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/download_report')
def download_report():
    """Generate and download PDF report"""
    try:
        session_data = app.config.get('SESSION_DATA')
        if not session_data:
            flash('No prediction data available. Please upload an image first.', 'error')
            return redirect(url_for('index'))
        
        # Prepare prediction data for PDF generator
        prediction = session_data.get('prediction', {})
        patient_id = session_data.get('patient_id', 'unknown')
        
        prediction_data = {
            'predicted_class': prediction.get('class', 'unknown'),
            'confidence_scores': prediction.get('all_scores', [0, 0, 0, 0]),
            'class_names': ['glioma', 'meningioma', 'no tumor', 'pituitary'],
            'max_confidence': prediction.get('confidence', 0),
            'timestamp': session_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            'original_image': session_data.get('image_path', '').replace('static/', ''),
            'heatmap_image': prediction.get('gradcam_path', '').replace('static/', ''),
            'overlay_image': prediction.get('gradcam_path', '').replace('static/', '')  # Using same as heatmap for now
        }
        
        # Generate PDF report
        from utils.pdf_generator import generate_pdf_report
        pdf_path = generate_pdf_report(patient_id, prediction_data)
        
        # Send file for download
        return send_file(pdf_path, 
                        as_attachment=True, 
                        download_name=f"brain_tumor_report_{patient_id}.pdf",
                        mimetype='application/pdf')
    
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for AJAX predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process image
        result = process_image(filepath)
        
        # Convert images to base64 for frontend
        with open(filepath, 'rb') as img_file:
            original_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        with open(result['gradcam_path'], 'rb') as img_file:
            gradcam_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': result,
            'images': {
                'original': f"data:image/jpeg;base64,{original_b64}",
                'gradcam': f"data:image/jpeg;base64,{gradcam_b64}"
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': True
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Load model on startup
    print("🧠 Loading brain tumor classification model...")
    initialize_model()
    print("✅ Model loaded successfully!")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000) 