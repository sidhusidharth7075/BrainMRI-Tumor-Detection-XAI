"""
PDF Report Generator Module
Generates professional PDF reports for brain tumor classification results
"""

import os
import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import numpy as np

class BrainTumorReportGenerator:
    """Generate comprehensive PDF reports for brain tumor classification results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        # Disclaimer style
        self.disclaimer_style = ParagraphStyle(
            'CustomDisclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            textColor=colors.grey
        )
    
    def generate_report(self, patient_id, prediction_data, output_path):
        """
        Generate a comprehensive PDF report for brain tumor classification.
        
        Args:
            patient_id: Unique patient identifier
            prediction_data: Dictionary containing prediction results
            output_path: Path where to save the PDF report
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Add title page
        story.extend(self._create_title_page(patient_id, prediction_data))
        
        # Add executive summary
        story.extend(self._create_executive_summary(prediction_data))
        
        # Add detailed results
        story.extend(self._create_detailed_results(prediction_data))
        
        # Add model information
        story.extend(self._create_model_information())
        
        # Add disclaimer
        story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_title_page(self, patient_id, prediction_data):
        """Create the title page of the report."""
        elements = []
        
        # Title
        title = Paragraph("Brain Tumor Classification Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 30))
        
        # Patient information
        patient_info = [
            ["Patient ID:", patient_id],
            ["Report Date:", prediction_data.get('timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
            ["Predicted Class:", prediction_data.get('predicted_class', 'Unknown').title()],
            ["Confidence Score:", f"{prediction_data.get('max_confidence', 0):.2%}"]
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 30))
        
        # Add images if available
        if 'original_image' in prediction_data and os.path.exists(f"static/{prediction_data['original_image']}"):
            try:
                img = Image(f"static/{prediction_data['original_image']}", width=3*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
            except:
                pass
        
        return elements
    
    def _create_executive_summary(self, prediction_data):
        """Create the executive summary section."""
        elements = []
        
        # Section header
        summary_header = Paragraph("Executive Summary", self.subtitle_style)
        elements.append(summary_header)
        elements.append(Spacer(1, 12))
        
        # Summary text
        predicted_class = prediction_data.get('predicted_class', 'Unknown').title()
        confidence = prediction_data.get('max_confidence', 0)
        
        summary_text = f"""
        This report presents the results of an automated brain tumor classification analysis 
        performed using a deep learning model. The analysis was conducted on an MRI brain scan 
        and provides a comprehensive assessment of the detected abnormalities.
        
        <b>Key Findings:</b>
        • Predicted Tumor Type: {predicted_class}
        • Confidence Level: {confidence:.2%}
        • Analysis Method: Convolutional Neural Network (CNN) with Grad-CAM visualization
        • Model Accuracy: 94% (validation set)
        
        The Grad-CAM visualization highlights the regions of the brain scan that were most 
        influential in the model's decision-making process, providing transparency and 
        interpretability to the classification results.
        """
        
        summary_para = Paragraph(summary_text, self.normal_style)
        elements.append(summary_para)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_detailed_results(self, prediction_data):
        """Create the detailed results section with visualizations."""
        elements = []
        
        # Section header
        results_header = Paragraph("Detailed Analysis Results", self.subtitle_style)
        elements.append(results_header)
        elements.append(Spacer(1, 12))
        
        # Confidence scores table
        confidence_header = Paragraph("Confidence Scores by Class", self.header_style)
        elements.append(confidence_header)
        elements.append(Spacer(1, 6))
        
        class_names = prediction_data.get('class_names', ['glioma', 'meningioma', 'no tumor', 'pituitary'])
        confidence_scores = prediction_data.get('confidence_scores', [0, 0, 0, 0])
        
        confidence_data = [["Class", "Confidence Score", "Percentage"]]
        for i, (class_name, score) in enumerate(zip(class_names, confidence_scores)):
            confidence_data.append([
                class_name.title(),
                f"{score:.4f}",
                f"{score:.2%}"
            ])
        
        confidence_table = Table(confidence_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        confidence_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey)
        ]))
        
        elements.append(confidence_table)
        elements.append(Spacer(1, 20))
        
        # Add visualizations
        if all(key in prediction_data for key in ['original_image', 'heatmap_image', 'overlay_image']):
            viz_header = Paragraph("Visual Analysis", self.header_style)
            elements.append(viz_header)
            elements.append(Spacer(1, 6))
            
            # Create a table with three images
            image_paths = [
                f"static/{prediction_data['original_image']}",
                f"static/{prediction_data['heatmap_image']}",
                f"static/{prediction_data['overlay_image']}"
            ]
            
            image_labels = ["Original Image", "Grad-CAM Heatmap", "Overlay Visualization"]
            
            # Check if images exist and add them
            for i, (img_path, label) in enumerate(zip(image_paths, image_labels)):
                if os.path.exists(img_path):
                    try:
                        img = Image(img_path, width=2*inch, height=2*inch)
                        elements.append(img)
                        label_para = Paragraph(label, self.normal_style)
                        elements.append(label_para)
                        elements.append(Spacer(1, 10))
                    except Exception as e:
                        print(f"Error adding image {img_path}: {e}")
        
        return elements
    
    def _create_model_information(self):
        """Create the model information section."""
        elements = []
        
        # Section header
        model_header = Paragraph("Model Information", self.subtitle_style)
        elements.append(model_header)
        elements.append(Spacer(1, 12))
        
        # Model details
        model_info = [
            ["Architecture:", "Convolutional Neural Network (CNN)"],
            ["Framework:", "TensorFlow/Keras"],
            ["Input Size:", "150x150 pixels"],
            ["Number of Classes:", "4 (Glioma, Meningioma, No Tumor, Pituitary)"],
            ["Training Date:", "2024"],
            ["Validation Accuracy:", "94%"],
            ["Explainability Method:", "Grad-CAM (Gradient-weighted Class Activation Mapping)"],
            ["Model Version:", "1.0"]
        ]
        
        model_table = Table(model_info, colWidths=[2*inch, 3*inch])
        model_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(model_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_disclaimer(self):
        """Create the disclaimer section."""
        elements = []
        
        # Section header
        disclaimer_header = Paragraph("Important Disclaimer", self.header_style)
        elements.append(disclaimer_header)
        elements.append(Spacer(1, 6))
        
        disclaimer_text = """
        <b>Medical Disclaimer:</b> This report is generated by an automated artificial intelligence 
        system and is intended for research and educational purposes only. The results should not 
        be used as the sole basis for clinical decision-making.
        
        • This analysis is not a substitute for professional medical diagnosis
        • All results should be reviewed and validated by qualified medical professionals
        • The model has been trained on a specific dataset and may not generalize to all cases
        • Clinical correlation with patient history and other diagnostic tests is essential
        • The confidence scores indicate model certainty, not clinical certainty
        
        <b>Technical Disclaimer:</b> The Grad-CAM visualization shows regions that influenced the 
        model's decision but does not guarantee clinical significance. The model's performance 
        may vary depending on image quality, acquisition parameters, and patient-specific factors.
        
        For clinical use, please consult with qualified radiologists and medical professionals.
        """
        
        disclaimer_para = Paragraph(disclaimer_text, self.disclaimer_style)
        elements.append(disclaimer_para)
        elements.append(Spacer(1, 20))
        
        return elements

def generate_pdf_report(patient_id, prediction_data=None):
    """
    Generate a PDF report for the given patient ID.
    
    Args:
        patient_id: Unique patient identifier
        prediction_data: Optional prediction data (for demo purposes)
        
    Returns:
        str: Path to the generated PDF report
    """
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Generate output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"reports/brain_tumor_report_{patient_id}_{timestamp}.pdf"
    
    # If no prediction data provided, create sample data for demo
    if prediction_data is None:
        prediction_data = {
            'predicted_class': 'glioma',
            'confidence_scores': [0.85, 0.08, 0.04, 0.03],
            'class_names': ['glioma', 'meningioma', 'no tumor', 'pituitary'],
            'max_confidence': 0.85,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original_image': 'sample_original.png',
            'heatmap_image': 'sample_heatmap.png',
            'overlay_image': 'sample_overlay.png'
        }
    
    # Generate report
    generator = BrainTumorReportGenerator()
    generator.generate_report(patient_id, prediction_data, output_path)
    
    return output_path 