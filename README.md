<div align="center">

# 🧠 Brain Tumor Classification AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Project-red.svg?style=for-the-badge&logo=researchgate&logoColor=white)]()

> **🚀 Advanced AI-powered brain tumor classification system with explainable AI (XAI) capabilities**

A comprehensive web application that uses deep learning to classify brain tumors from MRI scans with real-time Grad-CAM visualizations for explainable AI.

[![GitHub stars](https://img.shields.io/github/stars/yourusername/brain-tumor-classification?style=social&label=Star)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/brain-tumor-classification?style=social&label=Fork)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/brain-tumor-classification?style=flat-square&color=blue)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/brain-tumor-classification?style=flat-square&color=green)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git/pulls)

</div>

## 📋 Table of Contents

- [Features](#-features)
- [Screenshots](#-screenshots)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## ✨ Features

<div align="center">

### 🎯 Core Functionality



| Feature | Description | Status |
|---------|-------------|--------|
| 🧠 **Multi-class Classification** | Detects 4 types of brain tumors | ✅ Active |
| ⚡ **Real-time Processing** | Fast inference with < 5 second processing time | ✅ Active |
| 🔍 **Explainable AI** | Grad-CAM visualizations for transparency | ✅ Active |
| 🌐 **Web Interface** | Modern, responsive UI with drag-and-drop upload | ✅ Active |
| 📄 **PDF Reports** | Comprehensive analysis reports with visualizations | ✅ Active |
</div >

### 🧠 Tumor Types Classified

<div align="center">

| Tumor Type | Description | Confidence Range | Training Samples |
|:----------:|:-----------:|:----------------:|:----------------:|
| 🧠 **Glioma** | Primary brain tumors from glial cells | 85-95% | ~1,428 |
| 🧬 **Meningioma** | Tumors from protective brain membranes | 80-90% | ~1,428 |
| 🎯 **Pituitary** | Tumors in pituitary gland | 85-95% | ~1,428 |
| ✅ **No Tumor** | Normal brain tissue | 90-98% | ~1,428 |

</div align="center">

### 🔧 Technical Features

<div align="center">

| Feature | Specification | Status |
|:-------:|:-------------:|:------:|
| 🎯 **High Accuracy** | 94% validation accuracy | ✅ Verified |
| 🔧 **Robust Preprocessing** | Automatic image normalization and resizing | ✅ Active |
| 🛡️ **Error Handling** | Graceful fallbacks for model failures | ✅ Active |
| 🔐 **Session Management** | Secure patient data handling | ✅ Active |
| 📱 **Responsive Design** | Works on desktop, tablet, and mobile | ✅ Active |

</div >

## 📸 Screenshots

<div align="center">

### 🖥️ Application Interface

#### 📤 Main Upload Interface
**Drag & drop MRI images for analysis**

![Main Interface](Main%20Interface.png)

#### 📊 Results Page
**AI predictions with confidence scores**

![Results Page](Results%20Page.png)



</div>

### 📄 Sample PDF Report

<div align="center">

**Download a sample PDF report to see the comprehensive analysis output:**

![PDF Report](https://img.shields.io/badge/PDF-Report-red?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)

*Click the badge above to view/download the sample PDF report*

</div>

## 🛠 Technology Stack

<div align="center">

### 🏗️ Architecture Overview



| Category | Technology | Version | Purpose |
|:--------:|:----------:|:-------:|:--------|
| 🐍 **Backend** | Python | 3.8+ | Core programming language |
| 🌐 **Web Framework** | Flask | 2.x | Web application framework |
| 🤖 **Deep Learning** | TensorFlow | 2.x | Neural network framework |
| 🖼️ **Image Processing** | OpenCV | 4.8+ | Computer vision operations |
| 📄 **PDF Generation** | ReportLab | 4.0+ | Report creation |
| 🎨 **Frontend** | Bootstrap 5 | Latest | Responsive UI framework |
| 🧠 **AI/ML** | CNN + Grad-CAM | Custom | Brain tumor classification |
| 📊 **Data** | 5,712 MRI Images | 4 Classes | Training dataset |
</div>
</div>

## 🚀 Installation

<div align="center">

### ⚡ Quick Start



| Requirement | Specification | Status |
|:-----------:|:-------------:|:------:|
| 🐍 **Python** | 3.8 or higher | Required |
| 📦 **Package Manager** | pip | Required |
| 💾 **RAM** | 4GB+ (8GB recommended) | Required |
| 🎮 **GPU** | Optional (for faster inference) | Optional |

</div>
</div>

### 📋 Step-by-Step Setup

<details>
<summary><b>🔧 Step 1: Clone Repository</b></summary>

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

</details>

<details>
<summary><b>🐍 Step 2: Create Virtual Environment</b></summary>

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

</details>

<details>
<summary><b>📦 Step 3: Install Dependencies</b></summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><b>🧠 Step 4: Download Model</b></summary>

```bash
# The model file (best_model.h5) should be in the project root
# If not available, the system will provide instructions
```

</details>

<details>
<summary><b>🚀 Step 5: Run Application</b></summary>

```bash
python app.py
```

The application will be available at `http://localhost:5000`

</details>

## 📖 Usage

<div align="center">

### 🎯 User Workflow



| Step | Action | Description |
|:----:|:------:|:-----------:|
| 1️⃣ | **Upload Image** | Drag & drop or click to upload MRI image (JPG/PNG, max 16MB) |
| 2️⃣ | **Patient Info** | Enter patient details (optional, kept private) |
| 3️⃣ | **AI Analysis** | Click "Analyze" and wait 3-5 seconds for processing |
| 4️⃣ | **View Results** | See predictions, confidence scores, and Grad-CAM visualization |
| 5️⃣ | **Download Report** | Generate comprehensive PDF report with all details |

</div>
</div>

### API Usage

```python
import requests

# Upload and analyze image
url = "http://localhost:5000/api/predict"
files = {'file': open('mri_scan.jpg', 'rb')}
response = requests.post(url, files=files)

# Get results
result = response.json()
print(f"Predicted: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

## 🏗 Model Architecture

### CNN Architecture
```
Input Layer: 150x150x3 (RGB)
├── Conv2D (32 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (512) + ReLU + Dropout(0.5)
├── Dense (256) + ReLU + Dropout(0.3)
└── Output Layer: Dense (4) + Softmax
```

### Training Details
- **Dataset**: Brain MRI scans (4 classes)
- **Training Samples**: 5,712 images
- **Validation Split**: 20%
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, zoom, flip

## 📚 API Documentation

### Endpoints

#### POST `/api/predict`
Upload and analyze MRI image

**Request:**
```bash
curl -X POST -F "file=@mri_scan.jpg" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "glioma",
    "confidence": 0.85,
    "all_scores": [0.85, 0.08, 0.04, 0.03],
    "class_index": 0
  },
  "images": {
    "original": "data:image/jpeg;base64,...",
    "gradcam": "data:image/jpeg;base64,..."
  }
}
```

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_loaded": true
}
```

## 📁 Project Structure

```
brain-tumor-classification/
├── app.py                      # Main Flask application
├── model/
    ├── best_model.h5              # Trained CNN model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── static/                    # Static assets
│   └── uploads/               # Uploaded images
├── templates/                 # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Upload page
│   ├── result.html           # Results page
│   └── about.html            # About page
├── utils/                     # Utility modules
│   ├── model_loader.py       # Model loading utilities
│   ├── preprocess.py         # Image preprocessing
│   ├── grad_cam.py           # Grad-CAM implementation
│   └── pdf_generator.py      # PDF report generation
│ 
├── test_model_loading.py
├── check_model_file.py
└── quick_test.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set Flask environment
export FLASK_ENV=development
export FLASK_DEBUG=1

# Optional: Set upload folder
export UPLOAD_FOLDER=static/uploads
```

### Model Configuration
- **Input Size**: 150x150 pixels
- **Batch Size**: 1 (for inference)
- **Preprocessing**: Normalization to [0,1]
- **Color Mode**: RGB

## 🧪 Testing

### Manual Testing
```bash
# Test model loading
python test_model_loading.py

# Check model file
python check_model_file.py

# Quick functionality test
python quick_test.py
```

### Manual Testing
1. Start the application: `python app.py`
2. Open browser: `http://localhost:5000`
3. Upload test MRI image
4. Verify results and Grad-CAM visualization
5. Test PDF report generation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install pytest pytest-flask flake8 black

# Run linting
flake8 .

# Run manual tests
python test_model_loading.py
python check_model_file.py
python quick_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

### Medical Disclaimer
**This software is for research and educational purposes only.** It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical decisions.

### Technical Disclaimer
- The model has been trained on a specific dataset and may not generalize to all cases
- Accuracy may vary depending on image quality and acquisition parameters
- Clinical correlation with patient history is essential
- This system is not intended for clinical use without proper validation



### Common Issues

#### Model Loading Error
```bash
# Ensure model file exists
ls -la best_model.h5

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Image Upload Error
- Verify file format (JPG/PNG only)
- Check file size (< 16MB)
- Ensure proper permissions

#### Grad-CAM Generation Error
- Check model architecture compatibility
- Verify image preprocessing
- Review error logs for details

## 🏆 Acknowledgments

- **Dataset**: Brain MRI scans from medical research institutions
- **Research**: Based on state-of-the-art CNN architectures
- **Community**: Open source contributors and researchers
- **Medical Advisors**: Healthcare professionals for guidance

## 📊 Performance Metrics

| Metric | Value | Status |
|:------:|:-----:|:------:|
| 🎯 **Overall Accuracy** | 94% | ✅ Excellent |
| 📊 **Precision** | 92% | ✅ High |
| 🔍 **Recall** | 91% | ✅ High |
| ⚖️ **F1-Score** | 92% | ✅ Balanced |
| 🖼️ **Training Images** | 5,712 | ✅ Large Dataset |
| ⚡ **Processing Time** | < 5 seconds | ✅ Fast |
| 💾 **Model Size** | 45 MB | ✅ Compact |

### 📈 Training Progress (Final Epochs)

<div >

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|:-----:|:-----------------:|:------------------:|:-------------:|:---------------:|
| 48 | 93.86% | 94.74% | 0.1663 | 0.1684 |
| 49 | 93.89% | 91.91% | 0.1629 | 0.2278 |
| 50 | 94.85% | 93.36% | 0.1479 | 0.1921 |

</div>

## 🔮 Future Enhancements

<div align="center">

### 🚀 Roadmap



| Enhancement | Priority | Status | Timeline |
|:-----------:|:--------:|:------:|:--------:|
| 🎥 **Real-time Video Analysis** | High | 🔄 Planned | Q2 2024 |
| 🎯 **3D MRI Support** | Medium | 🔄 Planned | Q3 2024 |
| 🔗 **Multi-modal Fusion** | High | 🔄 Planned | Q4 2024 |
| 🌐 **Federated Learning** | Medium | 🔄 Planned | Q1 2025 |
| 📱 **Mobile App** | Medium | 🔄 Planned | Q2 2025 |
| ☁️ **Cloud Deployment** | High | 🔄 Planned | Q3 2024 |
| 🔍 **Advanced XAI Methods** | Low | 🔄 Planned | Q4 2025 |
| 🏥 **Clinical Validation** | High | 🔄 Planned | Q2 2025 |
</div>
</div>

---


<div align="center">

### 🤝 Get in Touch

For inquiries or collaboration opportunities, reach out via:

</div>

<div align="center">


| 📧 **Email** | [sidhusidharth7075@gmail.com](mailto:sidhusidharth7075@gmail.com)
| 💼 **LinkedIn** | [LohithSappa](https://www.linkedin.com/in/lohith-sappa-aab07629a/) |

</div>

---

<div align="center">

⭐ **Don't forget to star this repository if you found it helpful!** ⭐

**Made with ❤️ for Medical AI Research**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/brain-tumor-classification?style=social&label=Star)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/brain-tumor-classification?style=social&label=Fork)](https://github.com/sidhusidharth7075/BrainMRI-Tumor-Detection-XAI.git)

</div> 
