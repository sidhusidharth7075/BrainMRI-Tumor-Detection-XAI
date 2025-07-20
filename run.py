#!/usr/bin/env python3
"""
Brain Tumor Detection with Explainable AI - Startup Script
"""

import os
import sys
from app import app

def check_dependencies():
    """Check if required files and dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check if model file exists
    if not os.path.exists('best_model.h5'):
        print("❌ Error: best_model.h5 not found!")
        print("   Please ensure the model file is in the project root directory.")
        return False
    
    # Check if templates directory exists
    if not os.path.exists('templates'):
        print("❌ Error: templates directory not found!")
        return False
    
    # Check if index.html exists
    if not os.path.exists('templates/index.html'):
        print("❌ Error: templates/index.html not found!")
        return False
    
    print("✅ All dependencies found!")
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    """Main startup function"""
    print("🧠 Brain Tumor Detection with Explainable AI")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    print("\n🚀 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 