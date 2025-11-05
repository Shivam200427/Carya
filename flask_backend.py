"""
Flask Backend for Chest X-ray Disease Prediction
Integrates preprocessing from preprocess_xray_gui.py and inference from chest_xray_inference.py
Serves a web interface for file upload and returns PDF reports directly for download
"""

import os
import tempfile
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import model classes FIRST so they're available when loading the model
# This is necessary because PyTorch needs these classes to unpickle the saved model
import torch
import torch.nn as nn
import torchxrayvision as xrv
from einops import rearrange
from einops.layers.torch import Rearrange

# Import the model architecture classes from chest_xray_inference
# These must be imported before loading the model
from chest_xray_inference import (
    TransformerEncoderBlock,
    PartitionReconstructionAttentionBlock_LMSA,
    ConvSEBlock,
    SemaCheXFormer
)

# Make these classes available in the current module namespace
# This is necessary because PyTorch's unpickler looks for classes in __main__
# when the model was saved from a script run as __main__
import sys

# Add the classes to the current module so torch.load can find them
# This fixes the AttributeError when loading models saved from chest_xray_inference.py
sys.modules['__main__'].TransformerEncoderBlock = TransformerEncoderBlock
sys.modules['__main__'].PartitionReconstructionAttentionBlock_LMSA = PartitionReconstructionAttentionBlock_LMSA
sys.modules['__main__'].ConvSEBlock = ConvSEBlock
sys.modules['__main__'].SemaCheXFormer = SemaCheXFormer

# Import functions from your existing scripts
from preprocess_xray_gui import enhance_xray
from chest_xray_inference import predict_and_generate_report

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['REPORTS_FOLDER'] = os.path.join(os.getcwd(), 'reports')

# Create reports folder if it doesn't exist
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'PNG', 'JPG', 'JPEG', 'BMP', 'TIFF'}

# Default model path
DEFAULT_MODEL_PATH = "Model/final_model.pth"
DEFAULT_THRESHOLD = 0.5


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_preprocessed_image(preprocessed_array, output_path):
    """
    Save preprocessed numpy array (0-1 normalized) to image file.
    
    Args:
        preprocessed_array: numpy array in range [0, 1], shape (512, 512)
        output_path: path to save the image
    """
    # Convert from [0,1] to [0,255] and change to uint8
    img_uint8 = (preprocessed_array * 255).astype(np.uint8)
    
    # Save using OpenCV
    cv2.imwrite(output_path, img_uint8)
    return output_path


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/doctor')
def doctor():
    """Serve the Doctor Connect (video call) page."""
    return render_template('doctor.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Chest X-ray Prediction API is running',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint - returns PDF directly for download.
    
    Request (multipart/form-data):
        - file: image file (required)
        - model_path (optional): path to model file (default: Model/final_model.pth)
        - threshold (optional): probability threshold (default: 0.5)
    
    Response: PDF file (application/pdf) - downloads directly
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided. Please upload an image file.'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get optional parameters
        model_path = request.form.get('model_path', DEFAULT_MODEL_PATH)
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))

        # Validate/normalize model path: only allow typical torch checkpoints
        allowed_model_exts = {'.pth', '.pt'}
        _, ext = os.path.splitext(model_path)
        if ext and ext.lower() not in allowed_model_exts:
            return jsonify({
                'success': False,
                'error': f'Invalid model file extension: {ext}. Allowed: {", ".join(sorted(allowed_model_exts))}'
            }), 400
        
        # Validate model path
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': f'Model file not found: {model_path}'
            }), 404
        
        # Generate unique filename for report
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'Chest_Report_{timestamp}_{report_id[:8]}.pdf'
        output_pdf_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        # Create temporary files for input and preprocessed image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as input_tmp:
            input_path = input_tmp.name
            file.save(input_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as preprocessed_tmp:
            preprocessed_path = preprocessed_tmp.name
        
        try:
            # Step 1: Preprocess the image using enhance_xray()
            print("Step 1: Preprocessing image...")
            preprocessed_array = enhance_xray(input_path, target_size=(512, 512))
            print("✓ Image preprocessed")
            
            # Step 2: Save preprocessed image to temporary file
            save_preprocessed_image(preprocessed_array, preprocessed_path)
            print("✓ Preprocessed image saved")
            
            # Step 3: Run inference using the preprocessed image
            print("Step 2: Running inference and generating PDF report...")
            results = predict_and_generate_report(
                model_path=model_path,
                image_path=preprocessed_path,  # Use preprocessed image
                output_pdf=output_pdf_path,
                threshold=threshold,
                device=None  # Auto-detect device
            )
            print("✓ Inference completed and PDF generated")
            
            # Return PDF file directly as download
            return send_file(
                output_pdf_path,
                as_attachment=True,
                download_name=report_filename,
                mimetype='application/pdf'
            )
        
        finally:
            # Clean up temporary files (keep the PDF report)
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(preprocessed_path):
                    os.unlink(preprocessed_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary files: {e}")
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/predict_json', methods=['POST'])
def predict_json():
    """
    Alternative endpoint that returns JSON results instead of PDF.
    Useful for getting predictions without downloading PDF.
    
    Request: Same as /predict endpoint
    
    Response: JSON with predictions and report path
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided.'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file'
            }), 400
        
        model_path = request.form.get('model_path', DEFAULT_MODEL_PATH)
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': f'Model file not found: {model_path}'
            }), 404
        
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'Chest_Report_{timestamp}_{report_id[:8]}.pdf'
        output_pdf_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as input_tmp:
            input_path = input_tmp.name
            file.save(input_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as preprocessed_tmp:
            preprocessed_path = preprocessed_tmp.name
        
        try:
            # Preprocess
            preprocessed_array = enhance_xray(input_path, target_size=(512, 512))
            save_preprocessed_image(preprocessed_array, preprocessed_path)
            
            # Run inference
            results = predict_and_generate_report(
                model_path=model_path,
                image_path=preprocessed_path,
                output_pdf=output_pdf_path,
                threshold=threshold,
                device=None
            )
            
            # Return JSON results
            return jsonify({
                'success': True,
                'message': 'Prediction completed successfully',
                'predictions': results['predictions'],
                'probabilities': {k: float(v) for k, v in results['probabilities'].items()},
                'detected_diseases': results['detected_diseases'],
                'keywords': results['keywords'],
                'report_path': output_pdf_path,
                'report_filename': report_filename,
                'report_text': results['report_text']
            }), 200
        
        finally:
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(preprocessed_path):
                    os.unlink(preprocessed_path)
            except Exception:
                pass
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    """
    Download a previously generated PDF report.
    
    Args:
        filename: Name of the report file
    """
    try:
        filename = secure_filename(filename)
        
        if not filename.endswith('.pdf'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        file_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Report not found'}), 404
        
        if not file_path.startswith(app.config['REPORTS_FOLDER']):
            return jsonify({'error': 'Unauthorized access'}), 403
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500


if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"⚠ Warning: Model file not found at {DEFAULT_MODEL_PATH}")
        print("Please ensure the model file exists or provide model_path in the request.")
    
    print("="*70)
    print("Starting Chest X-ray Prediction API Server")
    print("="*70)
    print(f"Default model path: {DEFAULT_MODEL_PATH}")
    print(f"Reports folder: {app.config['REPORTS_FOLDER']}")
    print("\nWeb Interface:")
    print("  - Open your browser and go to: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  - POST /predict          : Upload image → Get PDF report (downloads directly)")
    print("  - POST /predict_json     : Upload image → Get JSON results")
    print("  - GET  /download_report/<file> : Download previously generated report")
    print("  - GET  /health           : Health check")
    print("="*70)
    print("\nServer starting on http://0.0.0.0:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)


