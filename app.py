"""
Unified Flask Application for Chest X-ray Disease Prediction
Combines Flask backend, React frontend, and Video Call functionality
"""

import os
import tempfile
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import model classes FIRST so they're available when loading the model
import torch
import torch.nn as nn
import torchxrayvision as xrv
from einops import rearrange
from einops.layers.torch import Rearrange

# Import the model architecture classes from chest_xray_inference
from chest_xray_inference import (
    TransformerEncoderBlock,
    PartitionReconstructionAttentionBlock_LMSA,
    ConvSEBlock,
    SemaCheXFormer
)

# Make these classes available in the current module namespace
import sys
sys.modules['__main__'].TransformerEncoderBlock = TransformerEncoderBlock
sys.modules['__main__'].PartitionReconstructionAttentionBlock_LMSA = PartitionReconstructionAttentionBlock_LMSA
sys.modules['__main__'].ConvSEBlock = ConvSEBlock
sys.modules['__main__'].SemaCheXFormer = SemaCheXFormer

# Import functions from your existing scripts
from preprocess_xray_gui import enhance_xray
from chest_xray_inference import predict_and_generate_report

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['REPORTS_FOLDER'] = os.path.join(os.getcwd(), 'reports')
app.config['VIDEO_UPLOADS_FOLDER'] = os.path.join(os.getcwd(), 'video_uploads')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Create necessary directories
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_UPLOADS_FOLDER'], exist_ok=True)

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# In-memory store of reports by room for video calls
reports_by_room = {}

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


# ============================================================================
# ROUTES FOR REACT FRONTEND
# ============================================================================

@app.route('/')
def index():
    """Serve the React frontend from dist folder."""
    frontend_dist = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
    if os.path.exists(os.path.join(frontend_dist, 'index.html')):
        return send_from_directory(frontend_dist, 'index.html')
    # Fallback to Flask template if React not built
    return render_template('index.html')


@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend static files."""
    frontend_dist = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
    if os.path.exists(frontend_dist):
        if os.path.exists(os.path.join(frontend_dist, path)):
            return send_from_directory(frontend_dist, path)
    # If path doesn't exist, serve index.html for React Router
    if os.path.exists(os.path.join(frontend_dist, 'index.html')):
        return send_from_directory(frontend_dist, 'index.html')
    return "Not found", 404


# ============================================================================
# ROUTES FOR VIDEO CALL APP
# ============================================================================

@app.route('/doctor')
def doctor():
    """Serve the Doctor Connect (video call) page."""
    return render_template('doctor.html')


@app.route('/video-call')
def video_call():
    """Serve the video call interface directly."""
    return render_template('video_call.html')


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Chest X-ray Prediction API is running',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/predict', methods=['POST'])
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


@app.route('/api/predict_json', methods=['POST'])
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


@app.route('/api/download_report/<filename>', methods=['GET'])
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


# ============================================================================
# VIDEO CALL UPLOAD ENDPOINT
# ============================================================================

@app.route('/api/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_report():
    """Upload PDF report for video call sharing."""
    try:
        if 'report' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['report']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files allowed'}), 400
        
        # Generate safe filename
        timestamp = int(datetime.now().timestamp() * 1000)
        safe_filename = secure_filename(file.filename)
        safe_filename = f"{timestamp}_{safe_filename}"
        
        # Save file
        file_path = os.path.join(app.config['VIDEO_UPLOADS_FOLDER'], safe_filename)
        file.save(file_path)
        
        return jsonify({
            'filename': safe_filename,
            'path': f'/api/video_uploads/{safe_filename}'
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/video_uploads/<filename>', methods=['GET'])
@app.route('/video_uploads/<filename>', methods=['GET'])
def serve_video_upload(filename):
    """Serve uploaded PDF files for video calls."""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config['VIDEO_UPLOADS_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404


# ============================================================================
# SOCKET.IO EVENT HANDLERS FOR VIDEO CALLS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f'Client connected: {request.sid}')
    emit('connected', {'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f'Client disconnected: {request.sid}')
    # Clean up rooms when user disconnects
    for room_id in list(reports_by_room.keys()):
        # Note: Socket.IO automatically handles room cleanup on disconnect
        pass


@socketio.on('join')
def handle_join(data):
    """Handle client joining a room."""
    room_id = data.get('roomId')
    if not room_id:
        emit('error', {'message': 'Room ID required'})
        return
    
    join_room(room_id)
    print(f'Client {request.sid} joined room {room_id}')
    
    # Notify others in the room
    emit('user-joined', {'sid': request.sid}, room=room_id, include_self=False)
    
    # Send existing reports to the joining client
    existing_reports = reports_by_room.get(room_id, [])
    emit('existing-reports', existing_reports)


@socketio.on('signal')
def handle_signal(data):
    """Handle WebRTC signaling (offer/answer/ICE candidates)."""
    room_id = data.get('roomId')
    signal_data = data.get('data')
    target_sid = data.get('to')
    
    if not signal_data:
        return
    
    # Send to specific user if target specified, otherwise broadcast to room
    if target_sid:
        emit('signal', {'from': request.sid, 'data': signal_data}, room=target_sid)
    elif room_id:
        emit('signal', {'from': request.sid, 'data': signal_data}, room=room_id, include_self=False)


@socketio.on('user-left')
def handle_user_left(data):
    """Handle user leaving a room."""
    room_id = data.get('roomId')
    if room_id:
        leave_room(room_id)
        emit('user-left', {'sid': request.sid}, room=room_id)


@socketio.on('report-shared')
def handle_report_shared(data):
    """Handle sharing a report link in a room."""
    room_id = data.get('roomId')
    filename = data.get('filename')
    url = data.get('url')
    
    if not room_id or not filename or not url:
        return
    
    # Store report in room history
    if room_id not in reports_by_room:
        reports_by_room[room_id] = []
    
    reports_by_room[room_id].append({
        'filename': filename,
        'url': url,
        'ts': datetime.now().timestamp()
    })
    
    # Keep only last 100 reports per room
    if len(reports_by_room[room_id]) > 100:
        reports_by_room[room_id] = reports_by_room[room_id][-100:]
    
    # Broadcast to others in the room
    emit('report-shared', {
        'filename': filename,
        'url': url,
        'from': request.sid
    }, room=room_id, include_self=False)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"⚠ Warning: Model file not found at {DEFAULT_MODEL_PATH}")
        print("Please ensure the model file exists or provide model_path in the request.")
    
    print("="*70)
    print("Starting Unified Chest X-ray AI Application")
    print("="*70)
    print(f"Default model path: {DEFAULT_MODEL_PATH}")
    print(f"Reports folder: {app.config['REPORTS_FOLDER']}")
    print(f"Video uploads folder: {app.config['VIDEO_UPLOADS_FOLDER']}")
    print("\nWeb Interface:")
    print("  - React Frontend: http://localhost:5000")
    print("  - Doctor Connect: http://localhost:5000/doctor")
    print("  - Video Call: http://localhost:5000/video-call")
    print("\nAPI Endpoints:")
    print("  - POST /api/predict          : Upload image → Get PDF report")
    print("  - POST /api/predict_json     : Upload image → Get JSON results")
    print("  - GET  /api/download_report/<file> : Download previously generated report")
    print("  - POST /api/upload           : Upload PDF for video call")
    print("  - GET  /api/health           : Health check")
    print("\nSocket.IO:")
    print("  - WebSocket connections for video calls")
    print("="*70)
    print("\nServer starting on http://0.0.0.0:5000")
    print("="*70)
    
    # Run with Socket.IO support
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

