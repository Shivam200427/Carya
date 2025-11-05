# Unified Chest X-ray AI Application

This is a single unified Flask application that combines:
- **Flask Backend** - API for chest X-ray disease prediction
- **React Frontend** - Modern UI served from `frontend/dist`
- **Video Call App** - Doctor Connect with WebRTC video calls via Socket.IO

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build React Frontend (First Time)

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the Unified Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

## Application Structure

### Routes

- **`/`** - React frontend (served from `frontend/dist`)
- **`/doctor`** - Doctor Connect page (iframe with video call)
- **`/video-call`** - Direct video call interface
- **`/api/health`** - Health check endpoint
- **`/api/predict`** - Upload X-ray → Get PDF report
- **`/api/predict_json`** - Upload X-ray → Get JSON results
- **`/api/download_report/<filename>`** - Download previously generated report
- **`/api/upload`** - Upload PDF for video call sharing
- **`/api/video_uploads/<filename>`** - Serve uploaded PDFs

### Socket.IO Events

- **`connect`** - Client connects
- **`join`** - Join a video call room
- **`signal`** - WebRTC signaling (offer/answer/ICE candidates)
- **`report-shared`** - Share PDF report in room
- **`user-left`** - User leaves room

## Features

### 1. Chest X-ray Prediction
- Upload chest X-ray images
- Get AI-powered disease predictions
- Generate PDF reports with Grad-CAM visualizations
- Download reports in JSON or PDF format

### 2. Doctor Connect
- Video call interface for patient-doctor consultations
- Real-time WebRTC video/audio communication
- Share PDF reports during calls
- Room-based communication system

### 3. React Frontend
- Modern, responsive UI
- Drag-and-drop file upload
- Real-time prediction preview
- Integrated with backend API

## Development

### Building Frontend

When you make changes to the React frontend:

```bash
cd frontend
npm run build
cd ..
```

The built files will be in `frontend/dist/` and automatically served by Flask.

### Running in Development Mode

For development, you can run the React dev server separately:

```bash
# Terminal 1: Flask backend
python app.py

# Terminal 2: React frontend (dev mode)
cd frontend
npm run dev
```

Then access:
- React dev: `http://localhost:5173`
- Flask: `http://localhost:5000`

## Production Deployment

### Requirements

- Python 3.10+
- Node.js 18+ (for building frontend)
- All dependencies from `requirements.txt`

### Deployment Steps

1. **Build the frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Set environment variables:**
   - `SECRET_KEY` - Flask secret key for sessions
   - `PORT` - Server port (default: 5000)

3. **Run with production server:**
   ```bash
   # Using gunicorn with eventlet workers (recommended for Socket.IO)
   pip install gunicorn
   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
   ```

   Or use the built-in server (not recommended for production):
   ```bash
   python app.py
   ```

## File Structure

```
.
├── app.py                    # Unified Flask application
├── flask_backend.py          # Original Flask backend (deprecated, use app.py)
├── requirements.txt          # Python dependencies
├── frontend/                 # React frontend
│   ├── src/
│   ├── dist/                 # Built frontend (served by Flask)
│   └── package.json
├── templates/                # Flask templates
│   ├── index.html
│   ├── doctor.html
│   └── video_call.html
├── static/                   # Static files
│   └── video-call/           # Video call app assets
│       ├── script.js
│       └── style.css
├── video-call-app/           # Original video call app (reference)
├── Model/                    # ML model files
├── reports/                  # Generated PDF reports
└── video_uploads/            # PDFs uploaded during video calls
```

## Notes

- The React frontend must be built before running (`npm run build` in `frontend/`)
- Socket.IO requires eventlet or gevent for async support
- Video calls use WebRTC peer-to-peer; ensure proper STUN/TURN servers for NAT traversal
- PDF uploads are stored in `video_uploads/` directory
- Generated reports are stored in `reports/` directory

## Troubleshooting

### Frontend not loading?
- Make sure you've built the React app: `cd frontend && npm run build`
- Check that `frontend/dist/index.html` exists

### Socket.IO not working?
- Ensure `flask-socketio` and `eventlet` are installed
- Check browser console for WebSocket connection errors
- Verify CORS settings if accessing from different domain

### Video call not connecting?
- Check browser permissions for camera/microphone
- Ensure both users are in the same room ID
- Check NAT/firewall settings for WebRTC

## Migration from Separate Servers

If you were running separate servers before:

1. **Old setup:**
   - Flask backend: `flask_backend.py` on port 5000
   - Video call server: `video-call-app/server.js` on port 3000
   - React frontend: Vite dev server on port 5173

2. **New setup:**
   - Everything runs on `app.py` on port 5000
   - No need for separate Node.js server
   - React frontend is built and served as static files

All routes and functionality remain the same, just unified into one server!

