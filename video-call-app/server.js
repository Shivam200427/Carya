const path = require('path');
const fs = require('fs');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const multer = require('multer');

const app = express();
const server = http.createServer(app);
const io = new Server(server);
// In-memory store of reports by room
const reportsByRoom = {};

const PORT = process.env.PORT || 3000;

// Serve static files from public
app.use(express.static(path.join(__dirname, 'public')));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer for PDF uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const safeOriginal = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, '_');
    cb(null, `${timestamp}_${safeOriginal}`);
  }
});
const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'application/pdf') cb(null, true);
  else cb(new Error('Only PDF files allowed'));
};
const upload = multer({ storage, fileFilter, limits: { fileSize: 10 * 1024 * 1024 } });

// Upload endpoint
app.post('/upload', upload.single('report'), (req, res) => {
  res.json({ filename: req.file.filename, path: `/uploads/${req.file.filename}` });
});

// Serve uploaded files statically (optional)
app.use('/uploads', express.static(uploadsDir));

io.on('connection', (socket) => {
  // Join a room for signaling
  socket.on('join', (roomId) => {
    socket.join(roomId);
    socket.to(roomId).emit('user-joined', socket.id);
    // Send existing reports to the joining client
    socket.emit('existing-reports', reportsByRoom[roomId] || []);
  });

  // Relay signaling data (offer/answer/ice) to peers
  socket.on('signal', ({ roomId, data, to }) => {
    if (to) {
      io.to(to).emit('signal', { from: socket.id, data });
    } else if (roomId) {
      socket.to(roomId).emit('signal', { from: socket.id, data });
    }
  });

  socket.on('disconnecting', () => {
    const rooms = [...socket.rooms].filter((r) => r !== socket.id);
    rooms.forEach((roomId) => socket.to(roomId).emit('user-left', socket.id));
  });

  // Relay shared report links to room peers
  socket.on('report-shared', ({ roomId, filename, url }) => {
    if (!roomId || !filename || !url) return;
    // Save to room history (cap to 100 items)
    if (!reportsByRoom[roomId]) reportsByRoom[roomId] = [];
    reportsByRoom[roomId].push({ filename, url, ts: Date.now() });
    if (reportsByRoom[roomId].length > 100) reportsByRoom[roomId].shift();
    socket.to(roomId).emit('report-shared', { filename, url, from: socket.id });
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});


