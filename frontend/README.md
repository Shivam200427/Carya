# Chest X-ray AI Frontend (React + Vite)

A modern React interface for your Flask Chest X-ray prediction backend.

- Drag-and-drop X-ray upload with preview
- Threshold control and optional model path
- Generate a downloadable PDF report (from `/predict`)
- Preview predictions as JSON (from `/predict_json`)
- Clean, responsive UI; dev-time API proxy to avoid CORS issues

## Quick start

Prereqs:
- Node.js 18+
- Flask backend running on http://localhost:5000

Install deps:

```powershell
cd "frontend"
npm install
```

Run dev server (with API proxy to port 5000):

```powershell
npm run dev
```

Open the app: http://localhost:5173

## Configure API base URL

By default, dev uses a Vite proxy (calls to `/api/*` are proxied to `http://localhost:5000`).
For production builds, set `VITE_API_BASE_URL` to your backend URL:

```powershell
# .env (example)
VITE_API_BASE_URL=http://your-backend-host:5000
```

## Build for production

```powershell
npm run build
npm run preview  # optional local preview
```

You can serve the `dist/` folder with any static server. For same-origin hosting with Flask, you can either
- serve `dist/` via nginx or
- copy `dist/` into your Flask static root and point Flask to it (or add a / route that serves `index.html`).

## Endpoints expected

- `POST /predict` → `application/pdf` (download)
- `POST /predict_json` → JSON with predictions, `report_filename`, and optional `report_text`
- `GET  /download_report/<filename>` → `application/pdf`
- `GET  /health` → JSON

If CORS errors occur in production, either serve the frontend from the same origin as Flask or enable CORS in Flask (e.g. `flask-cors`).
