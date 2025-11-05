# Streamlit App - Chest X-ray AI Analysis

**One unified app, no Flask, no React, no separate servers - just Streamlit!**

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in browser:**
   - The app will automatically open at `http://localhost:8501`

## Features

### 🏠 Home - X-ray Analysis
- Upload chest X-ray images
- AI-powered disease prediction (14 diseases)
- Real-time probability visualization
- Automatic PDF report generation
- Download reports instantly

### 👨‍⚕️ Doctor Connect
- Chat interface for doctor consultation
- Share generated reports with doctors
- Quick access to external video call services

### 📋 Generated Reports
- View all generated reports
- Download PDFs anytime
- Review report history

## Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add Streamlit app"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select your repository and branch
5. **Set main file path:** `streamlit_app.py`
6. Click "Deploy"

### Step 3: Configure Secrets (if needed)

If you need API keys or other secrets:
1. Go to app settings
2. Click "Secrets"
3. Add your secrets in TOML format

## File Structure

```
.
├── streamlit_app.py          # Main Streamlit app (THIS IS IT!)
├── requirements.txt          # Python dependencies
├── chest_xray_inference.py   # ML inference functions
├── preprocess_xray_gui.py    # Image preprocessing
├── Model/
│   └── final_model.pth       # Trained model
└── reports/                  # Generated PDF reports (auto-created)
```

## Requirements

- Python 3.10+
- Model file: `Model/final_model.pth`
- All dependencies in `requirements.txt`

## Model File

Make sure your model file is in the repository:
- Path: `Model/final_model.pth`
- The app will check for it and show an error if not found

## Notes

- **Reports are stored locally** in the `reports/` directory
- **Session state** is used to store chat history and reports (resets on refresh)
- **PDF reports** are generated with timestamps and unique IDs
- **No database needed** - everything works with files and session state

## Troubleshooting

### Model not found?
- Ensure `Model/final_model.pth` exists in the repository
- Check the file path in the sidebar (default: `Model/final_model.pth`)

### Import errors?
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that `chest_xray_inference.py` and `preprocess_xray_gui.py` are in the same directory

### App not loading?
- Check Streamlit Cloud logs for errors
- Ensure `streamlit_app.py` is the main file
- Verify all dependencies are in `requirements.txt`

## Customization

### Change model path:
Edit the `DEFAULT_MODEL_PATH` variable in `streamlit_app.py`

### Modify threshold:
The threshold slider is already in the UI - users can adjust it

### Add more features:
Just edit `streamlit_app.py` - it's all in one file!

## That's it!

No Flask, no React, no separate servers - just one Streamlit app that does everything! 🚀

