# Chest X-ray Disease Prediction and Report Generation

This script loads a trained SemaCheXFormer model and generates comprehensive PDF reports with Grad-CAM visualization for chest X-ray images.

## Features

- ✅ Loads trained PyTorch model (`final_model.pth`)
- ✅ Preprocesses images exactly as during training
- ✅ Predicts 14 diseases with probability scores
- ✅ Generates Grad-CAM heatmaps for localization
- ✅ Creates professional radiology-style reports
- ✅ Exports to PDF with three-column layout (Report, Keywords, Localization)

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (with File Picker)

Simply run the script without the `--image` argument to open a file picker dialog:

```bash
python chest_xray_inference.py
```

A file dialog will open where you can browse and select your X-ray image from your local PC.

### Advanced Usage

You can also provide the image path directly:

```bash
python chest_xray_inference.py --image path/to/xray_image.png --model Model/final_model.pth
```

Or with all options:

```bash
python chest_xray_inference.py \
    --image path/to/xray_image.png \
    --model Model/final_model.pth \
    --output Custom_Report.pdf \
    --threshold 0.5 \
    --device cuda
```

### Parameters

- `--image` (optional): Path to the chest X-ray image file. If not provided, a file picker dialog will open to select the image from your local PC.
- `--model` (default: `Model/final_model.pth`): Path to the trained model file
- `--output` (default: `Chest_Report.pdf`): Output PDF filename
- `--threshold` (default: 0.5): Probability threshold for disease detection (0.0-1.0)
- `--device` (default: auto): Device to use (`cuda` or `cpu`)

## Supported Image Formats

The script supports common image formats:
- PNG
- JPEG/JPG
- BMP
- TIFF
- DICOM (if converted to standard image format first)

The file picker dialog will filter and show only image files for easy selection.

## Output

The script generates:
1. **Console Output**: 
   - Prediction probabilities for all 14 diseases
   - Detection status (detected/not detected)
   
2. **PDF Report** (`Chest_Report.pdf`):
   - **Column 1**: Full radiology report with findings and impression
   - **Column 2**: Detected disease keywords (or "NA" if none)
   - **Column 3**: Side-by-side visualization of original X-ray and Grad-CAM heatmap

## Diseases Detected

The model can detect the following 14 diseases:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## Example Output

```
PREDICTION RESULTS:
============================================================
Atelectasis        : 12.3% - ✗ Not detected
Cardiomegaly       : 8.5% - ✗ Not detected
Effusion           : 65.2% - ✓ DETECTED
Infiltration       : 45.1% - ✗ Not detected
Mass               : 3.2% - ✗ Not detected
...
============================================================
✓ PDF report saved to: Chest_Report.pdf
```

## Notes

- The model expects images to be preprocessed the same way as during training (224x224, normalized to [0,1])
- Grad-CAM visualization highlights regions the model considers important for its predictions
- If no diseases are detected above the threshold, the report will indicate "NA" for keywords and provide a normal findings report

## Troubleshooting

1. **Model loading errors**: Ensure the model file was saved correctly. The script supports both full model saves and state_dict saves.

2. **Grad-CAM not working**: If Grad-CAM fails, the script will create a fallback visualization with just the original image.

3. **Memory issues**: If running out of GPU memory, use `--device cpu` to run on CPU (slower but uses less memory).

