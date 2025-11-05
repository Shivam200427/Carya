"""
Chest X-ray Disease Prediction and Report Generation System
Loads a trained SemaCheXFormer model and generates comprehensive PDF reports with Grad-CAM visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from einops import rearrange
from einops.layers.torch import Rearrange
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os
import argparse
import io

# File picker support (cross-platform)
try:
    from tkinter import filedialog
    import tkinter as tk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# ==============================================================================
# MODEL ARCHITECTURE (Same as training)
# ==============================================================================

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        normed_x = self.norm1(x)
        x = x + self.attn(normed_x, normed_x, normed_x)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PartitionReconstructionAttentionBlock_LMSA(nn.Module):
    def __init__(self, in_channels, dim, patch_size, num_heads, mlp_dim, num_classes, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * (patch_size ** 2)
        self.partition = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_projection = nn.Linear(patch_dim, dim)
        self.label_embeddings = nn.Parameter(torch.randn(1, num_classes, dim))
        self.attention_module = TransformerEncoderBlock(dim, num_heads, mlp_dim, dropout)
    
    def forward(self, x):
        b, _, h, w = x.shape
        tokens = self.partition(x)
        label_embs = self.label_embeddings.expand(b, -1, -1)
        tokens = self.patch_projection(tokens)
        processed_tokens = self.attention_module(torch.cat((tokens, label_embs), dim=1))
        processed_visual_tokens = processed_tokens[:, :tokens.shape[1]]
        recon_h, recon_w = h // self.patch_size, w // self.patch_size
        return rearrange(processed_visual_tokens, 'b (h w) d -> b d h w', h=recon_h, w=recon_w)

class ConvSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=8):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        cnn_out = self.conv_block(x)
        return cnn_out * self.se(cnn_out)

class SemaCheXFormer(nn.Module):
    def __init__(self, num_classes=14, dim=256, num_heads=8, mlp_dim=512, cnn_channels=256):
        super().__init__()
        pretrained_model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        self.backbone = pretrained_model.features
        self.backbone.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.norm0 = nn.BatchNorm2d(64)
        backbone_out_channels = 1024
        self.attention_stage = PartitionReconstructionAttentionBlock_LMSA(backbone_out_channels, dim, 1, num_heads, mlp_dim, num_classes)
        self.cnn_se_stage = ConvSEBlock(dim, cnn_channels)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(cnn_channels, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        attention_out = self.attention_stage(features)
        cnn_out = self.cnn_se_stage(attention_out)
        return self.classification_head(cnn_out)

# Make the model class discoverable under __main__ for loading pickled full-model checkpoints
# Some training scripts save with torch.save(model, ...) while the class lives in __main__ at save time.
# When loading from another module (e.g., Flask app), pickle looks for __main__.SemaCheXFormer.
# We register the class on the current __main__ module so torch.load can resolve it.
try:
    import sys as _sys
    _main_mod = _sys.modules.get('__main__')
    if _main_mod is not None and not hasattr(_main_mod, 'SemaCheXFormer'):
        setattr(_main_mod, 'SemaCheXFormer', SemaCheXFormer)
except Exception:
    # Non-fatal; we still support state_dict loading paths below
    pass

# ==============================================================================
# GRAD-CAM IMPLEMENTATION
# ==============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_forward = None
        self.hook_backward = None
        
        # Hook to store activations
        self.hook_forward = self.target_layer.register_forward_hook(self.save_activation)
        # Hook to store gradients
        self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        
        # Ensure input requires gradient
        if not input_image.requires_grad:
            input_image = input_image.requires_grad_()
        
        output = self.model(input_image)
        
        if class_idx is None:
            # Use the class with highest probability
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hook registration.")
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.cpu().numpy(), output
    
    def __del__(self):
        if self.hook_forward is not None:
            self.hook_forward.remove()
        if self.hook_backward is not None:
            self.hook_backward.remove()

# ==============================================================================
# IMAGE PREPROCESSING
# ==============================================================================

def preprocess_image(image_path, image_size=224):
    """Preprocess image exactly as done during training."""
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # Assume it's already a numpy array or PIL Image
        if isinstance(image_path, Image.Image):
            image = np.array(image_path.convert('RGB'))
        else:
            image = image_path
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Store original for visualization
    original_image = cv2.resize(image.copy(), (image_size, image_size))
    
    # Apply transform (same as validation transform)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    tensor_image = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return tensor_image, original_image

# ==============================================================================
# RADIOLOGY REPORT GENERATION
# ==============================================================================

def generate_radiology_report(predictions, probabilities, labels, threshold=0.5):
    """Generate a radiology-style report based on predictions."""
    detected_diseases = []
    high_confidence_diseases = []
    
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        if prob > threshold:
            detected_diseases.append((label, prob))
            if prob > 0.7:
                high_confidence_diseases.append((label, prob))
    
    # Sort by probability
    detected_diseases.sort(key=lambda x: x[1], reverse=True)
    
    if not detected_diseases:
        report = (
            "CHEST X-RAY REPORT\n\n"
            "EXAMINATION: Frontal chest radiograph\n\n"
            "CLINICAL INDICATION: Screening/Evaluation\n\n"
            "FINDINGS:\n"
            "No acute cardiopulmonary abnormalities identified. "
            "The cardiomediastinal silhouette is within normal limits. "
            "No focal consolidation, pleural effusion, or pneumothorax is demonstrated. "
            "The osseous structures are unremarkable.\n\n"
            "IMPRESSION:\n"
            "No significant abnormality detected on this chest radiograph."
        )
        keywords = "NA"
    else:
        # Build findings section
        findings = []
        findings.append("CHEST X-RAY REPORT\n\n")
        findings.append("EXAMINATION: Frontal chest radiograph\n\n")
        findings.append("CLINICAL INDICATION: Screening/Evaluation\n\n")
        findings.append("FINDINGS:\n")
        
        # Add specific findings for each detected disease
        disease_descriptions = {
            "Atelectasis": "There are areas of atelectasis present, suggesting volume loss or airway obstruction.",
            "Cardiomegaly": "Cardiomegaly is noted with enlargement of the cardiac silhouette.",
            "Effusion": "Pleural effusion is identified, indicating fluid accumulation in the pleural space.",
            "Infiltration": "Pulmonary infiltrates are observed, which may represent infection, inflammation, or edema.",
            "Mass": "A pulmonary mass is detected and requires further evaluation and follow-up imaging.",
            "Nodule": "Pulmonary nodule(s) are identified. Clinical correlation and follow-up imaging recommended.",
            "Pneumonia": "Consolidation patterns consistent with pneumonia are present.",
            "Pneumothorax": "Pneumothorax is detected with evidence of air in the pleural space.",
            "Consolidation": "Pulmonary consolidation is noted, consistent with pneumonia or other infiltrative processes.",
            "Edema": "Pulmonary edema is present, suggesting fluid overload or heart failure.",
            "Emphysema": "Features of emphysema are observed with hyperinflation and decreased lung markings.",
            "Fibrosis": "Pulmonary fibrosis is identified with evidence of interstitial lung disease.",
            "Pleural_Thickening": "Pleural thickening is noted, which may be related to prior inflammation or trauma.",
            "Hernia": "Hiatal hernia is identified in the upper abdomen."
        }
        
        for label, prob in detected_diseases:
            desc = disease_descriptions.get(label, f"{label} is detected.")
            confidence = "high confidence" if prob > 0.7 else "moderate confidence"
            findings.append(f"{desc} (Confidence: {confidence}, Probability: {prob:.1%})\n")
        
        findings.append("\nIMPRESSION:\n")
        if len(detected_diseases) == 1:
            findings.append(f"{detected_diseases[0][0]} detected with {detected_diseases[0][1]:.1%} confidence. Clinical correlation recommended.")
        else:
            primary = detected_diseases[0][0]
            findings.append(f"Multiple abnormalities detected. Primary finding: {primary}. "
                          f"Additional findings: {', '.join([d[0] for d in detected_diseases[1:]])}. "
                          f"Clinical correlation and follow-up imaging recommended.")
        
        report = "".join(findings)
        keywords = ", ".join([d[0] for d in detected_diseases])
    
    return report, keywords

# ==============================================================================
# GRAD-CAM VISUALIZATION
# ==============================================================================

def create_gradcam_visualization(original_image, cam, image_size=224):
    """Create side-by-side visualization of original and Grad-CAM overlay."""
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (image_size, image_size))
    
    # Convert to heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM overlay
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap, alpha=0.5, interpolation='bilinear')
    axes[1].set_title('Grad-CAM Localization', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Convert to image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def create_bounding_box_visualization(original_image, cam, image_size=224, threshold_percentile=75):
    """Create visualization with square outlines highlighting detected regions."""
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (image_size, image_size))
    
    # Threshold the CAM to find high-activation regions
    threshold = np.percentile(cam_resized, threshold_percentile)
    binary_mask = (cam_resized > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image for drawing
    image_with_boxes = original_image.copy()
    
    # If the image is grayscale, convert to RGB for colored boxes
    if len(image_with_boxes.shape) == 2:
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_GRAY2RGB)
    elif image_with_boxes.shape[2] == 1:
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes around detected regions
    box_color = (0, 255, 0)  # Green color
    box_thickness = 3
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small regions
        if w > 10 and h > 10:
            # Draw rectangle
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), box_color, box_thickness)
    
    # If no boxes were drawn, draw a box around the highest activation region
    if len(contours) == 0 or all(cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] < 100 for c in contours):
        # Find the region with maximum activation
        max_loc = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
        center_y, center_x = max_loc
        
        # Draw a box centered on the max activation point
        box_size = min(image_size // 4, 60)
        x1 = max(0, center_x - box_size // 2)
        y1 = max(0, center_y - box_size // 2)
        x2 = min(image_size, center_x + box_size // 2)
        y2 = min(image_size, center_y + box_size // 2)
        
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), box_color, box_thickness)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(image_with_boxes, cmap='gray' if len(original_image.shape) == 2 else None)
    ax.set_title('Abnormal Region Identification (Bounding Boxes)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

# ==============================================================================
# PDF GENERATION
# ==============================================================================

def create_pdf_report(report_text, keywords, gradcam_image_buf, bounding_box_image_buf, output_path="Chest_Report.pdf"):
    """Create a professional PDF report with three columns."""
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           rightMargin=0.4*inch, leftMargin=0.4*inch,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           allowSplitting=1)  # Allow content to split across pages
    
    # Container for elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1E88E5'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1E88E5'),
        spaceAfter=6,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    keyword_style = ParagraphStyle(
        'KeywordStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1976D2'),
        leading=16
    )
    
    # Title
    title = Paragraph("CHEST X-RAY ANALYSIS REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Date
    date_text = f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    date_para = Paragraph(date_text, styles['Normal'])
    elements.append(date_para)
    elements.append(Spacer(1, 0.3*inch))
    
    # Three-column table
    # Prepare content for each column
    col1_data = [
        [Paragraph("<b>RADIOLOGY REPORT</b>", header_style)],
        [Paragraph(report_text.replace('\n', '<br/>'), body_style)]
    ]
    
    col2_data = [
        [Paragraph("<b>KEYWORDS</b>", header_style)],
        [Paragraph(keywords, keyword_style)]
    ]
    
    # Load and prepare Grad-CAM image
    gradcam_image_buf.seek(0)
    # Adjust image size for full-width display
    img = RLImage(gradcam_image_buf, width=7.0*inch, height=3.5*inch)
    col3_data = [
        [Paragraph("<b>LOCALIZATION RESULT</b>", header_style)],
        [img]
    ]
    
    # Light blue color for headers (#ADD8E6 is light blue, #B3D9FF is a softer blue)
    header_color = colors.HexColor('#B3D9FF')
    grid_color = colors.HexColor('#90C7E7')
    
    # Create tables for each column - adjusted widths to fit A4 (total ~7.5 inches)
    col1_table = Table(col1_data, colWidths=[5.5*inch])
    col1_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, grid_color),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1565C0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    col2_table = Table(col2_data, colWidths=[1.6*inch])
    col2_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, grid_color),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1565C0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    
    col3_table = Table(col3_data, colWidths=[3.7*inch])
    col3_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, grid_color),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1565C0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    
    # Create a flexible layout: Put image on top, then two columns below
    # This prevents overflow issues with long reports
    
    # First row: Image spans full width
    img_row_table = Table([
        [Paragraph("<b>LOCALIZATION RESULT</b>", header_style)],
        [img]
    ], colWidths=[7.5*inch])
    img_row_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, grid_color),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1565C0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(img_row_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add bounding box visualization
    if bounding_box_image_buf is not None:
        bounding_box_image_buf.seek(0)
        bbox_img = RLImage(bounding_box_image_buf, width=7.0*inch, height=3.5*inch)
        
        bbox_table = Table([
            [Paragraph("<b>ABNORMAL REGION IDENTIFICATION</b>", header_style)],
            [bbox_img]
        ], colWidths=[7.5*inch])
        bbox_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, grid_color),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1565C0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(bbox_table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Second row: Two columns for report and keywords
    two_col_table_data = [
        [col1_table, col2_table]
    ]
    two_col_table = Table(two_col_table_data, colWidths=[5.5*inch, 2.0*inch])
    two_col_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    
    elements.append(two_col_table)
    
    # Build PDF
    doc.build(elements)
    print(f"✓ PDF report saved to: {output_path}")

# ==============================================================================
# MAIN INFERENCE FUNCTION
# ==============================================================================

def predict_and_generate_report(model_path, image_path, output_pdf="Chest_Report.pdf", device=None, threshold=0.5):
    """Main function to run inference and generate PDF report."""
    
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate unique filename with timestamp if using default
    if output_pdf == "Chest_Report.pdf":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = output_pdf.replace(".pdf", "")
        output_pdf = f"{base_name}_{timestamp}.pdf"
        print(f"Report will be saved as: {output_pdf}")
    
    # Disease labels (order must match training)
    LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    
    # Load model
    print(f"Loading model from {model_path}...")
    
    try:
        # First try: Load as full model (your model was saved with torch.save(model, ...))
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, nn.Module):
            # Full model object saved - use it directly
            print("✓ Detected full model object")
            model = checkpoint
        elif isinstance(checkpoint, dict):
            # State dict or checkpoint dictionary
            print("✓ Detected state dict or checkpoint dictionary")
            model = SemaCheXFormer(num_classes=len(LABELS))
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✓ Loaded from 'state_dict' key")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded from 'model_state_dict' key")
            else:
                # Assume it's a direct state dict
                try:
                    model.load_state_dict(checkpoint)
                    print("✓ Loaded from direct state dict")
                except RuntimeError as re:
                    # Try with strict=False for partial loading
                    print(f"Warning: Some keys may be missing. Using strict=False...")
                    model.load_state_dict(checkpoint, strict=False)
                    print("✓ Loaded with strict=False")
        else:
            # Unexpected format - try as state dict anyway
            print("⚠ Unexpected checkpoint format, attempting to load as state dict...")
            model = SemaCheXFormer(num_classes=len(LABELS))
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Loaded with strict=False")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Attempting direct model loading as fallback...")
        try:
            # Last resort: try loading directly
            model = torch.load(model_path, map_location=device)
            if not isinstance(model, nn.Module):
                raise ValueError(f"Loaded object is not a model (type: {type(model)})")
            print("✓ Loaded model directly")
        except Exception as e2:
            print(f"❌ All loading methods failed. Last error: {e2}")
            raise RuntimeError(f"Could not load model from {model_path}. Please ensure the model file is valid.")
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Preprocess image
    print(f"Preprocessing image: {image_path}...")
    try:
        tensor_image, original_image = preprocess_image(image_path)
        tensor_image = tensor_image.to(device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise
    print("✓ Image preprocessed")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(tensor_image)
        probabilities = torch.sigmoid(output[0]).cpu().numpy()
    
    predictions = (probabilities > threshold).astype(int)
    
    # Print predictions
    print("\n" + "="*60)
    print("PREDICTION RESULTS:")
    print("="*60)
    detected = []
    for i, (label, prob, pred) in enumerate(zip(LABELS, probabilities, predictions)):
        status = "✓ DETECTED" if pred == 1 else "✗ Not detected"
        print(f"{label:<20}: {prob:.1%} - {status}")
        if pred == 1:
            detected.append((label, prob))
    
    if not detected:
        print("\nNo diseases detected (all probabilities below threshold)")
    
    print("="*60 + "\n")
    
    # Generate Grad-CAM
    print("Generating Grad-CAM visualization...")
    try:
        # Use the CNN-SE stage output for Grad-CAM (good intermediate representation)
        grad_cam = GradCAM(model, model.cnn_se_stage)
        
        # Use highest probability class for CAM
        if len(detected) > 0:
            class_idx = np.argmax(probabilities)
        else:
            # Use highest probability even if below threshold
            class_idx = np.argmax(probabilities)
        
        cam, _ = grad_cam.generate_cam(tensor_image, class_idx=class_idx)
        gradcam_image_buf = create_gradcam_visualization(original_image, cam)
        
        # Generate bounding box visualization
        bounding_box_image_buf = create_bounding_box_visualization(original_image, cam)
        print("✓ Bounding box visualization generated")
        print("✓ Grad-CAM visualization generated")
    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM: {e}")
        print("Creating fallback visualization...")
        # Fallback: just show original image
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(original_image, cmap='gray')
        ax.set_title('Original X-ray', fontsize=14, fontweight='bold')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close()
        gradcam_image_buf = buf
        bounding_box_image_buf = None  # No bounding boxes if Grad-CAM failed
    
    # Generate radiology report
    print("Generating radiology report...")
    report_text, keywords = generate_radiology_report(predictions, probabilities, LABELS, threshold=threshold)
    print("✓ Report generated")
    
    # Create PDF
    print(f"Generating PDF report: {output_pdf}...")
    create_pdf_report(report_text, keywords, gradcam_image_buf, bounding_box_image_buf, output_pdf)
    print("✓ PDF report created successfully!")
    
    return {
        'predictions': dict(zip(LABELS, predictions)),
        'probabilities': dict(zip(LABELS, probabilities)),
        'detected_diseases': detected if detected else "NA",
        'report_text': report_text,
        'keywords': keywords
    }

# ==============================================================================
# FILE PICKER FUNCTION
# ==============================================================================

def select_image_file():
    """Open a file dialog to select an X-ray image (cross-platform)."""
    if not HAS_TKINTER:
        return None
    
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG *.bmp *.BMP *.tiff *.TIFF"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chest X-ray Disease Prediction and Report Generation")
    parser.add_argument("--model", type=str, default="Model/final_model.pth",
                        help="Path to the trained model file")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the chest X-ray image (if not provided, file picker will open)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF filename (if not provided, will auto-generate with timestamp)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for disease detection (default: 0.5)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu), auto-detect if not specified")
    
    args = parser.parse_args()
    
    # If image path not provided, open file picker
    if args.image is None:
        if HAS_TKINTER:
            print("\n" + "="*60)
            print("Opening file picker to select X-ray image...")
            print("="*60)
            args.image = select_image_file()
            if args.image is None:
                print("\nNo image selected. Exiting.")
                exit(0)
            print(f"✓ Selected image: {args.image}")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("ERROR: tkinter not available for file picker.")
            print("Please provide --image argument with the image path.")
            print("="*60)
            print("To install tkinter:")
            print("  - Windows/Mac: Usually pre-installed with Python")
            print("  - Linux: sudo apt-get install python3-tk")
            print("="*60)
            exit(1)
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"Chest_Report_{timestamp}.pdf"
        print(f"Output PDF will be saved as: {args.output}")
    
    # Run inference
    try:
        results = predict_and_generate_report(
            model_path=args.model,
            image_path=args.image,
            output_pdf=args.output,
            threshold=args.threshold,
            device=args.device
        )
        print("\n" + "="*60)
        print("SUCCESS! Report generation completed.")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

