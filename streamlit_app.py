"""
Unified Streamlit Application for Chest X-ray Disease Prediction
Everything in one place - no Flask, no React, no separate servers!
"""

import streamlit as st
import os
import tempfile
import uuid
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn

# Import model classes FIRST
import torchxrayvision as xrv
from einops import rearrange
from einops.layers.torch import Rearrange

# Import the model architecture classes
from chest_xray_inference import (
    TransformerEncoderBlock,
    PartitionReconstructionAttentionBlock_LMSA,
    ConvSEBlock,
    SemaCheXFormer
)

# Make these classes available
import sys
sys.modules['__main__'].TransformerEncoderBlock = TransformerEncoderBlock
sys.modules['__main__'].PartitionReconstructionAttentionBlock_LMSA = PartitionReconstructionAttentionBlock_LMSA
sys.modules['__main__'].ConvSEBlock = ConvSEBlock
sys.modules['__main__'].SemaCheXFormer = SemaCheXFormer

# Import functions
from preprocess_xray_gui import enhance_xray
from chest_xray_inference import predict_and_generate_report

# Page config
st.set_page_config(
    page_title="Chest X-ray AI Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default paths
DEFAULT_MODEL_PATH = "Model/final_model.pth"
DEFAULT_THRESHOLD = 0.5

# Initialize session state
if 'reports' not in st.session_state:
    st.session_state.reports = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def save_preprocessed_image(preprocessed_array, output_path):
    """Save preprocessed numpy array to image file."""
    img_uint8 = (preprocessed_array * 255).astype(np.uint8)
    cv2.imwrite(output_path, img_uint8)
    return output_path


# ============================================================================
# MAIN APP
# ============================================================================

st.title("🫁 Chest X-ray AI Analysis System")
st.markdown("**AI-powered radiology report generation with 14 disease classifications**")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Home - X-ray Analysis", "👨‍⚕️ Doctor Connect", "📋 Generated Reports"]
)

# ============================================================================
# PAGE 1: X-RAY ANALYSIS
# ============================================================================

if page == "🏠 Home - X-ray Analysis":
    st.header("Upload Chest X-ray Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
    
    with col2:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            help="Diseases above this probability will be detected"
        )
        
        model_path = st.text_input(
            "Model Path (optional)",
            value=DEFAULT_MODEL_PATH,
            help="Path to the trained model file"
        )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Chest X-ray Image", use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze X-ray", type="primary", use_container_width=True):
            with st.spinner("Processing image and generating report..."):
                try:
                    # Check model exists
                    if not os.path.exists(model_path):
                        st.error(f"❌ Model file not found: {model_path}")
                        st.stop()
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        input_path = tmp_file.name
                        image.save(input_path)
                    
                    # Preprocess image
                    st.info("📸 Preprocessing image...")
                    preprocessed_array = enhance_xray(input_path, target_size=(512, 512))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as preprocessed_tmp:
                        preprocessed_path = preprocessed_tmp.name
                        save_preprocessed_image(preprocessed_array, preprocessed_path)
                    
                    # Generate report
                    st.info("🤖 Running AI inference...")
                    report_id = str(uuid.uuid4())
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f'Chest_Report_{timestamp}_{report_id[:8]}.pdf'
                    
                    # Create reports directory
                    reports_dir = os.path.join(os.getcwd(), 'reports')
                    os.makedirs(reports_dir, exist_ok=True)
                    output_pdf_path = os.path.join(reports_dir, report_filename)
                    
                    # Run inference
                    results = predict_and_generate_report(
                        model_path=model_path,
                        image_path=preprocessed_path,
                        output_pdf=output_pdf_path,
                        threshold=threshold,
                        device=None
                    )
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Detected Diseases")
                        detected = results['detected_diseases']
                        if detected:
                            for disease, prob in detected:
                                st.markdown(f"**{disease}**: {prob:.1%}")
                        else:
                            st.info("No diseases detected above threshold")
                    
                    with col2:
                        st.markdown("### All Probabilities")
                        probs = results['probabilities']
                        for disease, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                            color = "🔴" if prob > threshold else "🟢"
                            st.markdown(f"{color} **{disease}**: {prob:.1%}")
                    
                    # Show report text
                    st.subheader("📝 Generated Report")
                    st.text_area("Report Text", results['report_text'], height=200)
                    
                    # Download PDF button
                    if os.path.exists(output_pdf_path):
                        with open(output_pdf_path, 'rb') as pdf_file:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_file,
                                file_name=report_filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        # Save to session state
                        st.session_state.reports.append({
                            'filename': report_filename,
                            'path': output_pdf_path,
                            'timestamp': timestamp,
                            'results': results
                        })
                    
                    # Cleanup temp files
                    try:
                        os.unlink(input_path)
                        os.unlink(preprocessed_path)
                    except:
                        pass
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")

# ============================================================================
# PAGE 2: DOCTOR CONNECT
# ============================================================================

elif page == "👨‍⚕️ Doctor Connect":
    st.header("👨‍⚕️ Doctor Connect - Video Consultation")
    
    st.markdown("""
    ### Connect with a Doctor
    
    Use this section to connect with healthcare professionals for consultation about your X-ray results.
    """)
    
    # Simple chat interface for doctor consultation
    st.subheader("💬 Chat with Doctor")
    
    # Chat input
    user_message = st.chat_input("Type your message here...")
    
    if user_message:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['message'])
                st.caption(msg['timestamp'])
        else:
            with st.chat_message("assistant"):
                st.write(msg['message'])
                st.caption(msg['timestamp'])
    
    # Auto-response for demo (replace with real doctor connection)
    if user_message:
        # Simple auto-response
        responses = [
            "Thank you for your question. I'm reviewing your X-ray report. Could you provide more details?",
            "Based on the AI analysis, I recommend consulting with a radiologist. Would you like me to explain any specific findings?",
            "I can help clarify the results. Which disease or finding would you like to know more about?",
            "For a comprehensive consultation, please schedule an appointment. In the meantime, I can answer general questions about your report."
        ]
        import random
        response = random.choice(responses)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'message': response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        st.rerun()
    
    # Share reports section
    st.subheader("📄 Share Reports with Doctor")
    
    if st.session_state.reports:
        selected_report = st.selectbox(
            "Select a report to share",
            options=[r['filename'] for r in st.session_state.reports],
            format_func=lambda x: f"{x} ({st.session_state.reports[[r['filename'] for r in st.session_state.reports].index(x)]['timestamp']})"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📤 Share Report", use_container_width=True):
                idx = [r['filename'] for r in st.session_state.reports].index(selected_report)
                report_data = st.session_state.reports[idx]
                
                st.success(f"✅ Report '{selected_report}' shared with doctor!")
                st.info("💡 The doctor can now view your report and provide consultation.")
                
                # Show report summary
                with st.expander("📋 Report Summary"):
                    st.json({
                        'filename': report_data['filename'],
                        'detected_diseases': report_data['results']['detected_diseases'],
                        'timestamp': report_data['timestamp']
                    })
        
        with col_btn2:
            idx = [r['filename'] for r in st.session_state.reports].index(selected_report)
            report_data = st.session_state.reports[idx]
            if os.path.exists(report_data['path']):
                with open(report_data['path'], 'rb') as pdf_file:
                    st.download_button(
                        label="📥 Download",
                        data=pdf_file,
                        file_name=report_data['filename'],
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        st.info("No reports available. Please generate a report first from the Home page.")
    
    # Video call option (external link)
    st.markdown("---")
    st.subheader("📹 Video Call Option")
    st.markdown("""
    For live video consultation, you can use external video call services:
    - **Google Meet**: [Start a meeting](https://meet.google.com)
    - **Zoom**: [Join a meeting](https://zoom.us)
    - **Microsoft Teams**: [Start a call](https://teams.microsoft.com)
    
    Share your report PDF with the doctor during the call.
    """)

# ============================================================================
# PAGE 3: GENERATED REPORTS
# ============================================================================

elif page == "📋 Generated Reports":
    st.header("📋 Generated Reports History")
    
    if st.session_state.reports:
        st.markdown(f"**Total Reports:** {len(st.session_state.reports)}")
        
        for idx, report in enumerate(reversed(st.session_state.reports)):
            with st.expander(f"📄 {report['filename']} - {report['timestamp']}", expanded=(idx == 0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Detected Diseases")
                    if report['results']['detected_diseases']:
                        for disease, prob in report['results']['detected_diseases']:
                            st.markdown(f"- **{disease}**: {prob:.1%}")
                    else:
                        st.info("No diseases detected")
                    
                    st.markdown("### Report Text")
                    st.text_area("", report['results']['report_text'], height=150, key=f"text_{idx}")
                
                with col2:
                    if os.path.exists(report['path']):
                        with open(report['path'], 'rb') as pdf_file:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_file,
                                file_name=report['filename'],
                                mime="application/pdf",
                                key=f"download_{idx}"
                            )
                    
                    st.caption(f"Generated: {report['timestamp']}")
    else:
        st.info("No reports generated yet. Go to the Home page to analyze an X-ray image.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | AI-powered Chest X-ray Analysis</p>
    <p><small>For clinical decision support only; not a diagnostic device</small></p>
</div>
""", unsafe_allow_html=True)

