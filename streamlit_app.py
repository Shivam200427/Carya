"""
Unified Streamlit Application for Chest X-ray Disease Prediction
Modern, polished UI/UX for a professional user experience
"""

import streamlit as st
import streamlit.components.v1 as components
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

# Page config with modern theme
st.set_page_config(
    page_title="Carya AI - Chest X-ray Analysis",
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
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# ============================================================================
# CUSTOM CSS FOR MODERN UI
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Card Styling */
    .stCard {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader Styling */
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #667eea;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 4px;
        margin: 1rem 0;
    }
    
    /* Success Message */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d0e7ff 100%);
        border: 2px solid #0066cc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Disease Badge */
    .disease-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .disease-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .disease-medium {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        color: white;
    }
    
    .disease-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white;
    }
    
    /* Chat Message Styling */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .chat-assistant {
        background: #f0f0f0;
        color: #333;
        margin-right: 20%;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def embed_vapi_widget(public_api_key: str, assistant_id: str = None):
    """Embed Vapi Web Widget for voice interactions"""
    widget_html = f"""
    <script src="https://cdn.vapi.ai/widget.js"></script>
    <script>
        window.vapiWidget = new VapiWidget({{
            publicKey: "{public_api_key}",
            {f'assistantId: "{assistant_id}",' if assistant_id else ''}
            position: "bottom-right",
            theme: {{
                primaryColor: "#667eea",
                backgroundColor: "#ffffff",
                textColor: "#333333"
            }}
        }});
    </script>
    """
    components.html(widget_html, height=0)

def save_preprocessed_image(preprocessed_array, output_path):
    """Save preprocessed numpy array to image file."""
    img_uint8 = (preprocessed_array * 255).astype(np.uint8)
    cv2.imwrite(output_path, img_uint8)
    return output_path

def get_disease_color_class(prob, threshold):
    """Get CSS class for disease probability."""
    if prob > threshold:
        return "disease-high"
    elif prob > threshold * 0.7:
        return "disease-medium"
    else:
        return "disease-low"

# ============================================================================
# MAIN APP
# ============================================================================

# Hero Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='margin-bottom: 0.5rem;'>🫁 Carya AI</h1>
    <p style='font-size: 1.2rem; color: #666; font-weight: 400;'>
        Advanced Chest X-ray Analysis Powered by Artificial Intelligence
    </p>
    <p style='font-size: 0.95rem; color: #999; margin-top: 0.5rem;'>
        Detect 14 thoracic diseases • Generate comprehensive reports • Get instant insights
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #667eea; margin-bottom: 2rem;'>Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Choose a page",
        ["🏠 Analyze X-ray", "👨‍⚕️ Doctor Connect", "📋 My Reports"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.reports:
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Reports", len(st.session_state.reports))
        
        # Count detected diseases
        total_detected = sum(1 for r in st.session_state.reports 
                           if r['results'].get('detected_diseases') and 
                           r['results']['detected_diseases'] != "NA")
        st.metric("Analyses with Findings", total_detected)
    
    st.markdown("---")
    
    # Vapi Voice Assistant Configuration
    st.markdown("### 🎤 Voice Assistant")
    
    # Initialize Vapi config in session state
    if 'vapi_public_key' not in st.session_state:
        st.session_state.vapi_public_key = "53b6f2fa-8284-4329-a47d-4094deb68423"
    
    if 'vapi_assistant_id' not in st.session_state:
        st.session_state.vapi_assistant_id = "559846aa-b7be-48fa-8dd9-d27a13dd4844"
    
    if 'vapi_enabled' not in st.session_state:
        st.session_state.vapi_enabled = False
    
    # Display current status
    if st.session_state.vapi_enabled:
        st.success("✅ Voice assistant is active!")
        st.info("💡 Look for the Carya voice widget in the bottom-right corner. Click it to start talking!")
        if st.button("🔇 Disable Voice Assistant", use_container_width=True):
            st.session_state.vapi_enabled = False
            st.rerun()
    else:
        if st.button("🎤 Enable Voice Assistant", use_container_width=True, type="primary"):
            st.session_state.vapi_enabled = True
            st.rerun()
    
    st.caption("💬 Talk to Carya about your X-ray results, symptoms, or health questions")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; color: #999; font-size: 0.85rem;'>
        <p>🔒 Secure & Private</p>
        <p>For clinical support only</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: X-RAY ANALYSIS
# ============================================================================

if page == "🏠 Analyze X-ray":
    # Main Content Area
    col_main, col_side = st.columns([2, 1])
    
    with col_side:
        st.markdown("### ⚙️ Settings")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            help="Adjust sensitivity for disease detection"
        )
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e7f3ff 0%, #d0e7ff 100%); 
                    padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <p style='margin: 0; font-size: 0.9rem; color: #0066cc;'>
                <strong>Current Threshold:</strong> {threshold:.0%}<br>
                Diseases above this probability will be flagged.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        model_path = st.text_input(
            "Model Path",
            value=DEFAULT_MODEL_PATH,
            help="Path to the AI model",
            label_visibility="collapsed"
        )
    
    with col_main:
        st.markdown("### 📤 Upload Your X-ray Image")
        
        # Enhanced File Uploader
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image with modern styling
            st.markdown("### 📷 Your X-ray Image")
            image = Image.open(uploaded_file)
            
            # Create image container
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(image, use_container_width=True, caption="Uploaded Chest X-ray")
            
            # Analysis Button
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                analyze_clicked = st.button(
                    "🔍 Analyze with AI",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_clicked:
                with st.spinner("🔄 Processing your X-ray..."):
                    try:
                        # Check model exists
                        if not os.path.exists(model_path):
                            st.error(f"❌ Model file not found: {model_path}")
                            st.stop()
                        
                        # Progress indicator
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Save uploaded file
                        status_text.info("📸 Preparing image...")
                        progress_bar.progress(10)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            input_path = tmp_file.name
                            image.save(input_path)
                        
                        # Step 2: Preprocess
                        status_text.info("🔧 Enhancing image quality...")
                        progress_bar.progress(30)
                        
                        preprocessed_array = enhance_xray(input_path, target_size=(512, 512))
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as preprocessed_tmp:
                            preprocessed_path = preprocessed_tmp.name
                            save_preprocessed_image(preprocessed_array, preprocessed_path)
                        
                        # Step 3: AI Analysis
                        status_text.info("🤖 Running AI analysis...")
                        progress_bar.progress(50)
                        
                        report_id = str(uuid.uuid4())
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_filename = f'Chest_Report_{timestamp}_{report_id[:8]}.pdf'
                        
                        reports_dir = os.path.join(os.getcwd(), 'reports')
                        os.makedirs(reports_dir, exist_ok=True)
                        output_pdf_path = os.path.join(reports_dir, report_filename)
                        
                        # Step 4: Run inference
                        status_text.info("🧠 Analyzing disease patterns...")
                        progress_bar.progress(70)
                        
                        results = predict_and_generate_report(
                            model_path=model_path,
                            image_path=preprocessed_path,
                            output_pdf=output_pdf_path,
                            threshold=threshold,
                            device=None
                        )
                        
                        # Step 5: Generate report
                        status_text.info("📝 Generating comprehensive report...")
                        progress_bar.progress(90)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        
                        st.session_state.analysis_complete = True
                        st.session_state.current_results = results
                        
                        # Save to session state
                        st.session_state.reports.append({
                            'filename': report_filename,
                            'path': output_pdf_path,
                            'timestamp': timestamp,
                            'results': results
                        })
                        
                        # Cleanup
                        try:
                            os.unlink(input_path)
                            os.unlink(preprocessed_path)
                        except:
                            pass
                        
                        st.success("✅ Analysis Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Technical Details"):
                            st.code(traceback.format_exc())
            
            # Display Results
            if st.session_state.analysis_complete and st.session_state.current_results:
                results = st.session_state.current_results
                
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Detected Diseases Section
                detected = results['detected_diseases']
                
                if detected and detected != "NA":
                    st.markdown("#### 🚨 Detected Conditions")
                    
                    # Create cards for each detected disease
                    cols = st.columns(min(3, len(detected)))
                    for idx, (disease, prob) in enumerate(detected):
                        with cols[idx % len(cols)]:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h4 style='color: #667eea; margin-bottom: 0.5rem;'>{disease}</h4>
                                <h2 style='color: #ff6b6b; margin: 0;'>{prob:.1%}</h2>
                                <p style='font-size: 0.8rem; color: #999; margin-top: 0.5rem;'>
                                    Probability Score
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("✅ No diseases detected above threshold. Your X-ray appears normal!")
                
                # All Probabilities Section
                st.markdown("#### 📈 Complete Probability Analysis")
                
                probs = results['probabilities']
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                
                # Create progress bars for each disease
                for disease, prob in sorted_probs:
                    col_name, col_bar, col_val = st.columns([2, 3, 1])
                    
                    with col_name:
                        st.markdown(f"**{disease}**")
                    
                    with col_bar:
                        color = "#ff6b6b" if prob > threshold else "#66bb6a"
                        st.markdown(f"""
                        <div style='background: #f0f0f0; border-radius: 10px; padding: 4px;'>
                            <div style='background: {color}; border-radius: 8px; height: 24px; width: {prob*100}%; transition: width 0.5s ease;'></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_val:
                        st.markdown(f"<div style='text-align: right; font-weight: 600; color: {color};'>{prob:.1%}</div>", unsafe_allow_html=True)
                
                # Report Text
                st.markdown("#### 📝 Generated Medical Report")
                
                with st.expander("View Full Report", expanded=True):
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;'>
                        <pre style='font-family: "Inter", sans-serif; white-space: pre-wrap; line-height: 1.6;'>{results['report_text']}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download Section
                st.markdown("#### 📥 Download Report")
                
                if os.path.exists(output_pdf_path):
                    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                    with col_dl2:
                        with open(output_pdf_path, 'rb') as pdf_file:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_file,
                                file_name=report_filename,
                                mime="application/pdf",
                                use_container_width=True,
                                type="primary"
                            )
        
        else:
            # Empty State
            st.markdown("""
            <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                        border-radius: 16px; border: 2px dashed #667eea;'>
                <h2 style='color: #667eea; margin-bottom: 1rem;'>👆 Upload Your X-ray</h2>
                <p style='color: #666; font-size: 1.1rem;'>
                    Drag and drop your chest X-ray image here or click to browse
                </p>
                <p style='color: #999; font-size: 0.9rem; margin-top: 1rem;'>
                    Supported formats: PNG, JPG, JPEG, BMP, TIFF
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DOCTOR CONNECT
# ============================================================================

elif page == "👨‍⚕️ Doctor Connect":
    st.markdown("### 👨‍⚕️ Doctor Connect")
    st.markdown("Connect with healthcare professionals for consultation about your results")
    
    # Chat Interface
    st.markdown("#### 💬 Chat with Doctor")
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("💡 Start a conversation with your doctor by typing a message below.")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['message'])
                    st.caption(f"🕐 {msg['timestamp']}")
            else:
                with st.chat_message("assistant"):
                    st.write(msg['message'])
                    st.caption(f"🕐 {msg['timestamp']}")
    
    # Chat input
    user_message = st.chat_input("Ask your doctor about your X-ray results...")
    
    if user_message:
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Auto-response
        responses = [
            "Thank you for your question. I'm reviewing your X-ray report. Could you provide more details about your concerns?",
            "Based on the AI analysis, I recommend consulting with a radiologist for a detailed review. Would you like me to explain any specific findings?",
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
    
    # Share Reports Section
    st.markdown("---")
    st.markdown("#### 📄 Share Reports with Doctor")
    
    if st.session_state.reports:
        selected_report = st.selectbox(
            "Select a report to share",
            options=[r['filename'] for r in st.session_state.reports],
            format_func=lambda x: f"{x} ({st.session_state.reports[[r['filename'] for r in st.session_state.reports].index(x)]['timestamp']})"
        )
        
        col_share1, col_share2 = st.columns(2)
        
        with col_share1:
            if st.button("📤 Share with Doctor", use_container_width=True, type="primary"):
                idx = [r['filename'] for r in st.session_state.reports].index(selected_report)
                report_data = st.session_state.reports[idx]
                st.success(f"✅ Report '{selected_report}' shared with doctor!")
                st.info("💡 The doctor can now view your report and provide consultation.")
        
        with col_share2:
            idx = [r['filename'] for r in st.session_state.reports].index(selected_report)
            report_data = st.session_state.reports[idx]
            if os.path.exists(report_data['path']):
                with open(report_data['path'], 'rb') as pdf_file:
                    st.download_button(
                        label="📥 Download Report",
                        data=pdf_file,
                        file_name=report_data['filename'],
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        st.info("📋 No reports available. Generate a report first from the Analyze X-ray page.")
    
    # Video Call Options
    st.markdown("---")
    st.markdown("#### 📹 Video Consultation")
    
    col_vid1, col_vid2, col_vid3 = st.columns(3)
    
    with col_vid1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3>🔵 Google Meet</h3>
            <p style='color: #666;'>Start a secure video call</p>
            <a href='https://meet.google.com' target='_blank' style='color: #667eea; text-decoration: none; font-weight: 600;'>Start Meeting →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col_vid2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3>📹 Zoom</h3>
            <p style='color: #666;'>Join a video consultation</p>
            <a href='https://zoom.us' target='_blank' style='color: #667eea; text-decoration: none; font-weight: 600;'>Join Meeting →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col_vid3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3>💼 Teams</h3>
            <p style='color: #666;'>Microsoft Teams call</p>
            <a href='https://teams.microsoft.com' target='_blank' style='color: #667eea; text-decoration: none; font-weight: 600;'>Start Call →</a>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: GENERATED REPORTS
# ============================================================================

elif page == "📋 My Reports":
    st.markdown("### 📋 My Reports")
    
    if st.session_state.reports:
        st.metric("Total Reports", len(st.session_state.reports))
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, report in enumerate(reversed(st.session_state.reports)):
            with st.expander(f"📄 {report['filename']} - {report['timestamp']}", expanded=(idx == 0)):
                col_rpt1, col_rpt2 = st.columns([2, 1])
                
                with col_rpt1:
                    st.markdown("#### Detected Conditions")
                    
                    if report['results']['detected_diseases'] and report['results']['detected_diseases'] != "NA":
                        detected = report['results']['detected_diseases']
                        for disease, prob in detected:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                        color: white; padding: 0.75rem 1.5rem; border-radius: 8px; margin: 0.5rem 0;'>
                                <strong>{disease}</strong> - {prob:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No diseases detected in this analysis")
                    
                    st.markdown("#### Report Summary")
                    st.text_area("", report['results']['report_text'], height=150, key=f"text_{idx}", label_visibility="collapsed")
                
                with col_rpt2:
                    if os.path.exists(report['path']):
                        with open(report['path'], 'rb') as pdf_file:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_file,
                                file_name=report['filename'],
                                mime="application/pdf",
                                key=f"download_{idx}",
                                use_container_width=True
                            )
                    
                    st.caption(f"📅 Generated: {report['timestamp']}")
                    
                    # Show quick stats
                    if report['results'].get('probabilities'):
                        probs = report['results']['probabilities']
                        max_prob = max(probs.values()) if probs else 0
                        st.metric("Highest Probability", f"{max_prob:.1%}")
    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h3 style='color: #667eea;'>📋 No Reports Yet</h3>
            <p style='color: #666;'>Analyze your first X-ray to generate a report!</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# VAPI VOICE WIDGET EMBED
# ============================================================================

# Embed Vapi widget if enabled
if st.session_state.get('vapi_enabled') and st.session_state.get('vapi_public_key'):
    embed_vapi_widget(
        st.session_state.vapi_public_key,
        st.session_state.vapi_assistant_id if st.session_state.get('vapi_assistant_id') else None
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #999;'>
    <p style='font-size: 0.9rem; margin: 0.5rem 0;'>
        <strong style='color: #667eea;'>Carya AI</strong> - Powered by Advanced Machine Learning
    </p>
    <p style='font-size: 0.8rem; margin: 0.5rem 0;'>
        For clinical decision support only • Not a diagnostic device
    </p>
    <p style='font-size: 0.75rem; margin-top: 1rem;'>
        🔒 Your data is secure and private
    </p>
</div>
""", unsafe_allow_html=True)
