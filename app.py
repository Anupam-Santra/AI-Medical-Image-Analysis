import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import time

# Ensure the 'src' directory is in the path so we can import your custom model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import build_model

# --- Configuration & Styling ---
st.set_page_config(page_title="AI Pneumonia Detector", page_icon="🩺", layout="centered")

# --- Model Loading (Cached for speed) ---
# show_spinner=False disables the tiny default loader so we can build a better one
@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the model once and caches it in memory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'pneumonia_cnn_model.pth')
    
    if not os.path.exists(model_path):
        return None, f"Model weights not found at {model_path}. Please train the model first."
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

# --- Startup Loading Screen ---
# This prominent spinner will ONLY show up on the very first load while PyTorch boots up.
with st.spinner("⚙️ Initializing AI Engine & Loading Model Weights (This happens once)..."):
    model, device_or_error = load_model()

# --- Preprocessing Pipeline ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image, model, device):
    """Runs the image through the CNN and returns the diagnosis and confidence."""
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img)
        prediction_prob = torch.sigmoid(output).item()
        
    confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
    label = "PNEUMONIA" if prediction_prob > 0.5 else "NORMAL"
    
    return label, confidence

# --- User Interface ---
st.title("🏥 AI-Powered Medical Image Analysis")
st.markdown("Upload a Chest X-Ray (JPEG/PNG) below to receive an instant, AI-assisted preliminary diagnosis.")

if isinstance(model, type(None)):
    st.error(device_or_error)
else:
    # File Uploader
    uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Immediate feedback when the image finishes uploading
        with st.spinner('Preparing uploaded image...'):
            image = Image.open(uploaded_file).convert('RGB')
            
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)

        st.markdown("---")
        
        # Analyze Button
        if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
            
            # Multi-step Loading Status for the prediction phase
            with st.status("🩺 Running AI Diagnostic Pipeline...", expanded=True) as status:
                st.write("1. Applying medical image filters...")
                time.sleep(0.3) # Tiny artificial delay so the user can read the step
                
                st.write("2. Extracting features via Convolutional Neural Network...")
                # Run the actual heavy lifting here
                label, confidence = predict(image, model, device_or_error)
                
                st.write("3. Calculating confidence scores...")
                time.sleep(0.3)
                
                status.update(label="Diagnostic Complete!", state="complete", expanded=False)
                
            # Display Results visually
            st.subheader("Analysis Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if label == "PNEUMONIA":
                    st.error(f"🩺 **Diagnosis:** {label}")
                else:
                    st.success(f"✅ **Diagnosis:** {label}")
                    
            with res_col2:
                st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
            
            # Progress bar for visual confidence representation
            st.progress(float(confidence), text="AI Certainty Level")
            
            st.caption("*Disclaimer: This tool is a simulation and should act as a 'second pair of eyes'. It does not replace professional medical advice.*")