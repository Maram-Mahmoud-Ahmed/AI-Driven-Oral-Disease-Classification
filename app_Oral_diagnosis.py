import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# -----------------------
# Configuration
# -----------------------
IMG_SIZE = 244
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(page_title="Dental AI Diagnoser", layout="centered")

def set_local_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.5); 
            border-radius: 20px;
            padding: 40px;
            max-width: 1100px; 
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* --- POSITION CONTROL: SHIFT EVERYTHING RIGHT --- */
        h1, h2, h3, .stMarkdown p, .stFileUploader, .stImage, .stAlert {{
            margin-left: 300px !important; 
            text-align: left !important;
            color: white !important;
        }}

        h1 {{
            margin-top: 300px !important;
            text-shadow: 4px 4px 16px #000000;
        }}

        .stMarkdown p {{
            margin-top: 10px !important;
            text-shadow: 2px 2px 4px #000000;
        }}

        .stFileUploader {{
            margin-top: 30px;
            max-width: 500px; 
        }}

        /* Keep the image from getting too huge when centered */
        .stImage > img {{
            max-width: 400px !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_local_background(r"C:\Users\alserag\Desktop\OpenGLProject\BackgroundOral.jpg")

# -----------------------
# Load Model (Cached)
# -----------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("dental_soft_tissue_baseline.keras")
        return model
    except Exception as e:
        st.error("Model loading failed.")
        return None

model = load_model()

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# --- UI DESIGN ---
st.title("🦷 Oral Diagnosis")
st.write("Upload a photo of a gum lesion to classify your diagnosis.")

# ------ Upload Section ---------
uploaded_file = st.file_uploader("Upload Dental Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image)

        if model is not None:
            with st.spinner("Analyzing image..."):
                processed = preprocess_image(image)
                start = time.time()
                preds = model.predict(processed)
                end = time.time()

                probs = preds[0]
                predicted_index = int(np.argmax(probs))
                predicted_class = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else "Unknown"
                confidence = probs[predicted_index]

            # --- Prediction Output ---
            st.subheader("Prediction Result")
            st.subheader(f"Predicted Condition: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2%}**")
            st.caption(f"Inference time: {end-start:.3f} seconds")

    except Exception as e:
        st.error("Image processing failed.")
        st.exception(e)