import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#fdfbfb,#ebedee);
}

.title{
text-align:center;
font-size:42px;
font-weight:700;
color:#1f2937;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:16px;
color:#4b5563;
margin-bottom:25px;
}

.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,0.08);
display:flex;
flex-direction:row;
align-items:center;
gap:20px;
}

.result-section{
min-width:150px;
padding:15px;
border-radius:15px;
background:#f3f4f6;
text-align:center;
}

.mask{
background:#22c55e;
color:white;
padding:8px 12px;
border-radius:10px;
font-weight:bold;
font-size:16px;
}

.nomask{
background:#ef4444;
color:white;
padding:8px 12px;
border-radius:10px;
font-weight:bold;
font-size:16px;
}

.confidence{
font-weight:bold;
font-size:18px;
color:#111827;
margin-top:8px;
}

.progress-bar{
height:15px;
border-radius:10px;
background:#e5e7eb;
}

.progress-fill{
height:100%;
border-radius:10px;
background: linear-gradient(to right, #4ade80, #16a34a);
text-align:right;
padding-right:5px;
color:white;
font-weight:bold;
font-size:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to detect if a person is wearing a mask</div>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_mask_detector_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH, compile=False)
IMG_SIZE = 224

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Unable to open this file. Please upload a valid image.")
        return None
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Predict Function
# -----------------------------
def predict(uploaded_file):
    img = preprocess(uploaded_file)
    if img is None:
        return None, None
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "Without Mask", float(prediction)
    else:
        return "With Mask", float(1 - prediction)

# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    label, confidence = predict(uploaded_file)
    if label is None:
        st.stop()  # stop execution if file invalid

    # Display prediction and confidence on left
    st.markdown("<div class='result-section'>", unsafe_allow_html=True)

    if label == "With Mask":
        st.markdown("<div class='mask'>✅ With Mask</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f"""
    <div class='confidence'>Confidence: {confidence:.2f}</div>
    <div class='progress-bar'>
        <div class='progress-fill' style='width:{confidence*100}%'>{int(confidence*100)}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Show uploaded image on the right
    st.image(Image.open(uploaded_file), use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)
