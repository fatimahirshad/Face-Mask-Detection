import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#667eea,#764ba2);
}

.title{
text-align:center;
font-size:45px;
font-weight:700;
color:white;
}

.subtitle{
text-align:center;
color:#e5e7eb;
margin-bottom:30px;
}

.card{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 10px 25px rgba(0,0,0,0.15);
}

.mask{
background:#22c55e;
color:white;
padding:10px;
border-radius:10px;
text-align:center;
font-weight:bold;
}

.nomask{
background:#ef4444;
color:white;
padding:10px;
border-radius:10px;
text-align:center;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to detect mask usage</div>", unsafe_allow_html=True)

# -----------------------------
# Load ONNX Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

IMG_SIZE = input_shape[2]

# -----------------------------
# Image Preprocess
# -----------------------------
def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))

    img = np.expand_dims(img, axis=0)

    return img

# -----------------------------
# Prediction
# -----------------------------
def predict(image):

    img = preprocess(image)

    outputs = session.run(None, {input_name: img})

    pred = outputs[0][0][0]

    if pred > 0.5:
        return "Without Mask", float(pred)
    else:
        return "With Mask", float(1 - pred)

# -----------------------------
# Upload UI
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        label, conf = predict(image)

    if label == "With Mask":
        st.markdown("<div class='mask'>✅ With Mask</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    st.progress(conf)

    st.write(f"Confidence: **{conf:.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
