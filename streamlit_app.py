import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Classifier",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(120deg,#f5f7fa,#c3cfe2);
}

.title{
text-align:center;
font-size:40px;
font-weight:700;
color:#1f2937;
}

.subtitle{
text-align:center;
color:#6b7280;
margin-bottom:30px;
}

.card{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 8px 20px rgba(0,0,0,0.08);
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
st.markdown("<div class='subtitle'>Upload an image to check if a mask is worn</div>", unsafe_allow_html=True)

# -----------------------------
# Load ONNX Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

IMG_SIZE = 224

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):

    img = preprocess(image)

    pred = session.run(None, {input_name: img})[0][0]

    if pred > 0.5:
        return "Without Mask", float(pred)
    else:
        return "With Mask", float(1 - pred)


# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    label, conf = predict(image)

    if label == "With Mask":
        st.markdown("<div class='mask'>✅ With Mask</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    st.progress(conf)

    st.write(f"Confidence: **{conf:.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
