import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
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
background: linear-gradient(135deg,#eef2f3,#8e9eab);
}

.title{
text-align:center;
font-size:40px;
font-weight:700;
color:#1f2937;
}

.subtitle{
text-align:center;
color:#4b5563;
margin-bottom:30px;
}

.card{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to detect face mask</div>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

model = YOLO(MODEL_PATH, task="detect")

# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_array = np.array(image)

    # Run detection
    results = model.predict(source=img_array, imgsz=640, conf=0.25)

    # Annotated image
    annotated = results[0].plot()

    st.image(annotated, use_container_width=True)

    # Show detected labels
    st.subheader("Detection Results")

    if len(results[0].boxes) == 0:
        st.write("No faces detected")

    else:
        for box, cls, conf in zip(
            results[0].boxes.xyxy,
            results[0].boxes.cls,
            results[0].boxes.conf
        ):
            class_name = results[0].names[int(cls)]
            st.write(f"**{class_name}** — Confidence: {conf:.2f}")

st.markdown("</div>", unsafe_allow_html=True)
