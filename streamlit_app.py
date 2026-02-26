import streamlit as st
import ultralytics
from ultralytics import YOLO
from PIL import Image
import numpy as np
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
color:white;
}

.main-title{
text-align:center;
font-size:45px;
font-weight:700;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
margin-bottom:30px;
opacity:0.9;
}

.card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 10px 25px rgba(0,0,0,0.15);
color:black;
}

.result-card{
padding:15px;
border-radius:12px;
margin-top:10px;
background:#f3f4f6;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI model detecting mask usage from images</div>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
model = YOLO(MODEL_PATH, task="detect")

# -----------------------------
# Upload Card
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):

        results = model.predict(
            source=img_array,
            imgsz=640,
            conf=0.25
        )

    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    st.subheader("Detection Summary")

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("No faces detected")

    else:

        for cls, conf in zip(boxes.cls, boxes.conf):

            label = results[0].names[int(cls)]
            confidence = float(conf)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.write(f"**Class:** {label}")

            st.progress(confidence)

            st.write(f"Confidence: {confidence:.2f}")

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
