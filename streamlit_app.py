import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Face Mask Detection", layout="centered")

# -----------------------------
# Minimal UI Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg,#f5f7fa,#c3cfe2);
    }

    h1 {
        text-align:center;
        color:#1f3c88;
    }

    .upload-box {
        padding:20px;
        border-radius:10px;
        background:white;
        box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("😷 Face Mask Detection System")

st.write("Upload an image or use your camera to detect whether a person is wearing a mask.")

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("best.onnx", task="detect")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.subheader("📁 Upload Image")

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.predict(
        source=image_np,
        imgsz=640,
        conf=0.4
    )

    annotated = results[0].plot()

    st.image(annotated, caption="Prediction", use_container_width=True)

    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)

        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {conf:.2f}")

# -----------------------------
# CAMERA DETECTION
# -----------------------------
st.subheader("📷 Live Camera Detection")

class MaskDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, imgsz=640, conf=0.4)

        annotated = results[0].plot()

        return annotated


webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetector,
    media_stream_constraints={"video": True, "audio": False},
)
