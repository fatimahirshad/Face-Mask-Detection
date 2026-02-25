import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Face Mask Detection", layout="centered")

# -----------------------------
# UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#eef2f3,#8e9eab);
}

.title {
    text-align:center;
    font-size:38px;
    font-weight:700;
    color:#1f2937;
}

.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.1);
    margin-bottom:25px;
}

.mask {
    color:white;
    background:#22c55e;
    padding:8px 18px;
    border-radius:10px;
    font-weight:bold;
}

.nomask {
    color:white;
    background:#ef4444;
    padding:8px 18px;
    border-radius:10px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)

st.write("Upload an image or use your webcam to detect whether a person is wearing a mask.")

# -----------------------------
# Load ONNX Model
# -----------------------------
 # Absolute path to ONNX file
model_path = os.path.join(os.path.dirname(__file__), "best.onnx")
session = ort.InferenceSession(model_path)

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))  # CHW

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Predict Function
# -----------------------------
def predict(image):

    img = preprocess(image)

    pred = session.run(None, {input_name: img})[0][0]

    if pred > 0.5:
        return "Without Mask", pred
    else:
        return "With Mask", 1 - pred


# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📁 Upload Image")

uploaded_file = st.file_uploader("Choose image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    label, conf = predict(image)

    if label == "With Mask":
        st.markdown("<div class='mask'>✅ With Mask</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    st.progress(float(conf))

    st.write(f"Confidence: **{conf:.2f}**")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Webcam Detection
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📷 Live Camera Detection")


class MaskDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)

        label, conf = predict(pil_img)

        if label == "With Mask":
            color = (0,255,0)
        else:
            color = (0,0,255)

        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return img


webrtc_streamer(
    key="maskcam",
    video_processor_factory=MaskDetector,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("</div>", unsafe_allow_html=True)
