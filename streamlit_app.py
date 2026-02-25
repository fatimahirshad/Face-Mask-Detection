import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#eef2f3,#8e9eab);
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#1f2937;
}

.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.1);
    margin-bottom:25px;
}

.badge-mask {
    background:#22c55e;
    padding:10px 20px;
    border-radius:10px;
    color:white;
    font-weight:bold;
}

.badge-nomask {
    background:#ef4444;
    padding:10px 20px;
    border-radius:10px;
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Detection</div>", unsafe_allow_html=True)

st.write("Upload an image or use your camera to detect if a person is wearing a mask.")

# -----------------------------
# Load CNN Model
# -----------------------------
model = tf.keras.models.load_model("model/mask_classifier.h5")

IMG_SIZE = 224

def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    return img

# -----------------------------
# IMAGE UPLOAD SECTION
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📁 Upload Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess(image)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:

        label = "Without Mask"
        confidence = prediction
        st.markdown("<div class='badge-nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    else:

        label = "With Mask"
        confidence = 1 - prediction
        st.markdown("<div class='badge-mask'>✅ With Mask</div>", unsafe_allow_html=True)

    st.progress(float(confidence))

    st.write(f"Confidence: **{confidence:.2f}**")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# LIVE CAMERA SECTION
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📷 Live Camera Detection")

class MaskDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)

        processed = preprocess(pil_img)

        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            label = "No Mask"
            color = (0,0,255)
        else:
            label = "Mask"
            color = (0,255,0)

        cv2.putText(
            img,
            label,
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return img

webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetector,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("</div>", unsafe_allow_html=True)
