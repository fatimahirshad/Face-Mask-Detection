import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Classifier",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(120deg,#fdfbfb,#ebedee);
}

.title{
text-align:center;
font-size:42px;
font-weight:700;
color:#111827;
}

.subtitle{
text-align:center;
font-size:18px;
color:#6b7280;
margin-bottom:30px;
}

.card{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 8px 20px rgba(0,0,0,0.08);
}

.result-mask{
background:#22c55e;
color:white;
padding:10px;
border-radius:10px;
text-align:center;
font-weight:bold;
}

.result-nomask{
background:#ef4444;
color:white;
padding:10px;
border-radius:10px;
text-align:center;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>😷 Face Mask Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to detect if a person is wearing a mask</div>", unsafe_allow_html=True)

# -----------------------------
# Load CNN Model
# -----------------------------
model = tf.keras.models.load_model("mask_model.h5")

IMG_SIZE = 224

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image)/255.0

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):

    img = preprocess(image)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Without Mask", prediction
    else:
        return "With Mask", 1 - prediction


# -----------------------------
# Upload UI
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence = predict(image)

    if label == "With Mask":
        st.markdown("<div class='result-mask'>✅ With Mask</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-nomask'>❌ Without Mask</div>", unsafe_allow_html=True)

    st.progress(float(confidence))

    st.write(f"Confidence: **{confidence:.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
