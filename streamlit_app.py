import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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
background: linear-gradient(120deg,#fdfbfb,#ebedee);
}

.title{
text-align:center;
font-size:40px;
font-weight:700;
color:#111827;
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

st.markdown("<div class='title'>😷 Face Mask Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to check if a person is wearing a mask</div>", unsafe_allow_html=True)

# -----------------------------
# Load PyTorch Model
# -----------------------------
model = torch.load("best.onnx", map_location=torch.device("cpu"))
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    prob = torch.sigmoid(output)[0][0].item()

    if prob > 0.5:
        return "Without Mask", prob
    else:
        return "With Mask", 1 - prob


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
