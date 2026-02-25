import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model
model = YOLO("best.pt")

st.title("Face Mask Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np)

    annotated = results[0].plot()

    st.image(annotated, caption="Prediction", use_column_width=True)
