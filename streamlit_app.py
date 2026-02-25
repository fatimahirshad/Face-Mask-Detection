import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load ONNX model
model = YOLO("best.onnx", task="detect")

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run inference
    results = model.predict(image_np)

    # Draw results
    annotated = results[0].plot()

    st.image(annotated, caption="Prediction", use_column_width=True)
