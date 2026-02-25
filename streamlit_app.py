import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Load ONNX model explicitly as detection
model = YOLO("best.onnx", task="detect")

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Predict
    results = model.predict(
        source=image_np,
        imgsz=640,
        conf=0.25,
        save=False,
        show=False
    )

    # Annotate image
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Prediction", use_column_width=True)

    # Show classes and confidence
    st.write("### Detected Objects:")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            class_name = results[0].names[int(cls)]
            st.write(f"- {class_name} (Confidence: {conf:.2f})")
    else:
        st.write("No faces detected.")
