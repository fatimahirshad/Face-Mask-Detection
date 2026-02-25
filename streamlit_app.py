import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Load ONNX YOLO model explicitly as a detection model
model = YOLO("best.onnx", task="detect")

st.title("Face Mask Detection")
st.write("Upload an image, and the model will detect faces with or without masks.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image and convert to numpy array
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run inference
    results = model.predict(
        source=image_np,
        imgsz=640,   # resize to model input size
        conf=0.25,   # confidence threshold
        show=False
    )

    # Annotated image with bounding boxes
    annotated_img = results[0].plot()

    # Display annotated image
    st.image(annotated_img, caption="Prediction", use_column_width=True)

    # Display detected classes and confidence
    st.write("### Detected Objects:")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            class_name = results[0].names[int(cls)]
            st.write(f"- {class_name} (Confidence: {conf:.2f})")
    else:
        st.write("No faces detected.")
