import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# -------------------------
# Load Model
# -------------------------
model = YOLO("best.onnx", task="detect")

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.title("😷 Face Mask Detection")
st.write("Detect whether a person is wearing a mask or not.")

option = st.radio(
    "Choose Detection Mode",
    ["Upload Image", "Live Webcam"]
)

# -------------------------
# IMAGE UPLOAD
# -------------------------
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        results = model.predict(
            source=image_np,
            conf=0.25,
            imgsz=640
        )

        boxes = results[0].boxes
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Result", use_container_width=True)

        if boxes is not None:

            st.subheader("Predictions")

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = model.names[cls]

                st.write(f"**{label}** — Confidence: `{conf:.2f}`")

# -------------------------
# LIVE WEBCAM
# -------------------------
elif option == "Live Webcam":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.write("Camera not available")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(
            source=frame_rgb,
            conf=0.25,
            imgsz=640
        )

        annotated = results[0].plot()

        FRAME_WINDOW.image(annotated)

    else:
        camera.release()
