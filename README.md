# Face Mask Detection

This project implements a **face mask detection system** using a Convolutional Neural Network (CNN) and computer vision.

## Features

- Detects whether a person is wearing a mask or not  
- Upload images to check mask status  
- Shows confidence score with a visual progress bar  
- Modern and responsive web interface using Streamlit  

## Dataset

The model was trained on a dataset containing images of:

- With Mask  
- Without Mask
- [Dataset Link](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)  

## Model

Model used: **Convolutional Neural Network (CNN)**  

Classes:  
- with_mask  
- without_mask  

## Technologies Used

- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow  

## Demo

Try the live app here: [Face Mask Detection App](https://face-mask-detection-cqlyzvgjmmgsbxwkzyypty.streamlit.app/)  

Check out the code: [GitHub Repository](https://github.com/fatimahirshad/Face-Mask-Detection)  

## Installation

```bash
git clone https://github.com/fatimahirshad/Face-Mask-Detection.git
cd Face-Mask-Detection
pip install -r requirements.txt
streamlit run streamlit_app.py
