import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load Model
model = YOLO("yolo76.pt")

st.title("Object Detection App")
st.write("Upload an image to detect objects.")

# Upload File
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform Detection
    results = model(image)
    result = results[0]  # Ambil hasil deteksi pertama
    result_image = np.array(result.plot())
    
    st.image(result_image, caption="Detected Objects", use_column_width=True)
