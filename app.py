import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load Model
model = YOLO("yolo76.pt")

# Custom CSS for Styling with Decorative Background
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #f3e5f5, #e1bee7);
        background-size: cover;
        font-family: "Arial", sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom, #f8f0fc 0%, #e1bee7 100%);
        color: #4a148c;
    }
    h1, h2, h3, h4, h5 {
        text-align: center;
        color: #4a148c;
    .block-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff; /* White background for content */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stFileUploader {
        border: 2px dashed #7b1fa2; /* Dashed purple border */
        border-radius: 10px;
        background-color: #ffffff; /* White background */
        padding: 1rem;
        color: #7b1fa2; /* Medium purple text */
    }
    .stFileUploader:hover {
        border-color: #4a148c; /* Dark purple border on hover */
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 12px;
        color: #7b1fa2; /* Medium purple for footer */
    }
    .stButton>button {
        background-color: #7b1fa2;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4a148c;
    }
    .wave {
        position: absolute;
        bottom: 0;
        width: 100%;
        height: 150px;
        background: url('https://www.transparenttextures.com/patterns/wavecut.png');
        opacity: 0.2;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h2>Detection Cell Cancer Cervix With AI</h2>", unsafe_allow_html=True)
st.markdown("<h3>Upload your images to detect objects in real-time!</h3>", unsafe_allow_html=True)

# Upload Multiple Files
uploaded_files = st.file_uploader(
    "Drop and Drop Images Here", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:  # Process uploaded files
    for uploaded_file in uploaded_files[:7]:  # Limit to 4 images
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.markdown(f"<h3>Uploaded Image: {uploaded_file.name}</h3>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)

            # Perform Detection
            results = model(image, iou=0.35, conf=0.3)  # Perform inference with custom IOU and confidence
            result_image = np.array(results[0].plot())  # Get and plot the first detection result

            # Display detected objects
            st.markdown(f"<h3>Detection Result: {uploaded_file.name}</h3>", unsafe_allow_html=True)
            st.image(result_image, use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

# Add Decorative Footer
st.markdown("""
    <div class="footer">
        <p>Made by Neural Nexus | Powered by YOLOv8 & Streamlit</p>
    </div>
    <div class="wave"></div>
""", unsafe_allow_html=True)
