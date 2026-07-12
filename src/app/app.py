"""
app.py

A Streamlit web app for classifying satellite images into 10 land-use
categories using our trained CNN model.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import SatelliteCNN
from utils import preprocess_image

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "satellite_cnn.pth")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

st.set_page_config(page_title="Satellite Image Analysis", page_icon="🛰️", layout="centered")
st.title("🛰️ Satellite Image Analysis with AI")
st.write(
    "Upload a satellite image and this app will classify it into one of "
    "10 land-use categories using a custom-trained CNN."
)

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    tensor = preprocess_image(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_idx = torch.argmax(probabilities).item()
    top_class = CLASS_NAMES[top_idx]
    top_confidence = probabilities[top_idx].item() * 100

    st.subheader("Prediction")
    st.success(f"**{top_class}** ({top_confidence:.1f}% confidence)")

    st.subheader("Confidence for all classes")
    for i, class_name in enumerate(CLASS_NAMES):
        confidence = probabilities[i].item() * 100
        st.write(f"{class_name}")
        st.progress(int(confidence))
        st.caption(f"{confidence:.1f}%")