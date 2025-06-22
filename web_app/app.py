# app.py

import streamlit as st
import torch
import matplotlib.pyplot as plt
from cvae_model import CVAE
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
@st.cache_resource
def load_model():
    model = CVAE().to(device)
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval()
    return model

model = load_model()

# One-hot encoder
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

# Generate digit images
def generate_images(digit, num_samples=5):
    with torch.no_grad():
        y = one_hot(torch.tensor([digit] * num_samples))
        z = torch.randn(num_samples, 20).to(device)
        images = model.decoder(z, y).view(-1, 28, 28).cpu().numpy()
        return images

# --- Streamlit Interface ---
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit}")
    gen_imgs = generate_images(digit)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(gen_imgs[i], width=100, caption=f"Sample {i+1}")
