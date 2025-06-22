import streamlit as st
import torch
from model import ConditionalVAE
from utils import generate_digit_images
import matplotlib.pyplot as plt

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate"):
    model = ConditionalVAE()
    model.load_state_dict(torch.load("vae_mnist.pth", map_location="cpu"))
    model.eval()

    images = generate_digit_images(model, digit, n_samples=5)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
