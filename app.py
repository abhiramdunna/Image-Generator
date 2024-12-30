import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv
import os
import time
import io

# Load environment variables
load_dotenv()

# Fetch token and model name from environment variables
token = os.getenv("TOKEN")
client_model_name = os.getenv("CLIENT_MODEL_NAME")

if not token or not client_model_name:
    st.error("Token or client model name is missing in the environment variables.")
    st.stop()

# Initialize Hugging Face Inference Client
client = InferenceClient(token=token)

# Streamlit UI
st.title("Image Generation Developed By Abhi")
st.write("Enter a description to generate an image. Generally, it takes 30 sec to generate.")

# User input for the description
new_description = st.text_input("Enter a description:", "")

if st.button("Generate Image"):
    if new_description.strip():
        # Progress bar visualization
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.03)  # Simulate progress
            progress_bar.progress(i)

        try:
            # Call the Hugging Face API to generate the image
            st.info("Generating image, please wait...")
            image_data = client.text_to_image(new_description, model=client_model_name)

            # Convert raw image data to displayable format
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption=f"Generated Image: {new_description}", use_column_width=True)
            st.success(f"Image successfully generated for: {new_description}")

        except Exception as e:
            st.error(f"Error generating image: {e}")
    else:
        st.warning("Please enter a valid description.")
