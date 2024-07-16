import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

# Load the model and processor
processor = BlipProcessor.from_pretrained("image-captioning")
model = BlipForConditionalGeneration.from_pretrained("image-captioning")

# API URL and headers for social media caption generation
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}

# Function to query the API for social media captions
def query_social_media_caption(image_caption):
    payload = {"inputs": f"Generate a social media caption with emojis and hashtags for : {image_caption}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    response_data = response.json()

    # Check if the response data is a list and contains a dictionary with 'generated_text'
    if isinstance(response_data, list) and response_data:
        result = response_data[0]
        if 'generated_text' in result:
            return result['generated_text']
    
    # Return a default message if no valid caption is found
    return "No social media caption found."

# Function to initialize session state variables
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.caption = ""
        st.session_state.social_media_caption = ""
        st.session_state.initialized = True

# Function to clear all inputs and outputs
def clear_all():
    st.session_state.uploaded_file = None
    st.session_state.caption = ""
    st.session_state.social_media_caption = ""
    st.experimental_rerun()

# Function to generate image caption using BLIP model
def generate_image_caption(uploaded_file):
    raw_image = Image.open(uploaded_file).convert('RGB')

    # Process image with the model
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
