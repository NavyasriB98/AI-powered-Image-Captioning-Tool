import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_image_to_text():
    # Load the image-to-text pipeline
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

st.title("Image to Text Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")++
    st.image(image, use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            image_to_text_pipe = load_image_to_text()
            result = image_to_text_pipe(image)
            caption = result[0]['generated_text']

        st.subheader("Generated Caption")
        st.write(caption)
