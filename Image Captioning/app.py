import streamlit as st
from PIL import Image
from server import query_social_media_caption, init_session_state, clear_all, generate_image_caption

# Injecting custom CSS
st.markdown(
    """
    <style>
    /* Customize buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }

    /* Customize text areas */
    .stTextArea > textarea {
        background-color: #f0f0f0;
        color: black;
    }

    /* Customize file uploader */
    .stFileUploader > label {
        background-color: #2196F3;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }

    /* Customize sidebar */
    .stSidebar {
        background-color: #f0f0f0;
    }

    /* Customize headings */
    h1 {
        color: #2196F3;
        text-align: center;
    }

    h2 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():

    top_image = Image.open("Social Media Image Caption Generator.png")  # Change "path/to/your/image.jpg" to the actual path of your image
    st.image(top_image, use_column_width=True)

    # Title and introduction
    # st.title("Social Media Image Caption Generator")
    st.write("Generate image captions and social media captions with ease.")

    # Initialize session state variables
    init_session_state()

    # Display an image at the top of the page
    # Load the image
    # top_image = Image.open("captioning.jpg")
    # Display the image
    # st.image(top_image, caption="Top Image", use_column_width=True)

    # Create a sidebar for controls
    with st.sidebar:
        st.header("Controls")

        # Image uploader with tooltip
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, help="Drag and drop an image or browse to upload."
        )

        # Update uploaded file in session state
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

        # Button to generate image caption
        if st.button("Generate Caption", help="Generate an image caption for the uploaded image."):
            if st.session_state.uploaded_file:
                # Generate image caption and update session state
                caption = generate_image_caption(st.session_state.uploaded_file)
                st.session_state.caption = caption
                st.success("Image caption generated successfully!")
            else:
                st.error("Please upload an image first!")

        # Button to generate social media caption
        if st.button("Generate Social Media Caption", help="Generate a social media caption with emojis and hashtags."):
            if st.session_state.caption:
                # Generate social media caption and update session state
                social_media_caption = query_social_media_caption(st.session_state.caption)
                st.session_state.social_media_caption = social_media_caption
                st.success("Social media caption generated successfully!")
            else:
                st.error("Please generate an image caption first!")

        # Clear button
        if st.button("Clear All", help="Clear all inputs and outputs."):
            clear_all()
            st.success("Cleared all inputs and outputs.")

    # Display uploaded image in the main content area
    if st.session_state.uploaded_file:
        # Convert the uploaded file to an Image object
        uploaded_image = Image.open(st.session_state.uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Display image caption and social media caption in text areas
    st.text_area("Image Caption", value=st.session_state.caption, height=200, help="Generated caption for the uploaded image.")
    st.text_area("Social Media Caption", value=st.session_state.social_media_caption, height=200, help="Generated social media caption with emojis and hashtags.")

if __name__ == "__main__":
    main()
