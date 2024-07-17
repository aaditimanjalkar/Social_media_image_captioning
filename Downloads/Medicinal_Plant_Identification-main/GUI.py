import streamlit as st
from transformers import pipeline
from PIL import Image
from medicinal_plant_info import medicinal_plant_info as mpi

# Function to initialize session state variables
def init_session_state():
    st.session_state.uploaded_file = None
    st.session_state.plant_name = ""
    st.session_state.info = ""

# Function to clear all inputs and outputs
def clear_all():
    st.session_state.uploaded_file = None
    st.session_state.plant_name = ""
    st.session_state.info = ""

# Function to classify the uploaded image using a local model
def classify_image(image):
    # Path to the local model
    model_path = "medicinal_plants_image_detection"  # Update this path to your local model path
    
    # Instantiate the image classification pipeline with the local model
    classifier = pipeline("image-classification", model=model_path)
    
    # Classify the image
    classification_result = classifier(image)
    
    # Return the classification results
    return classification_result

# Function to query information from the dictionary for the predicted class
def get_medicinal_plant_info(plant_name):
    info = mpi.get(plant_name, None)
    if info is not None:
        info_text = f"Scientific Name: {info['scientific_name']}\n"
        info_text += f"Use in Ayurveda: {info['use_in_ayurveda']}\n"
        info_text += f"Healing Properties: {info['healing_properties']}\n"
        info_text += f"Treatment: {info['treatment']}\n"
        return info_text
    else:
        return "No information found for the predicted plant."

# Main function to define the GUI layout
def main():
    st.title("Medicinal Plant Detection")

    # Initialize session state variables
    if 'initialized' not in st.session_state:
        init_session_state()
        st.session_state.initialized = True

    # Upload image field
    uploaded_file = st.file_uploader("Upload Image")

    # Update uploaded file in session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Display uploaded image
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to identify plant
    if st.button("Identify Plant"):
        if st.session_state.uploaded_file is not None:
            # Load the image from the uploaded file
            image = Image.open(st.session_state.uploaded_file)
            
            # Classify the image
            classification_result = classify_image(image)
            
            # Display the top classification result and all classification results
            results_text = ""
            top_result = classification_result[0]  # Highest probability class
            top_class = top_result['label']
            for result in classification_result:
                results_text += f"Predicted Class: {result['label']}, Confidence: {result['score'] * 100:.2f}%\n"
            
            # Update the plant name text area with all classification results
            st.session_state.plant_name = results_text
            
            # Get medicinal information using the highest probability class
            info_text = get_medicinal_plant_info(top_class)
            
            # Update session state info with retrieved text
            st.session_state.info = info_text
        else:
            st.error("Please upload an image first!")

    # Text area to display all classification results
    st.text_area("Plant Name", value=st.session_state.plant_name, height=200)

    # Text area to display medicinal information
    st.text_area("Medicinal Information", value=st.session_state.info, height=200)

    # Clear all button
    if st.button("Clear All"):
        clear_all()

if __name__ == "__main__":
    main()
