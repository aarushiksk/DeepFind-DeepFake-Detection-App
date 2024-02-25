import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load pre-trained MesoNet model
meso_model_path = 'meso_model.h5'
meso_model = tf.keras.models.load_model(meso_model_path)

# Function to classify image
def classify_image(image):
    # Preprocess image
    image = image.resize((256, 256))  # Resize image to match input shape
    image_array = img_to_array(image) / 255.0  # Convert image to array and normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Classify image using MesoNet
    prediction = meso_model.predict(image_array)[0][0]
    
    # Determine if it's a deepfake or not
    if prediction < 0.5:
        return "Real"
    else:
        return "Deepfake"

# Streamlit interface
st.title('DeepFind - Deepfake Detection App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    # Classify the image
    result = classify_image(image)
    st.write("Prediction:", result)
