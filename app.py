import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier.h5")

class_labels = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
# Define a function for preprocessing the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match the model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define a function for making predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability
    confidence = np.max(predictions)  # Get the highest probability
    return predicted_class, confidence

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify if it contains a brain tumor.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    with st.spinner("Classifying..."):
        predicted_class, confidence = predict(image)
    
    # Display results
    if predicted_class == 2:
        st.success(f"The model predicts: **No Brain Tumor**") #  with confidence {confidence:.2%}
    else:
        st.error(f"The model predicts: {class_labels[predicted_class]}") #  with confidence {confidence:.2%}
