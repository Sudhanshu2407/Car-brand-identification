import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# Set up the Streamlit app
st.set_page_config(page_title="Car Brand Identification", page_icon="🚗", layout="wide")

# Sidebar configuration
st.sidebar.title("Car Brand Identification")
st.sidebar.write("Upload an image of a car, and the app will predict the car brand using a pre-trained model.")

# Load the pre-trained model (replace 'car_identification_model.h5' with your model's path)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('car_identification_model.h5')
    return model

model = load_model()

# Preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((128, 128))  # Resize the image to the size expected by the model (128x128)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Image upload section
st.header("Upload Car Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img_array = preprocess_image(uploaded_file)
    
    # Perform prediction
    prediction = model.predict(img_array)
    
    # Display raw prediction probabilities
    st.write("Raw prediction probabilities:", prediction)
    
    # Assuming the model's output is a one-hot encoded vector
    class_names = ["Audi","Hyundai Creta","Mahindra Scorpio","Rolls Royce","Swift","Tata Safari","Toyota Innova"]  # Replace with actual brand names
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display prediction
    st.subheader(f"Predicted Brand: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Optionally, display the image with the predicted label overlay
    img = np.array(Image.open(uploaded_file))
    img_with_text = cv2.putText(img.copy(), predicted_class, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    st.image(img_with_text, caption=f'Prediction: {predicted_class}', use_column_width=True)
else:
    st.write("Please upload an image to get a prediction.")
