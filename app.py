import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load model
model = load_model("expression_model.h5")

# Class labels (change if needed)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Title
st.title("😐 Facial Expression & Emotion Detector")

# Camera input
img_file = st.camera_input("📸 Take a selfie to detect emotion")

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale & resize
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    resized = np.expand_dims(resized, axis=(0, -1))  # (1, 48, 48, 1)

    # Predict
    prediction = model.predict(resized)
    predicted_class = labels[np.argmax(prediction)]

    st.success(f"🎯 Emotion Detected: **{predicted_class}**")
