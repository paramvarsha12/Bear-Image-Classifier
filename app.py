import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import os

model = load_model('models/model.h5')
class_names = sorted(os.listdir('data/train'))

st.title(" Bear Image Classifier")
st.write("Upload an image of a bear and I’ll tell you what kind it is!")

uploaded_file = st.file_uploader("Upload an image of a bear", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"Predicted Bear Species: **{predicted_class}**")
