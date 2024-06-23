import streamlit as st
import tensorflow.keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2
import numpy as np

# โหลดโมเดล
model = load_model('model.h5')

# ฟังก์ชันสำหรับการทำนาย
def prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0

    # ทำการทำนาย
    prediction = model.predict(np.expand_dims(img, axis=0))

    # รับผลลัพธ์ของคลาส
    class_label = np.argmax(prediction)

    return class_label

# ส่วนของ Streamlit
st.title("AI CLOUD Classification")
st.header("...")
st.write("Upload your Image...")

uploaded_file = st.file_uploader("Choose a .jpg pic...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(imgRGB, channels="RGB")

    st.write("")
    st.write("Classifying...")

    pred = prediction(image)

    st.success(f'The image is classified as class: {pred}')
