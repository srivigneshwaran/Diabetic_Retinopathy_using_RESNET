import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model_path = "C:/_____PYHTON/python/retinopathy_diabetes/diabetic_retinopathy_model.h5"
model = load_model(model_path)


st.title("Diabetic Retinopathy Detection")


with st.sidebar:
    st.header("Diagnosis Stages")
    st.markdown("""
    - **No DR**: No Diabetic Retinopathy.
    - **Mild**: Early signs of Retinopathy.
    - **Moderate**: Retinopathy is progressing but not severe.
    - **Severe**: Signs of significant Retinopathy.
    - **Proliferative DR**: Advanced stage with potential vision loss.
    """)


uploaded_file = st.file_uploader("Upload an eye fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    
    try:
        image = load_img(uploaded_file, target_size=(224, 224))
        image_array = img_to_array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

        st.subheader("Prediction Result")
        st.write(f"**Diagnosis**: {labels[predicted_class]}")
        st.write(f"Confidence: {predictions[0][predicted_class]:.2f}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")


st.markdown("---")
st.write("This application is for educational purposes only. For a medical diagnosis, please consult a professional.")
    