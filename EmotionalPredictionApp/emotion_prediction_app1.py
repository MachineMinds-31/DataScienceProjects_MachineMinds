import streamlit as st
import tensorflow as tf
import numpy as np
from joblib import load
import librosa

# Function to extract audio features (Replace this with your feature extraction logic)
# Function to extract audio features (Replace this with your feature extraction logic)
def predict_emotion(audio_file_path):
    # Load the model
    loaded_model = tf.keras.models.load_model('C:\\Users\\ryzen\\Desktop\\Shhyam\\College_project\\streamlit\\optimized_model.h5')
    # Load the encoder
    enc = load('C:\\Users\\ryzen\\Desktop\\Shhyam\\College_project\\streamlit\\encoder.joblib')

    # Extract features
    X_test = np.expand_dims(extract_features(audio_file_path), -1)
    X_test = np.expand_dims(X_test, 0)

    # Make predictions
    pred = loaded_model.predict(X_test)
    y_pred = enc.inverse_transform(pred)

    return y_pred.flatten()[0]  # Return the predicted emotion

def extract_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-container {
        padding: 2rem;
        background-color: #f0f0f0;
        border-radius: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .title {
        font-size: 3rem;
        color: #2a6df4;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .button {
        font-size: 1.3rem;
        padding: 1rem 3rem;
        background-color: #2a9df4;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .button:hover {
        background-color: #007acc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app with custom styling and interactive widgets
def main():
    st.title("Audio Emotion Prediction")
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an audio file (MP3 or WAV)", type=['mp3', 'wav'])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3', start_time=0)

        if st.button("Predict Emotion", key='predict_button', help="Click to predict emotion"):
            with st.spinner('Predicting...'):
                predicted_emotion = predict_emotion(uploaded_file)
                st.success(f'Predicted Emotion: {predicted_emotion}')

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
