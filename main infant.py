import streamlit as st
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import tensorflow as tf
import pickle
import time
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play

# Load trained model
MODEL_PATH = r"C:\Users\nawfa\Desktop\deepfake\infant_cry_model2.keras"
LABEL_ENCODER_PATH = r"C:\Users\nawfa\Desktop\deepfake\label_encoder2.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

CLASSES = label_encoder.classes_

# Function to extract spectrogram
def audio_to_spectrogram(file_path, target_shape=(128, 128)):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize spectrogram
    if mel_spec_db.shape[1] < target_shape[1]:  
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_shape[1] - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:  
        mel_spec_db = mel_spec_db[:, :target_shape[1]]

    return mel_spec_db

# Function to predict class
def predict_audio(file_path):
    spectrogram = audio_to_spectrogram(file_path)
    spectrogram = spectrogram.reshape(1, 128, 128, 1)

    predictions = model.predict(spectrogram)[0]
    class_probs = {CLASSES[i]: round(float(predictions[i] * 100), 2) for i in range(len(CLASSES))}
    final_class = CLASSES[np.argmax(predictions)]
    
    return final_class, class_probs

# Function to record audio
def record_audio(duration=5, sr=22050):
    st.info(f"Recording for {duration} seconds... Speak now! 🎤")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.int16)
    sd.wait()
    file_path = "recorded_audio.wav"
    write(file_path, sr, recording)
    return file_path

# Streamlit UI
st.set_page_config(page_title="Infant Cry Detector", page_icon="🔊", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #f8f9fa; }
    .stButton button { background-color: #ff6b6b; color: white; font-size: 18px; border-radius: 10px; }
    .stFileUploader { font-size: 16px; }
    .css-1aumxhk { background-color: #f1f3f4 !important; }
    .stTextInput, .stTextArea { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👶 Infant Cry Classification")
st.write("Analyze your baby's cry and determine the reason using AI!")

# Sidebar for input
st.sidebar.header("Choose an option:")
option = st.sidebar.radio("", ["🎤 Record Audio", "📁 Upload File"])

if option == "🎤 Record Audio":
    if st.button("Start Recording"):
        file_path = record_audio()
        st.success("Recording complete! ✅")
        st.audio(file_path, format="audio/wav")

        # Prediction
        result_class, class_probs = predict_audio(file_path)

        # Display result
        st.subheader(f"Prediction: **{result_class}** 🎯")
        st.write("### Class Probabilities:")
        st.write(class_probs)

elif option == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav", "mp3"])

    if uploaded_file:
        file_path = "uploaded_audio.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(file_path, format="audio/wav")

        # Prediction
        result_class, class_probs = predict_audio(file_path)

        # Display result
        st.subheader(f"Prediction: **{result_class}** 🎯")
        st.write("### Class Probabilities:")
        st.write(class_probs)

        # Spectrogram Visualization
        st.write("### Spectrogram of the Uploaded Audio:")
        spectrogram = audio_to_spectrogram(file_path)
        fig, ax = plt.subplots(figsize=(5, 3))
        img = librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", cmap="coolwarm")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<center>© 2025 Developed by Nawfal | AI-Powered Cry Detection</center>", unsafe_allow_html=True)
