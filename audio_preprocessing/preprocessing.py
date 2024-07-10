# Import required libraries
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import shutil
import zipfile
import librosa
import matplotlib.pyplot as plt
import librosa
import os

emotions = {
    '01': 'neutral 😐',
    '02': 'happy 😄',
    '03': 'surprise 😲',
    '04': 'disgust 🤢',
    '05': 'disappointed 😤',
}

def extract_mfcc(file_path, n_mfcc=20, frame_length=0.025, hop_length=0.01, preemphasis=0.97):
    y, sr = librosa.load(file_path, sr=None)
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    
    frame_length = int(frame_length * sr)
    hop_length = int(hop_length * sr)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)
    mfccs_mean = np.mean(mfccs, axis=1)  # Taking mean of MFCCs over time
    return mfccs_mean

def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.show()
    st.pyplot(plt)

def plot_mel_spectrogram(y, sr):
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()
    st.pyplot(plt)

def classify_audio(file_path, model):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(f"Duration: {duration:.2f} seconds")
    
    if duration < 2:
        st.error("Audio duration is less than 3 seconds. Please upload a longer audio.")
        return
    
    y_segment = y[:3*sr]
    if len(y_segment) < 3*sr:
        y_segment = np.pad(y_segment, (0, 3*sr - len(y_segment)), 'constant')
    
    st.write("Waveform:")
    plot_waveform(y_segment, sr)
    
    st.write("Mel Spectrogram:")
    plot_mel_spectrogram(y_segment, sr)
    
    features = extract_mfcc(file_path, n_mfcc=20, frame_length=0.025, hop_length=0.01, preemphasis=0.97)
    
    st.write("MFCC Features:")
    st.dataframe(features.reshape(1, -1))  # Menampilkan MFCC secara horizontal

    
    emotion_code = model.predict([features])[0]
    print(f"Predicted Emotion Code: {emotion_code}")  # Debugging: Print predicted emotion code
    
    st.write(f"Predicted Emotion: {emotion_code}")




def process_audio_files_in_folder(folder_path, model):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('wav'))]
    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        st.write(f"Processing {audio_file}...")
        classify_audio(file_path, model)