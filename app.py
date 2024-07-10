import base64
import streamlit as st
import plotly.express as px
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import shutil
import zipfile
import librosa
import matplotlib.pyplot as plt
from audio_preprocessing.preprocessing import classify_audio, process_audio_files_in_folder
from model.model import load_model
from streamlit_option_menu import option_menu

# Load the SVM model
model = load_model('model/svm_mfcc_onlyy.pkl')

# CSS for background images and custom styles
@st.cache_data 
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
background-attachment: fixed;

}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Emotion Classification from Audio")
# st.sidebar.header("Configuration")

df = px.data.iris()

# Tambahkan footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #555;
            z-index: 1000;
        }
    </style>
    <div class="footer">
        © 2024 4C PPDM
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'temp_folder_path' not in st.session_state:
    st.session_state.temp_folder_path = None
if 'recording_path' not in st.session_state:
    st.session_state.recording_path = None

# Sidebar with sections
with st.sidebar:
    menu = option_menu(
        menu_title="Menu",
        options= ["Home", "Check Audio", "Check Audio by Folder", "Check Record Audio"],
        icons=["house", "music-note-list", "file-earmark-music", "mic"]
    )
    
if menu == "Home":
    st.header("Selamat Datang")
    
    # Add content for the Home menu
    
    st.write("Information")
    # st.image("tabel.png")
    # Tabel deskripsi faktor
    st.markdown("""
        <div class="white-background-table">
        <table>
            <thead>
                <tr>
                    <th>Identifier</th>
                    <th>Description of factor levels</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Actors</td>
                    <td>(01-10) actors, odd numbers are male actors, while even numbers are female actors</td>
                </tr>
                <tr>
                    <td>Emotion Class</td>
                    <td>(01) neutral, (02) happy, (03) surprise, (04) disgust, (05) disappointed</td>
                </tr>
                <tr>
                    <td>Intensity</td>
                    <td>(01) normal, (02) strong</td>
                </tr>
                <tr>
                    <td>Repetition</td>
                    <td>(01) first repetition, (02) second repetition, (03) third repetition</td>
                </tr>
            </tbody>
        </table>
        </div>
    """, unsafe_allow_html=True)

elif menu == "Check Audio":
    st.header("Check Audio")
    
    # Upload Audio File
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.session_state.temp_file_path = os.path.join("temp_audio", uploaded_file.name)
        os.makedirs("temp_audio", exist_ok=True)
        with open(st.session_state.temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(st.session_state.temp_file_path, format=uploaded_file.type)
    
        if st.button("Process Audio") and st.session_state.temp_file_path:
            classify_audio(st.session_state.temp_file_path, model)
            os.remove(st.session_state.temp_file_path)
            st.session_state.temp_file_path = None
            
elif menu == "Check Audio by Folder":
    st.header("Check Audio by Folder")
    
    # Upload Audio Folder
    uploaded_folder = st.file_uploader("Upload Audio Folder", type=["zip"], accept_multiple_files=False)
    if uploaded_folder is not None:
        st.session_state.temp_folder_path = "temp_folder"
        os.makedirs(st.session_state.temp_folder_path, exist_ok=True)
        with zipfile.ZipFile(uploaded_folder, 'r') as zip_ref:
            zip_ref.extractall(st.session_state.temp_folder_path)
    
        if st.button("Process Folder") and st.session_state.temp_folder_path:
            process_audio_files_in_folder(st.session_state.temp_folder_path, model)
            shutil.rmtree(st.session_state.temp_folder_path)
            st.session_state.temp_folder_path=None
            
elif menu == "Check Record Audio":
    st.header("Record Audio")
    duration = st.slider("Select recording duration (seconds)", min_value=2, max_value=4, value=3)
    fs = 44100  # Sample rate

    def record_audio(duration, fs):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        return recording

    if st.button("Start Recording"):
        st.write("Recording...")
        recording = record_audio(duration, fs)
        st.write("Recording finished.")
    
        st.session_state.recording_path = "recording.wav"
        write(st.session_state.recording_path, fs, recording)
        st.audio(st.session_state.recording_path, format='audio/wav')

    if st.session_state.recording_path and st.button("Process Recorded Audio"):
        classify_audio(st.session_state.recording_path, model)
        os.remove(st.session_state.recording_path)
        st.session_state.recording_path = None

