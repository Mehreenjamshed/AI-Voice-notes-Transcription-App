import os
import streamlit as st
import whisper
from transformers import pipeline
from datetime import datetime

# Load models
st.title("🎤 AI Voice Note Organizer")
st.caption("Upload your voice, transcribe it, summarize it, and download your notes.")

# Load Whisper
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# Load Summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Upload audio
uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Save file temporarily
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp_audio")

    # Transcribe
    with st.spinner("Transcribing with Whisper..."):
        result = model.transcribe("temp_audio")
        transcription = result["text"]
        st.success("✅ Transcription Complete")
        st.subheader("📄 Transcription")
        st.write(transcription)

    # Summarize
    with st.spinner("Summarizing..."):
        max_input = 1000
        chunks = [transcription[i:i+max_input] for i in range(0, len(transcription), max_input)]
        summary = ""
        for chunk in chunks:
            sum_out = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summary += sum_out[0]["summary_text"] + " "

        st.success("✅ Summary Ready")
        st.subheader("📝 Summary")
        st.write(summary.strip())

    # Download Buttons
    st.subheader("📥 Download Notes")
    transcript_file = f"Transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_file = f"Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    st.download_button("⬇️ Download Transcript", transcription, file_name=transcript_file)
    st.download_button("⬇️ Download Summary", summary.strip(), file_name=summary_file)

    # Optional: Save transcript/summary to memory (can be extended to DB)
