# app.py
import streamlit as st
from pathlib import Path
import tempfile
import os
from typing import List

# model imports
import whisper
from transformers import pipeline
import imageio_ffmpeg  # ensures we can find ffmpeg binary if available

# --------------------- Page config ---------------------
st.set_page_config(page_title="Audio Transcriber & Summarizer", layout="wide")
st.title("🎧 Audio Transcriber & Summarizer")
st.write("Upload audio → transcribe with Whisper → summarize. Professional, local, and simple.")

# --------------------- Helper utilities ---------------------
@st.cache_resource
def get_whisper_model(size: str = "base"):
    return whisper.load_model(size)

@st.cache_resource
def get_summarizer_model(model_name: str = "facebook/bart-large-cnn"):
    return pipeline("summarization", model=model_name, device=-1)

def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def chunk_text_by_chars(text: str, max_chars: int = 1000) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        window = text[start:end]
        last_period = window.rfind('.')
        if last_period == -1 or end == n:
            chunks.append(text[start:end])
            start = end
        else:
            chunks.append(text[start:start + last_period + 1])
            start = start + last_period + 1
    return chunks

def summarize_long_text(summarizer, text: str, chunk_chars: int, min_len: int, max_len: int) -> str:
    chunks = chunk_text_by_chars(text, max_chars=chunk_chars)
    partials = []
    for c in chunks:
        if not c.strip():
            continue
        try:
            out = summarizer(c, max_length=max_len, min_length=min_len, do_sample=False)
            partials.append(out[0]['summary_text'])
        except Exception:
            partials.append(c[:chunk_chars])
    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]
    combined = "\n\n".join(partials)
    try:
        final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
        return final[0]['summary_text']
    except Exception:
        return combined

def ensure_ffmpeg_available():
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

# --------------------- Sidebar (controls) ---------------------
with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model size", options=["tiny", "base", "small", "medium", "large"], index=1)
    summarizer_model = st.text_input("Summarizer model (HF)", value="facebook/bart-large-cnn")
    max_summary_tokens = st.slider("Max summary tokens", 60, 400, 130)
    min_summary_tokens = st.slider("Min summary tokens", 10, 200, 30)
    chunk_chars = st.slider("Chunk size for summarization (chars)", 400, 4000, 1200, step=100)
    keep_files = st.checkbox("Save transcript/summary to outputs/ folder", value=True)
    st.markdown("---")
    st.info("Tip: choose a smaller Whisper model on CPU for faster processing.")

# --------------------- Main UI layout ---------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("1) Upload audio")
    uploaded_file = st.file_uploader("Supported: mp3, wav, m4a, mp4, ogg", type=["mp3", "wav", "m4a", "mp4", "ogg"])
    st.write("You can also drag-and-drop a file here.")

    if uploaded_file:
        st.audio(uploaded_file)
        st.markdown("---")
        ffmpeg_path = ensure_ffmpeg_available()
        if ffmpeg_path:
            st.caption(f"Using ffmpeg at: {ffmpeg_path}")
        else:
            st.caption("ffmpeg not found in Python env. If you get 'ffmpeg not found' install system ffmpeg or run pip install imageio[ffmpeg].")

with col_right:
    st.subheader("Actions")
    load_models_btn = st.button("Load models (optional)")
    transcribe_btn = st.button("Transcribe & Summarize")

# Model loading
if load_models_btn:
    with st.spinner("Loading models..."):
        try:
            _ = get_whisper_model(whisper_size)
            _ = get_summarizer_model(summarizer_model)
            st.success("Models loaded. Ready!")
        except Exception as e:
            st.error(f"Model loading failed: {e}")

# Containers for results
status = st.empty()
progress = st.empty()
results_container = st.container()

if transcribe_btn and uploaded_file:
    outdir = Path.cwd() / "outputs"
    if keep_files:
        outdir.mkdir(exist_ok=True)

    temp_path = save_uploaded_file(uploaded_file)

    try:
        status.info("Loading Whisper model...")
        whisper_model = get_whisper_model(whisper_size)
        progress.progress(10)

        status.info("Transcribing audio (Whisper)...")
        result = whisper_model.transcribe(temp_path)
        transcript = result.get("text", "").strip()
        progress.progress(50)

        if not transcript:
            status.error("Transcription produced no text.")
        else:
            with results_container:
                st.subheader("Transcript")
                st.text_area("Full transcript", transcript, height=300)

            status.info("Loading summarization model...")
            summarizer = get_summarizer_model(summarizer_model)
            status.info("Summarizing transcript (may take a while)...")
            progress.progress(70)

            summary = summarize_long_text(summarizer, transcript, chunk_chars, min_summary_tokens, max_summary_tokens)
            progress.progress(95)

            with results_container:
                st.subheader("Summary")
                st.write(summary)

            if keep_files:
                base = Path(uploaded_file.name).stem
                tfile = outdir / f"transcript_{base}.txt"
                sfile = outdir / f"summary_{base}.txt"
                tfile.write_text(transcript, encoding="utf-8")
                sfile.write_text(summary, encoding="utf-8")

            col1_dl, col2_dl = st.columns(2)
            with col1_dl:
                st.download_button("Download transcript (.txt)", transcript.encode("utf-8"), file_name=f"transcript_{Path(uploaded_file.name).stem}.txt", mime="text/plain")
            with col2_dl:
                st.download_button("Download summary (.txt)", summary.encode("utf-8"), file_name=f"summary_{Path(uploaded_file.name).stem}.txt", mime="text/plain")

            status.success("Done — transcript and summary are ready.")
            progress.progress(100)

    except Exception as e:
        status.error(f"An error occurred: {e}")
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

elif transcribe_btn and not uploaded_file:
    st.warning("Please upload an audio file before clicking 'Transcribe & Summarize'.")

# Show saved outputs
if Path.cwd().joinpath("outputs").exists():
    st.markdown("---")
    st.subheader("Saved outputs")
    out_files = list(Path.cwd().joinpath("outputs").glob("*"))
    if out_files:
        for f in sorted(out_files, key=os.path.getmtime, reverse=True)[:10]:
            st.write(f.name)

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.markdown("*Tips*")
st.sidebar.write("- For local CPU, choose 'tiny' or 'base' Whisper for speed.")
st.sidebar.write("- Use smaller summarizer models if CPU-only.")
st.sidebar.write("- If transformers import fails, check for a local file named transformers.py in this folder.")

#to run
#streamlit run transcriber.py

