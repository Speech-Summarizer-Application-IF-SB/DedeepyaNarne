import streamlit as st
from datetime import datetime
import threading
import time
import queue
import base64

PARTIALS_QUEUE = queue.Queue()
RECORDING_EVENT = threading.Event()

st.set_page_config(page_title="Meeting Summarizer — Live", layout="wide", page_icon="📝")

st.markdown(
    """
    <style>
    .reportview-container, .main {
      background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
      color: #e6eef8;
      font-family: 'Inter', sans-serif;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(2,6,23,0.6);
      margin-bottom: 12px;
    }
    .status-badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; }
    .status-idle { background: #94a3b8; color: #041125; }
    .status-recording { background: #ef4444; color: white; }
    .status-processing { background: #f59e0b; color: white; }
    .status-ready { background: #10b981; color: white; }
    .muted { color: #9aa9bf; font-size:13px; }
    h1,h2,h3 { color: #e6eef8; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "status" not in st.session_state:
    st.session_state.status = "idle"
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "diarized" not in st.session_state:
    st.session_state.diarized = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "meta" not in st.session_state:
    st.session_state.meta = {"session_id": None}
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
    st.session_state.logs = st.session_state.logs[-300:]

def _worker_put_chunks():
    i = 1
    while RECORDING_EVENT.is_set():
        PARTIALS_QUEUE.put(f"Speaker {1 if i % 2 else 2}: mock chunk #{i}")
        i += 1
        time.sleep(1.0)

def backend_start_recording(title: str, opts: dict):
    st.session_state.meta["session_id"] = f"mock-{int(time.time())}"
    RECORDING_EVENT.set()
    threading.Thread(target=_worker_put_chunks, daemon=True).start()
    return {"session_id": st.session_state.meta["session_id"], "started_at": datetime.now().isoformat()}

def backend_stop_recording(session_id: str):
    RECORDING_EVENT.clear()
    time.sleep(0.15)
    return {"wav_path": None, "session_id": session_id}

def backend_diarize_and_summarize(wav, transcript, opts):
    lines = [l for l in transcript.splitlines() if l.strip()]
    diarized = "\n".join([f"[Speaker {1 if i % 2 else 2}] {l}" for i, l in enumerate(lines)])
    summary = ("**TL;DR:** Mock meeting summary.\n\n"
               "**Decisions:**\n- Move to testing.\n\n"
               "**Action Items:**\n- Assign testers.\n")
    time.sleep(0.8)
    return {"diarized": diarized, "summary": summary}

def backend_export_markdown(results):
    md = f"# Meeting Summary\n\n{results.get('summary','')}\n\n```\n{results.get('diarized','')}\n```"
    return md.encode("utf-8")

def backend_export_pdf(results):
    return (results.get("summary","") + "\n\n" + results.get("diarized","")).encode("utf-8")

def backend_send_email(to, subject, body, attachment=None, attachment_name=None):
    log(f"email to={to}")
    time.sleep(0.3)
    return {"status": "sent"}

header1, header2 = st.columns([4,1])
with header1:
    st.markdown("<h1>📝 Live Meeting Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Real-time transcript • diarization • summary</div>", unsafe_allow_html=True)
with header2:
    badge = {
        "idle":"status-idle",
        "recording":"status-recording",
        "processing":"status-processing",
        "ready":"status-ready"
    }.get(st.session_state.status, "status-idle")
    st.markdown(f"<div style='text-align:right'><span class='status-badge {badge}'>{st.session_state.status.upper()}</span></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
col_left, col_mid, col_right = st.columns([1.6,2.2,1.4])
with col_left:
    meeting_title = st.text_input("Meeting title", st.session_state.meta.get("title","Weekly Sync"))
    participants = st.text_input("Participants", st.session_state.meta.get("participants","Alice,Bob"))
    meeting_date = st.date_input("Meeting date", datetime.now().date())
with col_mid:
    st.markdown("<b>Controls</b>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Start"):
            if st.session_state.status != "recording":
                st.session_state.status = "recording"
                st.session_state.meta.update({"title": meeting_title, "participants": participants})
                backend_start_recording(meeting_title, {})
                log("recording started")
    with c2:
        if st.button("⏸ Pause"):
            log("pause")
    with c3:
        if st.button("⏹ Stop"):
            if st.session_state.status == "recording":
                st.session_state.status = "processing"
                wav_info = backend_stop_recording(st.session_state.meta.get("session_id"))
                st.session_state.meta.update(wav_info)
                log("recording stopped")
with col_right:
    st.selectbox("STT", ["Vosk","Whisper","WhisperX","Groq"])
    st.selectbox("Diarization", ["pyannote","spectral"])
    uploaded_file = st.file_uploader("Upload audio to summarize", type=["wav","mp3","m4a","aac","ogg"])
    if uploaded_file is not None:
        st.markdown(f"Uploaded: {uploaded_file.name}", unsafe_allow_html=True)
        if st.button("Process Upload"):
            data = uploaded_file.read()
            mock_lines = []
            for i in range(1,6):
                mock_lines.append(f"Speaker {1 if i%2 else 2}: Transcribed line {i} from {uploaded_file.name}")
            st.session_state.live_text = "\n".join(mock_lines) + ("\n" + st.session_state.live_text if st.session_state.live_text else "")
            st.session_state.status = "processing"
            res = backend_diarize_and_summarize(None, st.session_state.live_text, {})
            st.session_state.diarized = res["diarized"]
            st.session_state.summary = res["summary"]
            st.session_state.status = "ready"
            log(f"processed upload {uploaded_file.name}")
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("<div class='card'><h3>Live Transcript</h3>", unsafe_allow_html=True)
    while not PARTIALS_QUEUE.empty():
        try:
            chunk = PARTIALS_QUEUE.get_nowait()
            st.session_state.live_text += chunk + "\n"
        except queue.Empty:
            break
    st.text_area("Live", st.session_state.live_text, height=420)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.status == "processing":
        with st.spinner("Processing"):
            res = backend_diarize_and_summarize(None, st.session_state.live_text, {})
            st.session_state.diarized = res["diarized"]
            st.session_state.summary = res["summary"]
            st.session_state.status = "ready"
            log("processed")
with col2:
    st.markdown("<div class='card'><h3>Summary</h3>", unsafe_allow_html=True)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>Export</h4>", unsafe_allow_html=True)
    if st.button("Export Markdown"):
        md = backend_export_markdown({"summary": st.session_state.summary, "diarized": st.session_state.diarized})
        st.download_button("Download Markdown", md, file_name="summary.md")
    if st.button("Export PDF"):
        pdf = backend_export_pdf({"summary": st.session_state.summary, "diarized": st.session_state.diarized})
        st.download_button("Download PDF", pdf, file_name="summary.pdf")
    email_to = st.text_input("Email", "you@example.com")
    subj = st.text_input("Subject", f"Summary — {meeting_title}")
    if st.button("Send Email"):
        backend_send_email(email_to, subj, st.session_state.summary.encode("utf-8"))
        st.success("Email sent")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><h3>Logs</h3></div>", unsafe_allow_html=True)
st.text_area("Logs", "\n".join(st.session_state.logs[-120:]), height=160)
