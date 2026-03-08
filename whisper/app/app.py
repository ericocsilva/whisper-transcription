"""
Whisper Transcription & Speaker Diarization App

A Streamlit application for transcribing audio/video files using OpenAI Whisper
with optional speaker diarization via pyannote.audio.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from worker import (
    MODEL_DESCRIPTIONS,
    MODEL_SPEED_FACTORS,
    LANGUAGES,
    start_worker,
    _new_progress,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Whisper Transcription",
    page_icon="🎙️",  # noqa: RUF001
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS for a professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #666;
        font-size: 1rem;
        margin: 0;
    }

    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
    }
    .step-item {
        text-align: center;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .step-active {
        background-color: #1f77b4;
        color: white;
    }
    .step-done {
        background-color: #2ca02c;
        color: white;
    }
    .step-pending {
        background-color: #f0f0f0;
        color: #999;
    }

    /* Log output */
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        max-height: 250px;
        overflow-y: auto;
        margin: 1rem 0;
    }
    .log-container .log-line {
        margin: 2px 0;
    }

    /* Result card */
    .result-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        flex: 1;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "progress" not in st.session_state:
    st.session_state.progress = _new_progress()
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def get_current_step():
    return st.session_state.progress.get("step", "idle")


def is_processing():
    step = get_current_step()
    return step in ("uploading", "extracting", "transcribing", "diarizing")


def is_done():
    return get_current_step() == "done"


def is_error():
    return get_current_step() == "error"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Whisper Transcription & Diarization</h1>
        <p>Upload audio or video files for transcription with optional speaker identification</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Step indicator
# ---------------------------------------------------------------------------
def render_step_indicator():
    current = get_current_step()
    steps_order = ["idle", "uploading", "extracting", "transcribing", "diarizing", "done"]

    display_steps = [
        ("Upload", ["idle"]),
        ("Extract Audio", ["uploading", "extracting"]),
        ("Transcribe", ["transcribing"]),
        ("Diarize", ["diarizing"]),
        ("Results", ["done"]),
    ]

    if current == "idle":
        current_idx = 0
    elif current in ("uploading", "extracting"):
        current_idx = 1
    elif current == "transcribing":
        current_idx = 2
    elif current == "diarizing":
        current_idx = 3
    elif current == "done":
        current_idx = 4
    elif current == "error":
        current_idx = -1
    else:
        current_idx = 0

    html_parts = []
    for i, (label, _) in enumerate(display_steps):
        if current == "error":
            cls = "step-pending"
        elif i < current_idx:
            cls = "step-done"
        elif i == current_idx:
            cls = "step-active"
        else:
            cls = "step-pending"
        html_parts.append(f'<div class="step-item {cls}">{label}</div>')

    st.markdown(
        f'<div class="step-indicator">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )


render_step_indicator()


# ---------------------------------------------------------------------------
# UPLOAD & CONFIGURE (shown when idle or error)
# ---------------------------------------------------------------------------
if not is_processing() and not is_done():

    # Show previous error if any
    if is_error():
        st.error(f"Previous run failed: {st.session_state.progress.get('error', 'Unknown error')}")
        if st.button("Clear error and start over"):
            st.session_state.progress = _new_progress()
            st.rerun()

    col_upload, col_config = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("1. Upload File")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse for an audio/video file",
            type=["mp4", "mp3", "wav", "m4a", "flac", "ogg", "webm", "mkv", "avi", "mov"],
            help="Supported: MP4, MP3, WAV, M4A, FLAC, OGG, WebM, MKV, AVI, MOV",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"**{uploaded_file.name}** ({size_mb:.1f} MB)")

    with col_config:
        st.subheader("2. Configure")

        model_size = st.selectbox(
            "Whisper Model",
            options=list(MODEL_DESCRIPTIONS.keys()),
            index=1,  # default: base
            format_func=lambda x: MODEL_DESCRIPTIONS[x],
            help="Larger models are more accurate but significantly slower on CPU",
        )

        language = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            index=2,  # default: Portuguese
            format_func=lambda x: LANGUAGES[x],
            help="Select 'Auto-detect' to let Whisper identify the language",
        )

        enable_diarization = st.toggle(
            "Enable Speaker Diarization",
            value=True,
            help="Identify and label different speakers in the audio",
        )

        # HF token
        default_hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
        if enable_diarization:
            hf_token = st.text_input(
                "HuggingFace Token",
                value=default_hf_token,
                type="password",
                help="Required for pyannote speaker diarization. Get one at huggingface.co/settings/tokens",
            )
            if not hf_token:
                st.warning("A HuggingFace token is required for speaker diarization.")
        else:
            hf_token = default_hf_token

    st.divider()

    # Output configuration
    with st.expander("Output Options", expanded=False):
        save_directory = st.text_input(
            "Save to directory (optional)",
            placeholder="/Volumes/catalog/schema/volume/transcriptions/",
            help="Leave empty to only use browser download. On Databricks, use a Volume path.",
        )

    # Start button
    col_start, col_spacer = st.columns([1, 2])
    with col_start:
        can_start = uploaded_file is not None
        if enable_diarization and not hf_token:
            can_start = False

        if st.button(
            "Start Transcription",
            type="primary",
            disabled=not can_start,
            use_container_width=True,
        ):
            # Store config for later use
            st.session_state.save_directory = save_directory
            st.session_state.file_name = uploaded_file.name
            st.session_state.processing = True

            # Read file bytes and start worker
            file_bytes = uploaded_file.getvalue()
            st.session_state.worker_thread = start_worker(
                file_bytes=file_bytes,
                file_name=uploaded_file.name,
                model_size=model_size,
                language=language,
                enable_diarization=enable_diarization,
                hf_token=hf_token,
                progress=st.session_state.progress,
            )
            st.rerun()


# ---------------------------------------------------------------------------
# PROCESSING (shown while worker is running)
# ---------------------------------------------------------------------------
if is_processing():
    progress = st.session_state.progress
    step = progress["step"]
    pct = progress["percent"]

    # Progress bar
    st.progress(pct / 100, text=f"Step: **{step.capitalize()}** -- {pct}%")

    # ETA display
    eta_col1, eta_col2 = st.columns(2)
    with eta_col1:
        eta_secs = progress.get("eta_seconds")
        if eta_secs and eta_secs > 0:
            # Recalculate based on elapsed time and percent
            elapsed = time.time() - progress.get("start_time", time.time())
            if pct > 5:
                total_est = elapsed / (pct / 100)
                remaining = max(0, total_est - elapsed)
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                st.metric("Estimated Time Remaining", f"{mins}m {secs}s")
            else:
                mins = int(eta_secs // 60)
                st.metric("Estimated Time Remaining", f"~{mins} min")
        else:
            st.metric("Estimated Time Remaining", "Calculating...")

    with eta_col2:
        elapsed = time.time() - progress.get("start_time", time.time())
        if pct > 5:
            total_est = elapsed / (pct / 100)
            completion = datetime.now() + timedelta(seconds=max(0, total_est - elapsed))
            st.metric("Estimated Completion", completion.strftime("%H:%M:%S"))
        else:
            st.metric("Estimated Completion", "--:--:--")

    # Segment progress
    if step == "transcribing":
        seg_done = progress.get("segments_done", 0)
        seg_total = progress.get("segments_total", 0)
        if seg_total > 0:
            st.caption(f"Segments: {seg_done} / {seg_total}")

    # Log output
    log_lines = progress.get("log", [])
    if log_lines:
        log_html = "".join(f'<div class="log-line">{line}</div>' for line in log_lines[-20:])
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

    # Cancel button
    if st.button("Cancel", type="secondary"):
        progress["cancelled"] = True
        st.session_state.processing = False
        st.session_state.progress = _new_progress()
        st.rerun()

    # Auto-refresh
    time.sleep(1)
    st.rerun()


# ---------------------------------------------------------------------------
# RESULTS (shown when done)
# ---------------------------------------------------------------------------
if is_done():
    progress = st.session_state.progress
    result = progress.get("result", {})

    if not result:
        st.error("No result data available.")
    else:
        st.balloons()

        # Stats row
        elapsed = result.get("elapsed_seconds", 0)
        num_segments = result.get("num_segments", 0)
        num_speakers = result.get("num_speakers", 0)
        detected_lang = result.get("language", "?")

        cols = st.columns(4)
        cols[0].metric("Language", detected_lang.upper())
        cols[1].metric("Segments", str(num_segments))
        cols[2].metric("Speakers", str(num_speakers) if num_speakers > 0 else "N/A")
        cols[3].metric("Processing Time", f"{elapsed/60:.1f} min")

        st.divider()

        # Transcript display
        if result.get("diarized_transcript"):
            st.subheader("Diarized Transcript")
            st.text_area(
                "Full transcript with speaker labels",
                value=result["diarized_transcript"],
                height=400,
                label_visibility="collapsed",
            )
        else:
            st.subheader("Transcript")
            st.text_area(
                "Full transcript",
                value=result.get("plain_transcript", ""),
                height=400,
                label_visibility="collapsed",
            )

        st.divider()

        # Download buttons
        st.subheader("Downloads")
        dl_cols = st.columns(4)

        with dl_cols[0]:
            st.download_button(
                "Download .txt",
                data=result.get("plain_transcript", ""),
                file_name=f"{Path(st.session_state.get('file_name', 'transcript')).stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_cols[1]:
            st.download_button(
                "Download .srt",
                data=result.get("srt", ""),
                file_name=f"{Path(st.session_state.get('file_name', 'transcript')).stem}.srt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_cols[2]:
            st.download_button(
                "Download .json",
                data=result.get("json_output", "{}"),
                file_name=f"{Path(st.session_state.get('file_name', 'transcript')).stem}.json",
                mime="application/json",
                use_container_width=True,
            )

        with dl_cols[3]:
            if result.get("diarized_transcript"):
                st.download_button(
                    "Download diarized .txt",
                    data=result["diarized_transcript"],
                    file_name=f"{Path(st.session_state.get('file_name', 'transcript')).stem}_diarized.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.button("No diarization", disabled=True, use_container_width=True)

        # Save to directory
        save_dir = st.session_state.get("save_directory", "")
        if save_dir:
            st.divider()
            st.subheader("Save to Directory")
            if st.button(f"Save all files to: {save_dir}", type="primary"):
                try:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    stem = Path(st.session_state.get("file_name", "transcript")).stem

                    files_saved = []

                    txt_path = save_path / f"{stem}.txt"
                    txt_path.write_text(result.get("plain_transcript", ""), encoding="utf-8")
                    files_saved.append(str(txt_path))

                    srt_path = save_path / f"{stem}.srt"
                    srt_path.write_text(result.get("srt", ""), encoding="utf-8")
                    files_saved.append(str(srt_path))

                    json_path = save_path / f"{stem}.json"
                    json_path.write_text(result.get("json_output", "{}"), encoding="utf-8")
                    files_saved.append(str(json_path))

                    if result.get("diarized_transcript"):
                        dia_path = save_path / f"{stem}_diarized.txt"
                        dia_path.write_text(result["diarized_transcript"], encoding="utf-8")
                        files_saved.append(str(dia_path))

                    st.success("Files saved successfully:")
                    for fp in files_saved:
                        st.code(fp, language=None)

                except Exception as e:
                    st.error(f"Failed to save files: {e}")

        # Show processing log
        with st.expander("Processing Log", expanded=False):
            log_lines = progress.get("log", [])
            if log_lines:
                log_html = "".join(f'<div class="log-line">{line}</div>' for line in log_lines)
                st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

        # New transcription button
        st.divider()
        if st.button("Start New Transcription", type="primary"):
            st.session_state.progress = _new_progress()
            st.session_state.processing = False
            st.session_state.worker_thread = None
            st.rerun()
