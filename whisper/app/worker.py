"""
Background worker for audio transcription and speaker diarization.

Runs Whisper transcription and pyannote diarization in a background thread,
updating a shared progress dictionary that the Streamlit UI polls.
"""

import os
import time
import tempfile
import subprocess
import threading
import traceback
import imageio_ffmpeg

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Inject the bundled ffmpeg binary directory into PATH so that openai-whisper,
# pydub, and any other library that shells out to "ffmpeg" can find it.
_ffmpeg_dir = os.path.dirname(_FFMPEG)
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Speed factor estimates (relative to audio duration) on CPU
# ---------------------------------------------------------------------------
MODEL_SPEED_FACTORS = {
    "tiny": 0.3,
    "base": 0.5,
    "small": 1.0,
    "medium": 2.0,
    "large": 4.0,
}

MODEL_DESCRIPTIONS = {
    "tiny": "Tiny  --  ~39 M params  |  fastest, lowest accuracy",
    "base": "Base  --  ~74 M params  |  fast, decent accuracy",
    "small": "Small  --  ~244 M params  |  balanced speed/accuracy",
    "medium": "Medium  --  ~769 M params  |  slow, good accuracy",
    "large": "Large  --  ~1550 M params  |  slowest, best accuracy",
}

LANGUAGES = {
    "auto": "Auto-detect",
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _new_progress() -> dict:
    """Return a fresh progress dictionary."""
    return {
        "step": "idle",
        "percent": 0,
        "eta_seconds": None,
        "segments_done": 0,
        "segments_total": 0,
        "log": [],
        "result": None,
        "error": None,
        "cancelled": False,
        "start_time": None,
    }


def _log(progress: dict, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    progress["log"].append(f"[{ts}] {msg}")


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                _FFMPEG, "-v", "quiet", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                "-i", audio_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_audio(video_path: str, progress: dict) -> str:
    """Extract audio from video file using ffmpeg. Returns path to WAV file."""
    progress["step"] = "extracting"
    progress["percent"] = 5
    _log(progress, "Extracting audio from video file...")

    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    cmd = [
        _FFMPEG, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

    progress["percent"] = 10
    duration = _get_audio_duration(audio_path)
    _log(progress, f"Audio extracted: {duration:.1f}s ({duration/60:.1f} min)")
    return audio_path


def _transcribe(
    audio_path: str,
    model_size: str,
    language: str,
    progress: dict,
) -> dict:
    """Run Whisper transcription with segment-level progress."""
    import whisper

    progress["step"] = "transcribing"
    progress["percent"] = 15
    _log(progress, f"Loading Whisper model '{model_size}'...")

    model = whisper.load_model(model_size)
    _log(progress, "Model loaded. Starting transcription...")

    duration = _get_audio_duration(audio_path)
    est_segments = max(1, int(duration / 30))
    progress["segments_total"] = est_segments

    speed_factor = MODEL_SPEED_FACTORS.get(model_size, 1.0)
    eta_total = duration * speed_factor
    progress["eta_seconds"] = eta_total
    progress["start_time"] = time.time()

    _log(progress, f"Estimated {est_segments} segments, ETA ~{eta_total/60:.1f} min")

    # Transcribe options
    opts = {"verbose": False, "fp16": False}
    if language != "auto":
        opts["language"] = language

    result = whisper.transcribe(model, audio_path, **opts)

    segments = result.get("segments", [])
    progress["segments_done"] = len(segments)
    progress["segments_total"] = len(segments)
    progress["percent"] = 60

    _log(progress, f"Transcription complete: {len(segments)} segments, language='{result.get('language', '?')}'")
    return result


def _diarize(audio_path: str, hf_token: str, progress: dict) -> object:
    """Run pyannote speaker diarization."""
    from pyannote.audio import Pipeline as PyannotePipeline

    progress["step"] = "diarizing"
    progress["percent"] = 65
    _log(progress, "Loading pyannote diarization pipeline...")

    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    _log(progress, "Pipeline loaded. Running diarization...")

    hook_data = {"total": 0, "completed": 0}

    def progress_hook(step_name, step_artefact, file, total=None, completed=None):
        if progress["cancelled"]:
            raise InterruptedError("Cancelled by user")
        if total and completed:
            hook_data["total"] = total
            hook_data["completed"] = completed
            pct = completed / total
            progress["percent"] = 65 + int(pct * 25)
            if completed % max(1, total // 10) == 0:
                _log(progress, f"Diarization: {step_name} {completed}/{total}")

    diarization = pipeline(audio_path, hook=progress_hook)

    progress["percent"] = 90
    num_speakers = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
    _log(progress, f"Diarization complete: {num_speakers} speakers detected")
    return diarization


def _merge_transcript_diarization(whisper_result: dict, diarization) -> list:
    """Merge Whisper segments with pyannote speaker labels."""
    merged = []

    for seg in whisper_result.get("segments", []):
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2.0

        # Find the speaker active at the segment midpoint
        speaker = "UNKNOWN"
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= seg_mid <= turn.end:
                speaker = label
                break

        merged.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"].strip(),
            "speaker": speaker,
        })

    return merged


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_short(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_plain_transcript(whisper_result: dict) -> str:
    """Format plain transcript with timestamps."""
    lines = []
    for seg in whisper_result.get("segments", []):
        ts = _format_time_short(seg["start"])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


def format_diarized_transcript(merged: list) -> str:
    """Format diarized transcript with speaker labels and timestamps."""
    lines = []
    current_speaker = None
    for entry in merged:
        ts = _format_time_short(entry["start"])
        if entry["speaker"] != current_speaker:
            current_speaker = entry["speaker"]
            lines.append(f"\n[{ts}] {current_speaker}:")
        lines.append(f"  [{ts}] {entry['text']}")
    return "\n".join(lines).strip()


def format_srt(whisper_result: dict, merged: Optional[list] = None) -> str:
    """Format as SRT subtitle file."""
    source = merged if merged else whisper_result.get("segments", [])
    lines = []
    for i, seg in enumerate(source, 1):
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def format_json_output(whisper_result: dict, merged: Optional[list] = None) -> str:
    """Format as JSON."""
    import json

    output = {
        "language": whisper_result.get("language", "unknown"),
        "segments": merged if merged else whisper_result.get("segments", []),
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def run_pipeline(
    file_bytes: bytes,
    file_name: str,
    model_size: str,
    language: str,
    enable_diarization: bool,
    hf_token: str,
    progress: dict,
):
    """
    Main pipeline entry point. Runs in a background thread.

    Updates `progress` dict in-place so the Streamlit UI can poll it.
    """
    progress.update(_new_progress())
    progress["step"] = "uploading"
    progress["percent"] = 0
    progress["start_time"] = time.time()
    _log(progress, f"Starting pipeline: file={file_name}, model={model_size}, lang={language}")

    tmp_dir = None
    try:
        # Save uploaded file
        tmp_dir = tempfile.mkdtemp(prefix="whisper_app_")
        input_path = os.path.join(tmp_dir, file_name)
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        progress["percent"] = 2
        _log(progress, f"File saved to temp directory ({len(file_bytes)/1024/1024:.1f} MB)")

        if progress["cancelled"]:
            raise InterruptedError("Cancelled by user")

        # Extract audio
        ext = file_name.rsplit(".", 1)[-1].lower()
        if ext in ("wav", "mp3", "flac", "ogg", "m4a"):
            audio_path = input_path
            _log(progress, "Audio file detected, skipping extraction")
            progress["percent"] = 10
        else:
            audio_path = _extract_audio(input_path, progress)

        if progress["cancelled"]:
            raise InterruptedError("Cancelled by user")

        # Transcribe
        whisper_result = _transcribe(audio_path, model_size, language, progress)

        if progress["cancelled"]:
            raise InterruptedError("Cancelled by user")

        # Diarize
        merged = None
        if enable_diarization:
            if not hf_token:
                _log(progress, "WARNING: No HuggingFace token provided, skipping diarization")
            else:
                merged = None
                diarization = _diarize(audio_path, hf_token, progress)
                merged = _merge_transcript_diarization(whisper_result, diarization)
        else:
            _log(progress, "Diarization disabled, skipping")
            progress["percent"] = 90

        # Build results
        progress["step"] = "done"
        progress["percent"] = 100
        progress["eta_seconds"] = 0

        elapsed = time.time() - progress["start_time"]
        _log(progress, f"Pipeline complete in {elapsed/60:.1f} minutes")

        progress["result"] = {
            "whisper_result": whisper_result,
            "merged": merged,
            "plain_transcript": format_plain_transcript(whisper_result),
            "diarized_transcript": format_diarized_transcript(merged) if merged else None,
            "srt": format_srt(whisper_result, merged),
            "json_output": format_json_output(whisper_result, merged),
            "language": whisper_result.get("language", "unknown"),
            "num_segments": len(whisper_result.get("segments", [])),
            "num_speakers": len(set(e["speaker"] for e in merged)) if merged else 0,
            "elapsed_seconds": elapsed,
        }

    except InterruptedError:
        progress["step"] = "cancelled"
        progress["error"] = "Processing cancelled by user"
        _log(progress, "Pipeline cancelled by user")

    except Exception as e:
        progress["step"] = "error"
        progress["error"] = str(e)
        _log(progress, f"ERROR: {e}")
        _log(progress, traceback.format_exc())

    finally:
        # Clean up temp files
        if tmp_dir:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def start_worker(
    file_bytes: bytes,
    file_name: str,
    model_size: str,
    language: str,
    enable_diarization: bool,
    hf_token: str,
    progress: dict,
) -> threading.Thread:
    """Start the pipeline in a background thread. Returns the thread."""
    t = threading.Thread(
        target=run_pipeline,
        args=(file_bytes, file_name, model_size, language, enable_diarization, hf_token, progress),
        daemon=True,
    )
    t.start()
    return t
