import json
import os
from pyannote.audio import Pipeline

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
AUDIO_FILE = "/Users/erico.silva/Whisper/recordings/pismo_databricks_20260223.wav"
TRANSCRIPT_JSON = "/Users/erico.silva/Whisper/transcriptions/pismo_databricks_20260223.json"
OUTPUT_FILE = "/Users/erico.silva/Whisper/transcriptions/pismo_databricks_20260223_diarized.txt"

print("Loading diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

print("Running diarization (this may take a while)...")
diarization = pipeline(AUDIO_FILE)

# Save raw diarization RTTM
rttm_path = "/Users/erico.silva/Whisper/transcriptions/pismo_databricks_20260223.rttm"
with open(rttm_path, "w") as f:
    diarization.write_rttm(f)
print(f"Saved RTTM to {rttm_path}")

# Load whisper transcript segments
with open(TRANSCRIPT_JSON) as f:
    transcript = json.load(f)

segments = transcript["segments"]

def get_speaker(start, end):
    overlap = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        o = min(turn.end, end) - max(turn.start, start)
        if o > 0:
            overlap[speaker] = overlap.get(speaker, 0) + o
    if not overlap:
        return "UNKNOWN"
    return max(overlap, key=overlap.get)

print("Aligning transcript with speakers...")
lines = []
current_speaker = None
current_text = []
current_start = None

for seg in segments:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()
    speaker = get_speaker(start, end)

    if speaker != current_speaker:
        if current_speaker and current_text:
            timestamp = f"[{int(current_start//60):02d}:{int(current_start%60):02d}]"
            lines.append(f"{timestamp} {current_speaker}: {' '.join(current_text)}")
        current_speaker = speaker
        current_text = [text]
        current_start = start
    else:
        current_text.append(text)

if current_speaker and current_text:
    timestamp = f"[{int(current_start//60):02d}:{int(current_start%60):02d}]"
    lines.append(f"{timestamp} {current_speaker}: {' '.join(current_text)}")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n\n".join(lines))

print(f"\nDiarized transcript saved to {OUTPUT_FILE}")
print(f"Total speaker turns: {len(lines)}")
