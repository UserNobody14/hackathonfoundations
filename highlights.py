import json, math
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

# ---------- Structured output ----------
class TimeStampSegment(BaseModel):
    start_time: float = Field(..., ge=0, description="Start time in seconds, ≥ 0")
    end_time: float = Field(..., gt=0, description="End time in seconds, > 0")

class ImportantSegments(BaseModel):
    segments: List[TimeStampSegment] = Field(..., min_items=1, max_items=15)

# ---------- Helpers ----------
def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS,mmm format to seconds."""
    time_part, ms_part = timestamp.split(',')
    h, m, s = time_part.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000

def load_transcript_from_json(json_file_path: str) -> tuple[str, str, float]:
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    if not data or "text" not in data[0]:
        raise ValueError("First JSON item must be full transcript with 'text' key")

    full_transcript = data[0]["text"].strip()
    formatted_segments = []
    max_end = 0.0

    for seg in data[1:]:
        if "start" in seg and "end" in seg and "text" in seg:
            start = timestamp_to_seconds(seg["start"])
            end = timestamp_to_seconds(seg["end"])
            max_end = max(max_end, end)
            text = seg["text"].strip()
            if text:
                formatted_segments.append(f"[{start:.2f}-{end:.2f}] {text}")

    return full_transcript, "\n".join(formatted_segments), max_end

# ---------- Prompt builders ----------
def build_system_prompt(total_duration: float, max_segments: int) -> str:
    return f"""You are given:
1. Full transcript (context only)
2. Optional user context
3. Timestamped segments (candidates)

Your task:
- Use the full transcript AND optional user context for understanding what’s important.
- Select highlight segments ONLY from the timestamped segments.

Rules:
- 3–10 seconds each
- Up to {max_segments} segments total
- Chronological order, no overlaps
- 0.00 ≤ start_time < end_time ≤ {total_duration:.2f}
- Return only valid JSON per schema
"""

def build_user_prompt(full_transcript: str, segments: str, user_context: Optional[str]) -> str:
    parts = []
    if user_context:
        parts.append(f"User context:\n{user_context}\n")
    parts.append(f"Full transcript (context only):\n{full_transcript}\n")
    parts.append(f"Timestamped segments (candidates):\n{segments}\n")
    parts.append("Pick the most important segments based on all context but using only timestamped segments.")
    return "\n".join(parts)

# ---------- Core ----------
def get_important_timestamps(json_file_path: str, user_context: Optional[str] = None) -> List[Tuple[float, float]]:
    full_transcript, segments, total_duration = load_transcript_from_json(json_file_path)

    # Number of clips heuristic: 1 per 90s, min 3, max 12
    max_segments = max(3, min(12, math.ceil(total_duration / 90.0)))

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": build_system_prompt(total_duration, max_segments)},
            {"role": "user", "content": build_user_prompt(full_transcript, segments, user_context)},
        ],
        text_format=ImportantSegments,
        temperature=0.2,
    )

    parsed = response.output_parsed

    # Post-process & validate
    cleaned: List[Tuple[float, float]] = []
    last_end = -1.0
    for seg in parsed.segments:
        s = max(0.0, float(seg.start_time))
        e = min(float(seg.end_time), total_duration)
        if e - s < 3.0 or e - s > 10.5:
            continue
        if s <= last_end:
            continue
        cleaned.append((s, e))
        last_end = e

    return cleaned[:max_segments]
