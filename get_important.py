from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Tuple

load_dotenv()

client = OpenAI()

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS,mmm format to seconds"""
    time_part, ms_part = timestamp.split(',')
    hours, minutes, seconds = time_part.split(':')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    total_seconds += int(ms_part) / 1000
    return total_seconds

def load_transcript_from_json(json_file_path: str) -> str:
    """Load transcript from JSON file and format for LLM"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    formatted_transcript = ""
    for segment in data:
        if 'start' in segment and 'text' in segment:
            start_seconds = timestamp_to_seconds(segment['start'])
            end_seconds = timestamp_to_seconds(segment['end'])
            formatted_transcript += f"[{start_seconds:.1f}s - {end_seconds:.1f}s] {segment['text']}\n"
    
    return formatted_transcript

class TimeStampSegment(BaseModel):
    start_time: float
    end_time: float

class ImportantSegments(BaseModel):
    segments: List[TimeStampSegment]

def get_important_timestamps(json_file_path: str) -> List[Tuple[float, float]]:
    """
    Extract important timestamp pairs from transcript
    Returns in format: [(1.0, 5.0, "reason"), (7.0, 10.0, "reason"), ...]
    """
    transcript = load_transcript_from_json(json_file_path)


    response = client.responses.parse(
        model="gpt-5",
        input=[
            {
                "role": "system", 
                "content": "Analyze the transcript and identify the most important segments. Focus on key information, main points, and crucial insights. Return segments that are typically 3-10 seconds long for optimal viewing."
            },
            {
                "role": "user",
                "content": f"Extract the most important segments from this transcript with timestamps:\n\n{transcript}"
            },
        ],
        text_format=ImportantSegments,
    )

    segments = response.output_parsed.segments
    
    # Convert to the exact format you need
    timestamp_pairs = [(seg.start_time, seg.end_time) for seg in segments]
    print(timestamp_pairs)
    return timestamp_pairs



def main():
    json_file_path = "nn_segmented.json"
    
    # Load and format transcript from your JSON file
    #transcript = load_transcript_from_json(json_file_path)
    
    # Extract important timestamps
    timestamps = get_important_timestamps(json_file_path)
    print("Important segments:", timestamps)
    
    # Example of how to use the results
    for start_time, end_time in timestamps:
        print(f"Segment: {start_time:.1f}s - {end_time:.1f}s")

if __name__ == "__main__":
    main()
    