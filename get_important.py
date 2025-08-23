from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Tuple

load_dotenv()

client = OpenAI()

class TimeStampSegment(BaseModel):
    start_time: float
    end_time: float
    reason: str

class ImportantSegments(BaseModel):
    segments: List[TimeStampSegment]

def get_important_timestamps(transcript_with_timestamps: str) -> List[Tuple[float, float]]:
    """
    Extract important timestamp pairs from transcript
    Returns in format: [(1.0, 5.0), (7.0, 10.0), ...]
    """
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system", 
                "content": "Analyze the transcript and identify the most important segments. Focus on key information, main points, and crucial insights. Return segments that are typically 3-10 seconds long for optimal viewing."
            },
            {
                "role": "user",
                "content": f"Extract the most important segments from this transcript with timestamps:\n\n{transcript_with_timestamps}"
            },
        ],
        text_format=ImportantSegments,
    )

    segments = response.output_parsed.segments
    
    # Convert to the exact format you need
    timestamp_pairs = [(seg.start_time, seg.end_time, seg.reason) for seg in segments]
    
    return timestamp_pairs



def main():
    
    # Example of timestamp extraction
    sample_transcript = """
    [0.0s] Hi everyone, welcome to today's tutorial
    [5.0s] Today we're going to learn about machine learning
    [10.0s] A neural network is a computational model
    [15.0s] Here's the key insight: they can learn patterns
    [20.0s] Let me show you a practical example
    """
    
    timestamps = get_important_timestamps(sample_transcript)
    print("Important segments:", timestamps)

if __name__ == "__main__":
    main()
