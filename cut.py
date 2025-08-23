from moviepy import VideoFileClip, concatenate_videoclips
import os
from typing import List, Tuple


def cut_and_stitch_video(
    timestamp_pairs: List[Tuple[float, float]],
    input_video_path: str,
    output_video_path: str,
) -> None:
    """
    Cut video segments from input video based on timestamp pairs and stitch them together.

    Args:
        timestamp_pairs: List of (start_time, end_time) tuples in seconds
        input_video_path: Path to the input video file
        output_video_path: Path where the output video will be saved

    Raises:
        FileNotFoundError: If input video file doesn't exist
        ValueError: If timestamp pairs are invalid
        Exception: For other moviepy-related errors
    """
    # Validate input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Validate timestamp pairs
    if not timestamp_pairs:
        raise ValueError("Timestamp pairs list cannot be empty")

    for i, (start, end) in enumerate(timestamp_pairs):
        if start < 0 or end < 0:
            raise ValueError(f"Timestamp pair {i}: timestamps cannot be negative")
        if start >= end:
            raise ValueError(
                f"Timestamp pair {i}: start time ({start}) must be less than end time ({end})"
            )

    try:
        # Load the input video
        video = VideoFileClip(input_video_path)

        # Validate timestamps against video duration
        video_duration = video.duration
        for i, (start, end) in enumerate(timestamp_pairs):
            if end > video_duration:
                raise ValueError(
                    f"Timestamp pair {i}: end time ({end}) exceeds video duration ({video_duration})"
                )

        # Create video clips for each timestamp pair
        clips = []
        for start, end in timestamp_pairs:
            clip = video.subclipped(start, end)
            clips.append(clip)

        # Concatenate all clips
        final_video = concatenate_videoclips(clips)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the final video to output path
        final_video.write_videofile(output_video_path, logger=None)

        # Clean up clips to free memory
        for clip in clips:
            clip.close()
        final_video.close()
        video.close()

        print(
            f"Successfully created video with {len(clips)} segments: {output_video_path}"
        )

    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")
