#!/usr/bin/env python3
"""
Test script for the video cutting function using pretty_BERYL.mp4
"""

from cut import cut_and_stitch_video
import os


def test_video_cutting():
    """Test the video cutting function with pretty_BERYL.mp4"""

    # Input and output file paths
    input_video = "input/pretty_BERYL.mp4"
    output_video = "output/cut_pretty_BERYL.mp4"

    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found!")
        return

    # Define timestamp pairs to cut (start_time, end_time) in seconds
    # Video is 10 seconds long, so all timestamps must be within 0-10s
    timestamp_pairs = [
        (1.0, 5.0),  # Cut from 1s to 5s (4 second segment)
        (7.0, 10.0),  # Cut from 7s to 11s (4 second segment)
    ]

    print(f"Testing video cutting with {input_video}")
    print(f"Extracting {len(timestamp_pairs)} segments:")
    for i, (start, end) in enumerate(timestamp_pairs, 1):
        duration = end - start
        print(f"  Segment {i}: {start}s to {end}s ({duration}s duration)")

    try:
        # Cut and stitch the video
        cut_and_stitch_video(timestamp_pairs, input_video, output_video)
        print(f"\n✅ Success! Cut video saved as: {output_video}")

        # Check if output file was created
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video)
            print(f"Output file size: {file_size / (1024 * 1024):.2f} MB")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error processing video: {e}")


def test_with_different_segments():
    """Test with different segment combinations"""

    input_video = "input/pretty_BERYL.mp4"
    output_video = "output/cut_pretty_BERYL_short.mp4"

    # Shorter segments for quick testing (within 10s video)
    timestamp_pairs = [
        (0.5, 3.0),  # 2.5 second segment from beginning
        (8.0, 10.0),  # 2 second segment from end
    ]

    print("\nTesting with shorter segments:")
    for i, (start, end) in enumerate(timestamp_pairs, 1):
        duration = end - start
        print(f"  Segment {i}: {start}s to {end}s ({duration}s duration)")

    try:
        cut_and_stitch_video(timestamp_pairs, input_video, output_video)
        print(f"✅ Short version saved as: {output_video}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎬 Video Cutting Test Script")
    print("=" * 40)

    # Run the main test
    test_video_cutting()

    # Run test with different segments
    test_with_different_segments()

    print("\n" + "=" * 40)
    print("Test completed!")
