"""
Video Processing Pipeline

A comprehensive pipeline that takes a YouTube URL and processes it through:
1. YouTube video download and transcription
2. AI-powered important segment extraction
3. Video cutting and stitching to create a shortened version

Usage:
    from pipeline import VideoPipeline

    pipeline = VideoPipeline()
    result = pipeline.process_youtube_video("https://youtube.com/watch?v=...")
"""

import os
import json

# import tempfile  # Not currently used
# import shutil     # Not currently used
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
import pickle

from dotenv import load_dotenv
from transcriber import YouTubeTranscriber
from highlights import get_important_timestamps
from cut import cut_and_stitch_video

# Load environment variables
load_dotenv()


class PipelineState(Enum):
    """States of the video processing pipeline."""

    INITIAL = auto()
    DOWNLOADING = auto()
    DOWNLOADED = auto()
    TRANSCRIBING = auto()
    TRANSCRIBED = auto()
    EXTRACTING_SEGMENTS = auto()
    SEGMENTS_EXTRACTED = auto()
    CUTTING_VIDEO = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class PipelineData:
    """Stateful data container for the pipeline."""

    # Input data
    youtube_url: str = ""
    output_filename: Optional[str] = None

    # Processing metadata
    video_name: str = ""
    session_id: str = ""
    start_time: Optional[datetime] = None

    # Downloaded/processed files
    original_video_path: Optional[str] = None
    transcription_file_path: Optional[str] = None

    # Processed data
    transcription_data: Optional[Dict[str, Any]] = None
    important_segments: Optional[List[Tuple[float, float]]] = None
    shortened_video_path: Optional[str] = None

    # Processing stats
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    last_successful_state: Optional[PipelineState] = None


@dataclass
class PipelineResult:
    """Result of the video processing pipeline."""

    success: bool
    original_video_path: str
    shortened_video_path: Optional[str] = None
    transcription_data: Optional[Dict[str, Any]] = None
    important_segments: Optional[List[Tuple[float, float]]] = None
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    openai_api_key: str
    work_dir: str = "pipeline_work"
    output_dir: str = "output"
    keep_intermediate_files: bool = False
    max_segments: int = 10
    min_segment_duration: float = 3.0
    max_segment_duration: float = 15.0
    log_level: str = "INFO"


class VideoPipeline:
    """Stateful video processing pipeline class."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the video processing pipeline.

        Args:
            config: Pipeline configuration. If None, uses default config with env variables.
        """
        if config is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            config = PipelineConfig(openai_api_key=api_key)

        self.config = config
        self.logger = self._setup_logging()
        self.transcriber = YouTubeTranscriber(config.openai_api_key)

        # Create work directories
        self.work_dir = Path(config.work_dir)
        self.output_dir = Path(config.output_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize state machine
        self.current_state = PipelineState.INITIAL
        self.data = PipelineData()

        # Valid state transitions
        self.valid_transitions = {
            PipelineState.INITIAL: [PipelineState.DOWNLOADING, PipelineState.FAILED],
            PipelineState.DOWNLOADING: [PipelineState.DOWNLOADED, PipelineState.FAILED],
            PipelineState.DOWNLOADED: [
                PipelineState.TRANSCRIBING,
                PipelineState.FAILED,
            ],
            PipelineState.TRANSCRIBING: [
                PipelineState.TRANSCRIBED,
                PipelineState.FAILED,
            ],
            PipelineState.TRANSCRIBED: [
                PipelineState.EXTRACTING_SEGMENTS,
                PipelineState.FAILED,
            ],
            PipelineState.EXTRACTING_SEGMENTS: [
                PipelineState.SEGMENTS_EXTRACTED,
                PipelineState.FAILED,
            ],
            PipelineState.SEGMENTS_EXTRACTED: [
                PipelineState.CUTTING_VIDEO,
                PipelineState.FAILED,
            ],
            PipelineState.CUTTING_VIDEO: [
                PipelineState.COMPLETED,
                PipelineState.FAILED,
            ],
            PipelineState.COMPLETED: [],
            PipelineState.FAILED: [],
        }

        self.logger.info(f"Pipeline initialized with work_dir: {self.work_dir}")

    def _transition_to_state(self, new_state: PipelineState) -> bool:
        """
        Safely transition to a new state with validation.

        Args:
            new_state: Target state

        Returns:
            True if transition was successful, False otherwise
        """
        if new_state not in self.valid_transitions.get(self.current_state, []):
            self.logger.error(
                f"Invalid state transition from {self.current_state} to {new_state}"
            )
            return False

        self.logger.info(f"State transition: {self.current_state} -> {new_state}")

        # Save last successful state before potential failure
        if new_state != PipelineState.FAILED:
            self.data.last_successful_state = self.current_state

        self.current_state = new_state
        return True

    def save_state(self, filepath: Optional[str] = None) -> str:
        """
        Save current pipeline state to disk for recovery.

        Args:
            filepath: Optional custom filepath for state file

        Returns:
            Path to saved state file
        """
        if filepath is None:
            filepath = str(self.work_dir / f"pipeline_state_{self.data.session_id}.pkl")

        state_data = {
            "current_state": self.current_state,
            "pipeline_data": self.data,
            "config": self.config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

        self.logger.info(f"Pipeline state saved to: {filepath}")
        return str(filepath)

    def load_state(self, filepath: str) -> bool:
        """
        Load pipeline state from disk for recovery.

        Args:
            filepath: Path to state file

        Returns:
            True if state was loaded successfully
        """
        try:
            with open(filepath, "rb") as f:
                state_data = pickle.load(f)

            self.current_state = state_data["current_state"]
            self.data = state_data["pipeline_data"]
            # Note: config is not restored to prevent security issues

            self.logger.info(f"Pipeline state loaded from: {filepath}")
            self.logger.info(f"Resumed at state: {self.current_state}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load state from {filepath}: {e}")
            return False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger("VideoPipeline")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def initialize_processing(
        self,
        youtube_url: str,
        output_filename: Optional[str] = None,
    ) -> bool:
        """
        Initialize processing with input parameters.

        Args:
            youtube_url: YouTube video URL
            output_filename: Optional custom output filename

        Returns:
            True if initialization successful
        """
        if self.current_state != PipelineState.INITIAL:
            self.logger.error("Pipeline must be in INITIAL state to start processing")
            return False

        # Initialize data
        self.data.youtube_url = youtube_url
        self.data.output_filename = output_filename
        self.data.start_time = datetime.now()
        self.data.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data.video_name = f"video_{self.data.session_id}"

        self.logger.info(f"Initialized processing for URL: {youtube_url}")
        return True

    def process_youtube_video(
        self,
        youtube_url: str,
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> PipelineResult:
        """
        Process a YouTube video through the complete pipeline using state machine.

        Args:
            youtube_url: YouTube video URL
            output_filename: Optional custom output filename (without extension)
            progress_callback: Optional callback function for progress updates

        Returns:
            PipelineResult containing all processing results and metadata
        """

        def update_progress(message: str):
            self.logger.info(message)
            if progress_callback:
                progress_callback(message)

        try:
            # Initialize processing
            if not self.initialize_processing(youtube_url, output_filename):
                raise ValueError("Failed to initialize processing")

            update_progress("Starting video processing pipeline...")

            # Execute pipeline steps in order
            steps = [
                (self.step_download_video, "Downloading video..."),
                (self.step_transcribe_video, "Transcribing video..."),
                (self.step_extract_segments, "Extracting important segments..."),
                (self.step_cut_video, "Creating shortened video..."),
            ]

            for step_func, step_msg in steps:
                update_progress(step_msg)
                if not step_func():
                    raise RuntimeError(f"Pipeline step failed: {step_msg}")

                # Save state after each successful step
                self.save_state()

            # Calculate final stats
            self._calculate_final_stats()

            if not self._transition_to_state(PipelineState.COMPLETED):
                raise RuntimeError("Failed to transition to completed state")

            update_progress("Pipeline completed successfully!")

            # Cleanup intermediate files if requested
            if not self.config.keep_intermediate_files:
                self._cleanup_intermediate_files()

            return self._create_result(success=True)

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.data.error_message = error_msg
            self._transition_to_state(PipelineState.FAILED)
            self.logger.error(error_msg, exc_info=True)
            update_progress(error_msg)

            return self._create_result(success=False)

    def resume_processing(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> PipelineResult:
        """
        Resume processing from current state.

        Args:
            progress_callback: Optional callback function for progress updates

        Returns:
            PipelineResult containing all processing results and metadata
        """

        def update_progress(message: str):
            self.logger.info(message)
            if progress_callback:
                progress_callback(message)

        if self.current_state == PipelineState.INITIAL:
            raise ValueError(
                "Cannot resume from INITIAL state. Use process_youtube_video instead."
            )

        if self.current_state in [PipelineState.COMPLETED, PipelineState.FAILED]:
            self.logger.info("Pipeline already in terminal state")
            return self._create_result(
                success=self.current_state == PipelineState.COMPLETED
            )

        try:
            update_progress(f"Resuming pipeline from state: {self.current_state}")

            # Resume from current state
            remaining_steps = []

            if self.current_state == PipelineState.DOWNLOADED:
                remaining_steps.extend(
                    [
                        (self.step_transcribe_video, "Transcribing video..."),
                        (
                            self.step_extract_segments,
                            "Extracting important segments...",
                        ),
                        (self.step_cut_video, "Creating shortened video..."),
                    ]
                )
            elif self.current_state == PipelineState.TRANSCRIBED:
                remaining_steps.extend(
                    [
                        (
                            self.step_extract_segments,
                            "Extracting important segments...",
                        ),
                        (self.step_cut_video, "Creating shortened video..."),
                    ]
                )
            elif self.current_state == PipelineState.SEGMENTS_EXTRACTED:
                remaining_steps.append(
                    (self.step_cut_video, "Creating shortened video...")
                )

            for step_func, step_msg in remaining_steps:
                update_progress(step_msg)
                if not step_func():
                    raise RuntimeError(f"Pipeline step failed: {step_msg}")
                self.save_state()

            # Calculate final stats and complete
            self._calculate_final_stats()
            if not self._transition_to_state(PipelineState.COMPLETED):
                raise RuntimeError("Failed to transition to completed state")

            update_progress("Pipeline resumed and completed successfully!")
            return self._create_result(success=True)

        except Exception as e:
            error_msg = f"Pipeline resume failed: {str(e)}"
            self.data.error_message = error_msg
            self._transition_to_state(PipelineState.FAILED)
            self.logger.error(error_msg, exc_info=True)
            update_progress(error_msg)

            return self._create_result(success=False)

    def step_download_video(self) -> bool:
        """Download video from YouTube."""
        if not self._transition_to_state(PipelineState.DOWNLOADING):
            return False

        try:
            # Use transcriber to download video (transcription comes later)
            video_path = self.transcriber.download_video_mp4(
                self.data.youtube_url, Path("input") / self.data.video_name
            )

            self.data.original_video_path = video_path
            self.logger.info(f"Video downloaded to: {video_path}")

            return self._transition_to_state(PipelineState.DOWNLOADED)

        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
            self._transition_to_state(PipelineState.FAILED)
            return False

    # ---------- REPLACE your step_transcribe_video with this ----------
    def step_transcribe_video(self) -> bool:
        """Transcribe the downloaded video and save JSON in the format highlights.py expects."""
        if not self._transition_to_state(PipelineState.TRANSCRIBING):
            return False

        if not self.data.original_video_path:
            self.logger.error("No video path available for transcription")
            self._transition_to_state(PipelineState.FAILED)
            return False

        try:
            # 1) Extract audio & split
            audio_path = self.transcriber.extract_audio_from_video(
                self.data.original_video_path, Path("input") / self.data.video_name
            )
            part_files = self.transcriber.split_by_duration(audio_path)

            # 2) Transcribe and merge
            results = [self.transcriber.transcribe_file(p) for p in part_files]
            transcription_data = self.transcriber.merge_transcripts(results, part_files)
            self.data.transcription_data = transcription_data  # keep raw in memory for stats, etc.

            # 3) (Optional) Keep a raw debug dump
            raw_file = self.work_dir / f"transcription_raw_{self.data.session_id}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Raw transcription saved to: {raw_file}")

            # 4) Save JSON in the EXACT format expected by highlights.py
            # Prefer the transcriber's save_all() if it exists (writes .json/.srt/.txt),
            # otherwise fall back to our local writer.
            if hasattr(self.transcriber, "save_all"):
                out_base = self.work_dir / f"transcription_{self.data.session_id}"
                self.transcriber.save_all(transcription_data, out_base, include_json=True)
                segments_json = out_base.with_suffix(".json")
            else:
                segments_json = self._write_segments_json(
                    transcription_data,
                    self.work_dir / f"transcription_{self.data.session_id}.json",
                )

            self.data.transcription_file_path = str(segments_json)
            self.logger.info(f"Segments JSON (for highlights.py) saved to: {segments_json}")

            return self._transition_to_state(PipelineState.TRANSCRIBED)

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            self._transition_to_state(PipelineState.FAILED)
            return False


    # ---------- REPLACE your step_extract_segments with this ----------
    def step_extract_segments(self) -> bool:
        """Extract important segments from the already-correct segments JSON."""
        if not self._transition_to_state(PipelineState.EXTRACTING_SEGMENTS):
            return False

        if not self.data.transcription_file_path:
            self.logger.error("No segments JSON available for extraction")
            self._transition_to_state(PipelineState.FAILED)
            return False

        try:
            # Directly use the JSON file that matches highlights.py expectations
            important_segments = get_important_timestamps(self.data.transcription_file_path)

            if not important_segments:
                self.logger.error("No important segments identified")
                self._transition_to_state(PipelineState.FAILED)
                return False

            self.data.important_segments = important_segments
            self.logger.info(f"Identified {len(important_segments)} important segments")
            return self._transition_to_state(PipelineState.SEGMENTS_EXTRACTED)

        except Exception as e:
            self.logger.error(f"Segment extraction failed: {e}")
            self._transition_to_state(PipelineState.FAILED)
            return False


    # ---------- ADD this helper somewhere in the class (e.g., near _seconds_to_timestamp) ----------
    def _write_segments_json(self, merged: Dict[str, Any], out_path: Path) -> Path:
        """
        Write JSON in the format expected by highlights.load_transcript_from_json:
        [
            {"text": "<full transcript>"},
            {"start": "HH:MM:SS,ms", "end": "HH:MM:SS,ms", "text": "<segment>"},
            ...
        ]
        Falls back to grouping words into short segments if native segments aren't present.
        """
        full_text = (merged.get("text") or "").strip()

        segments = merged.get("segments") or []
        if not segments:
            # fallback: build lightweight segments from words
            words = merged.get("words") or []
            segments = []
            phrase, cur_start, last_end = [], None, None

            def flush():
                nonlocal phrase, cur_start, last_end
                if phrase and cur_start is not None and last_end is not None:
                    segments.append({
                        "start": cur_start,
                        "end": last_end,
                        "text": " ".join(phrase).strip()
                    })
                phrase, cur_start, last_end = [], None, None

            for w in words:
                word = (w.get("word") or "").strip()
                s = float(w.get("start", 0.0))
                e = float(w.get("end", s))
                if cur_start is None:
                    cur_start = s
                phrase.append(word)
                last_end = e
                if word.endswith((".", "!", "?", ",")) or len(phrase) >= 10:
                    flush()
            if phrase:
                flush()

        # Build pretty array
        pretty = [{"text": full_text}]
        for seg in segments:
            s = seg.get("start")
            e = seg.get("end")
            txt = (seg.get("text") or "").strip()
            if txt and s is not None and e is not None:
                # Convert numeric seconds to "HH:MM:SS,mmm" if needed
                s_str = self._seconds_to_timestamp(float(s)) if isinstance(s, (int, float)) else str(s)
                e_str = self._seconds_to_timestamp(float(e)) if isinstance(e, (int, float)) else str(e)
                pretty.append({"start": s_str, "end": e_str, "text": txt})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pretty, f, ensure_ascii=False, indent=2)
        return out_path


    def step_cut_video(self) -> bool:
        """Cut and stitch video segments."""
        if not self._transition_to_state(PipelineState.CUTTING_VIDEO):
            return False

        if not self.data.original_video_path or not self.data.important_segments:
            self.logger.error("Missing video path or segments for cutting")
            self._transition_to_state(PipelineState.FAILED)
            return False

        try:
            # Generate output filename
            output_filename = (
                self.data.output_filename or f"shortened_video_{self.data.session_id}"
            )
            output_path = self.output_dir / f"{output_filename}.mp4"

            # Use the existing cut_and_stitch_video function
            cut_and_stitch_video(
                self.data.important_segments,
                self.data.original_video_path,
                str(output_path),
            )

            self.data.shortened_video_path = str(output_path)
            self.logger.info(f"Shortened video saved to: {output_path}")

            return True

        except Exception as e:
            self.logger.error(f"Video cutting failed: {e}")
            self._transition_to_state(PipelineState.FAILED)
            return False

    def _convert_transcription_for_get_important(self) -> list:
        """
        Convert transcriber output to format expected by get_important.py.

        From: {"words": [{"word": str, "start": float, "end": float}]}
        To: [{"start": "HH:MM:SS,mmm", "text": str}]
        """
        converted = []

        if (
            not self.data.transcription_data
            or "words" not in self.data.transcription_data
        ):
            return converted

        # Group words into phrases for better analysis
        current_phrase = []
        current_start = None

        for word_data in self.data.transcription_data["words"]:
            word = word_data.get("word", "").strip()
            start = word_data.get("start", 0)

            if current_start is None:
                current_start = start

            current_phrase.append(word)

            # End phrase on punctuation or after reasonable length
            ends_sentence = any(word.endswith(p) for p in [".", "!", "?"])
            long_phrase = len(current_phrase) >= 10

            if ends_sentence or long_phrase:
                phrase_text = " ".join(current_phrase).strip()
                if phrase_text:
                    # Convert timestamp to HH:MM:SS,mmm format
                    start_timestamp = self._seconds_to_timestamp(current_start)

                    converted.append({"start": start_timestamp, "text": phrase_text})

                current_phrase = []
                current_start = None

        # Handle remaining phrase
        if current_phrase and current_start is not None:
            phrase_text = " ".join(current_phrase).strip()
            if phrase_text:
                start_timestamp = self._seconds_to_timestamp(current_start)
                converted.append({"start": start_timestamp, "text": phrase_text})

        return converted

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS,mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _calculate_final_stats(self):
        """Calculate final processing statistics."""
        if not self.data.start_time:
            return

        end_time = datetime.now()
        processing_time = (end_time - self.data.start_time).total_seconds()

        stats = {
            "processing_time_seconds": processing_time,
            "session_id": self.data.session_id,
        }

        if self.data.important_segments:
            stats.update(
                {
                    "num_segments": len(self.data.important_segments),
                    "total_segment_duration": sum(
                        end - start for start, end in self.data.important_segments
                    ),
                }
            )

            if self.data.original_video_path:
                stats["compression_ratio"] = self._calculate_compression_ratio(
                    self.data.original_video_path, self.data.important_segments
                )

        self.data.processing_stats = stats
        self.logger.info(f"Final stats: {stats}")

    def _create_result(self, success: bool) -> PipelineResult:
        """Create a PipelineResult from current state."""
        return PipelineResult(
            success=success,
            original_video_path=self.data.original_video_path or "",
            shortened_video_path=self.data.shortened_video_path,
            transcription_data=self.data.transcription_data,
            important_segments=self.data.important_segments,
            error_message=self.data.error_message,
            processing_stats=self.data.processing_stats,
        )

    def _calculate_compression_ratio(
        self, original_video_path: str, segments: List[Tuple[float, float]]
    ) -> float:
        """Calculate compression ratio (shortened duration / original duration)."""
        try:
            from moviepy import VideoFileClip

            with VideoFileClip(original_video_path) as clip:
                original_duration = clip.duration

            if original_duration is None or original_duration <= 0:
                return 0.0

            shortened_duration = sum(end - start for start, end in segments)

            return shortened_duration / original_duration

        except Exception as e:
            self.logger.warning(f"Could not calculate compression ratio: {e}")
            return 0.0

    def _cleanup_intermediate_files(self):
        """Clean up intermediate files if requested."""
        try:
            # Note: The transcriber creates files in input/ directory
            # We don't clean those up as they might be needed for reference
            pass
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


def main():
    """Example usage of the video processing pipeline."""
    # Example configuration
    config = PipelineConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        keep_intermediate_files=True,  # Keep files for debugging
        max_segments=8,
        log_level="INFO",
    )

    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        return

    # Initialize pipeline
    pipeline = VideoPipeline(config)

    # Get YouTube URL from user
    youtube_url = input("Enter YouTube URL: ").strip()
    if not youtube_url:
        print("No URL provided")
        return

    # Progress callback
    def progress_callback(message: str):
        print("📹 " + message)

    # Process video
    result = pipeline.process_youtube_video(
        youtube_url=youtube_url, progress_callback=progress_callback
    )

    # Display results
    if result.success:
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Original video: {result.original_video_path}")
        print(f"✂️  Shortened video: {result.shortened_video_path}")

        if result.processing_stats:
            stats = result.processing_stats
            print(f"⏱️  Processing time: {stats['processing_time_seconds']:.1f} seconds")
            print(f"📊 Segments: {stats['num_segments']}")
            print(f"🗜️  Compression ratio: {stats['compression_ratio']:.1%}")
            print(
                f"⏰ Total segment duration: {stats['total_segment_duration']:.1f} seconds"
            )
    else:
        print(f"\n❌ Pipeline failed: {result.error_message}")


if __name__ == "__main__":
    main()
