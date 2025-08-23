import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import pygame
from moviepy import VideoFileClip
import tempfile
from dotenv import load_dotenv
from pipeline import VideoPipeline, PipelineConfig

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ERROR")


def parse_timestamps(timestamp_text):
    """Parse timestamp pairs from input text."""
    try:
        pairs = []
        segments = timestamp_text.split(";")

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            start_str, end_str = segment.split(",")
            start = float(start_str.strip())
            end = float(end_str.strip())
            pairs.append((start, end))

        return pairs
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {str(e)}")


class VideoShorteningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Shortening Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # Video player variables
        self.current_video_path: str | None = None
        self.shortened_video_path: str | None = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.video_thread = None
        self.stop_playback = False

        # Video display variables
        self.original_video_label: tk.Label | None = None
        self.shortened_video_label: tk.Label | None = None
        self.original_cap: cv2.VideoCapture | None = None
        self.shortened_cap: cv2.VideoCapture | None = None
        self.original_playing = False
        self.shortened_playing = False

        # Audio variables
        pygame.mixer.init()
        self.original_audio_path: str | None = None
        self.shortened_audio_path: str | None = None
        self.audio_start_time: float = 0
        self.temp_audio_files = []  # Track temp files for cleanup

        # Transcription variables
        self.transcription_data: dict | None = None

        # Pipeline for stateful processing
        config = PipelineConfig(
            openai_api_key=OPENAI_API_KEY,
            work_dir="gui_pipeline_work",
            output_dir="output",
            keep_intermediate_files=True,
        )
        self.pipeline = VideoPipeline(config)

        self.setup_ui()

    def extract_audio_from_video(self, video_path: str) -> str | None:
        """Extract audio from video file and return path to temporary audio file."""
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                return None

            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()

            # Extract audio
            video_clip.audio.write_audiofile(temp_audio_path, logger=None)
            video_clip.close()

            # Track temp file for cleanup
            self.temp_audio_files.append(temp_audio_path)
            return temp_audio_path

        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def setup_ui(self):
        # Main title
        title_label = tk.Label(
            self.root,
            text="Video Shortening Tool",
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#333333",
        )
        title_label.pack(pady=10)

        # YouTube URL input frame
        url_frame = tk.Frame(self.root, bg="#f0f0f0")
        url_frame.pack(pady=10)

        url_label = tk.Label(
            url_frame,
            text="YouTube URL:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#333333",
        )
        url_label.pack(side=tk.LEFT, padx=5)

        self.url_entry = tk.Entry(
            url_frame,
            width=50,
            font=("Arial", 11),
            cursor="ibeam",
            relief="sunken",
            borderwidth=2,
        )
        self.url_entry.pack(side=tk.LEFT, padx=5)
        self.url_entry.insert(0, "https://www.youtube.com/watch?v=example")

        self.download_button = tk.Button(
            url_frame,
            text="Download Video",
            command=self.download_youtube_video,
            font=("Arial", 12, "bold"),
            bg="#FF5722",
            fg="black",
            padx=20,
            pady=5,
            relief="raised",
            borderwidth=2,
            cursor="hand2",
        )
        self.download_button.pack(side=tk.LEFT, padx=5)

        # File selection frame (as alternative option)
        file_frame = tk.Frame(self.root, bg="#f0f0f0")
        file_frame.pack(pady=5)

        select_button = tk.Button(
            file_frame,
            text="Or Select Local Video File",
            command=self.select_video_file,
            font=("Arial", 10),
            bg="#4CAF50",
            fg="black",
            padx=15,
            pady=3,
            relief="raised",
            borderwidth=2,
            cursor="hand2",
        )
        select_button.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666666",
        )
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Video display frame
        video_frame = tk.Frame(self.root, bg="#f0f0f0")
        video_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Original video section
        original_section = tk.Frame(video_frame, bg="#f0f0f0")
        original_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        original_title = tk.Label(
            original_section,
            text="Original Video",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333",
        )
        original_title.pack(pady=5)

        self.original_video_frame = tk.Frame(
            original_section, bg="black", width=400, height=300
        )
        self.original_video_frame.pack(pady=5)
        self.original_video_frame.pack_propagate(False)

        self.original_video_label = tk.Label(
            self.original_video_frame,
            text="No video loaded",
            bg="black",
            fg="black",
            font=("Arial", 12),
        )
        self.original_video_label.pack(expand=True)

        # Original video controls
        original_controls = tk.Frame(original_section, bg="#f0f0f0")
        original_controls.pack(pady=5)

        self.original_play_button = tk.Button(
            original_controls,
            text="Play",
            command=self.toggle_original_playback,
            state=tk.DISABLED,
            font=("Arial", 10, "bold"),
            bg="#1976D2",
            fg="black",
            padx=15,
            pady=3,
            relief="raised",
            borderwidth=2,
            disabledforeground="#CCCCCC",
            cursor="hand2",
        )
        self.original_play_button.pack(side=tk.LEFT, padx=5)

        # Shortened video section
        shortened_section = tk.Frame(video_frame, bg="#f0f0f0")
        shortened_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        shortened_title = tk.Label(
            shortened_section,
            text="Shortened Video",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333",
        )
        shortened_title.pack(pady=5)

        self.shortened_video_frame = tk.Frame(
            shortened_section, bg="black", width=400, height=300
        )
        self.shortened_video_frame.pack(pady=5)
        self.shortened_video_frame.pack_propagate(False)

        self.shortened_video_label = tk.Label(
            self.shortened_video_frame,
            text="No shortened video",
            bg="black",
            fg="black",
            font=("Arial", 12),
        )
        self.shortened_video_label.pack(expand=True)

        # Shortened video controls
        shortened_controls = tk.Frame(shortened_section, bg="#f0f0f0")
        shortened_controls.pack(pady=5)

        self.shortened_play_button = tk.Button(
            shortened_controls,
            text="Play",
            command=self.toggle_shortened_playback,
            state=tk.DISABLED,
            font=("Arial", 10, "bold"),
            bg="#1976D2",
            fg="black",
            padx=15,
            pady=3,
            relief="raised",
            borderwidth=2,
            disabledforeground="#CCCCCC",
            cursor="hand2",
        )
        self.shortened_play_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(
            shortened_controls,
            text="Save Video",
            command=self.save_shortened_video,
            state=tk.DISABLED,
            font=("Arial", 10, "bold"),
            bg="#F57C00",
            fg="black",
            padx=15,
            pady=3,
            relief="raised",
            borderwidth=2,
            disabledforeground="#CCCCCC",
            cursor="hand2",
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.transcription_button = tk.Button(
            shortened_controls,
            text="View Transcription",
            command=self.view_transcription,
            state=tk.DISABLED,
            font=("Arial", 10, "bold"),
            bg="#9C27B0",
            fg="white",
            padx=15,
            pady=3,
            relief="raised",
            borderwidth=2,
            disabledforeground="#CCCCCC",
            cursor="hand2",
        )
        self.transcription_button.pack(side=tk.LEFT, padx=5)

        # Action buttons frame
        action_frame = tk.Frame(self.root, bg="#f0f0f0")
        action_frame.pack(pady=20)

        # Info label for automatic processing
        info_label = tk.Label(
            action_frame,
            text="After downloading a video, AI will automatically identify important segments to create a shortened version.",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666666",
            wraplength=500,
        )
        info_label.pack(pady=10)

        # Shorten button
        self.shorten_button = tk.Button(
            action_frame,
            text="View Shortened Video",
            command=self.shorten_video,
            state=tk.DISABLED,
            font=("Arial", 14, "bold"),
            bg="#D32F2F",
            fg="black",
            padx=30,
            pady=10,
            relief="raised",
            borderwidth=3,
            activebackground="#B71C1C",
            activeforeground="white",
            disabledforeground="#CCCCCC",
            cursor="hand2",
        )
        self.shorten_button.pack(pady=10)

        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(
            action_frame,
            textvariable=self.progress_var,
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666666",
        )
        self.progress_label.pack(pady=5)

    def setup_timestamp_input(self, timestamp_frame):
        tk.Label(
            timestamp_frame,
            text="Timestamp pairs (start,end in seconds, separated by semicolons):",
            font=("Arial", 10),
            bg="#f0f0f0",
        ).pack()

        tk.Label(
            timestamp_frame,
            text="Example: 10,20;30,45;60,80",
            font=("Arial", 9, "italic"),
            bg="#f0f0f0",
            fg="#666666",
        ).pack()

        self.timestamp_entry = tk.Entry(
            timestamp_frame, width=50, font=("Arial", 10), cursor="ibeam"
        )
        self.timestamp_entry.pack(pady=5)
        self.timestamp_entry.insert(0, "10,20;30,40")  # Default example

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.current_video_path = file_path
            self.file_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.load_original_video()
            self.shorten_button.config(state=tk.NORMAL)

    def download_youtube_video(self):
        """Download YouTube video with proper loading state management."""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL.")
            return

        # Basic URL validation
        if not any(domain in url.lower() for domain in ["youtube.com", "youtu.be"]):
            messagebox.showerror("Error", "Please enter a valid YouTube URL.")
            return

        # Disable UI elements during download
        self.download_button.config(state=tk.DISABLED, text="Downloading...")
        self.url_entry.config(state=tk.DISABLED)
        self.shorten_button.config(state=tk.DISABLED)
        self.progress_var.set("Downloading video from YouTube... Please wait.")

        # Start download in separate thread
        threading.Thread(
            target=self.process_youtube_download, args=(url,), daemon=True
        ).start()

    def process_youtube_download(self, url: str):
        """Process YouTube download in background thread using stateful pipeline."""
        try:
            # Progress callback to update UI
            def progress_callback(message):
                self.root.after(0, lambda: self.progress_var.set(message))

            # Use the stateful pipeline
            result = self.pipeline.process_youtube_video(
                youtube_url=url, progress_callback=progress_callback
            )

            if result.success:
                # Store transcription data
                self.transcription_data = result.transcription_data

                # Update UI in main thread
                self.root.after(
                    0, lambda: self.on_download_success(result.original_video_path)
                )
            else:
                error_msg = result.error_message or "Unknown pipeline error"
                self.root.after(0, lambda: self.on_download_error(error_msg))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.on_download_error(error_msg))

    def on_download_success(self, download_path: str):
        """Handle successful YouTube download."""
        self.current_video_path = download_path
        self.file_label.config(text=f"Downloaded: {os.path.basename(download_path)}")
        self.progress_var.set("Download and transcription completed successfully!")

        # Re-enable UI elements
        self.download_button.config(
            state=tk.NORMAL,
            text="Download Video",
            bg="#4CAF50",
            fg="black",
        )
        self.url_entry.config(state=tk.NORMAL)

        # Enable shorten button since pipeline automatically creates shortened video
        self.shorten_button.config(state=tk.NORMAL)

        # Load the downloaded video
        self.load_original_video()

        # Show success message with transcription info
        transcription_info = ""
        if self.transcription_data:
            word_count = len(self.transcription_data.get("words", []))
            transcription_info = f"\nTranscription: {word_count} words detected"

        messagebox.showinfo(
            "Success",
            f"Video downloaded and transcribed successfully!{transcription_info}",
        )

        # Enable transcription button if we have transcription data
        if self.transcription_data:
            self.transcription_button.config(state=tk.NORMAL)

    def view_transcription(self):
        """Display transcription in a new window."""
        if not self.transcription_data:
            messagebox.showwarning(
                "No Transcription", "No transcription data available."
            )
            return

        # Create new window for transcription
        transcription_window = tk.Toplevel(self.root)
        transcription_window.title("Video Transcription")
        transcription_window.geometry("800x600")
        transcription_window.configure(bg="#f0f0f0")

        # Add scrollable text area
        text_frame = tk.Frame(transcription_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_area = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Arial", 11),
            padx=10,
            pady=10,
        )
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_area.yview)

        # Add transcription text with timestamps
        transcription_text = ""
        for word_data in self.transcription_data.get("words", []):
            start_time = word_data.get("start", 0)
            end_time = word_data.get("end", 0)
            word = word_data.get("word", "")

            # Format time as MM:SS
            start_min, start_sec = divmod(int(start_time), 60)
            end_min, end_sec = divmod(int(end_time), 60)

            transcription_text += f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] {word}\n"

        text_area.insert(tk.END, transcription_text)
        text_area.config(state=tk.DISABLED)  # Make read-only

        # Add export button
        export_button = tk.Button(
            transcription_window,
            text="Export Transcription",
            command=lambda: self.export_transcription(),
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5,
        )
        export_button.pack(pady=10)

    def export_transcription(self):
        """Export transcription to a file."""
        if not self.transcription_data:
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Transcription",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("SRT files", "*.srt"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                if file_path.endswith(".json"):
                    # Export as JSON
                    import json

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(
                            self.transcription_data, f, ensure_ascii=False, indent=2
                        )
                elif file_path.endswith(".srt"):
                    # Export as SRT (using the transcriber's format)
                    from transcriber import YouTubeTranscriber

                    temp_transcriber = YouTubeTranscriber(
                        ""
                    )  # Empty key for format only
                    with open(file_path, "w", encoding="utf-8") as f:
                        # Generate SRT content
                        idx, phrase, cur_start = 1, [], None
                        for w in self.transcription_data.get("words", []):
                            if cur_start is None:
                                cur_start = w["start"]
                            phrase.append(w["word"])
                            end = w["end"]
                            ends_sentence = any(
                                phrase[-1].rstrip().endswith(p)
                                for p in [".", "!", "?", ","]
                            )
                            too_long = (end - cur_start) > 5.0
                            if ends_sentence or too_long:
                                text = " ".join(phrase).strip()
                                if text:
                                    f.write(f"{idx}\n")
                                    f.write(
                                        f"{temp_transcriber._srt_time(cur_start)} --> {temp_transcriber._srt_time(end)}\n"
                                    )
                                    f.write(f"{text}\n\n")
                                    idx += 1
                                phrase, cur_start = [], None
                        if phrase:
                            end = (
                                self.transcription_data["words"][-1]["end"]
                                if self.transcription_data.get("words")
                                else 0.0
                            )
                            text = " ".join(phrase).strip()
                            if text:
                                f.write(f"{idx}\n")
                                f.write(
                                    f"{temp_transcriber._srt_time(cur_start or 0.0)} --> {temp_transcriber._srt_time(end)}\n"
                                )
                                f.write(f"{text}\n")
                else:
                    # Export as plain text
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(self.transcription_data.get("text", ""))

                messagebox.showinfo(
                    "Success", f"Transcription exported to: {file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to export transcription: {str(e)}"
                )

    def on_download_error(self, error_msg: str):
        """Handle YouTube download error."""
        self.progress_var.set("Download failed.")

        # Re-enable UI elements
        self.download_button.config(state=tk.NORMAL, text="Download Video")
        self.url_entry.config(state=tk.NORMAL)

        messagebox.showerror("Download Error", f"Failed to download video: {error_msg}")

    def load_original_video(self):
        if self.original_cap:
            self.original_cap.release()

        if not self.current_video_path:
            return

        self.original_cap = cv2.VideoCapture(self.current_video_path)
        self.original_play_button.config(state=tk.NORMAL)

        # Extract audio from original video
        self.original_audio_path = self.extract_audio_from_video(
            self.current_video_path
        )

        # Display first frame
        if self.original_cap.isOpened():
            ret, frame = self.original_cap.read()
            if ret:
                self.display_frame(frame, self.original_video_label)

            # Reset to beginning
            self.original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def load_shortened_video(self):
        if self.shortened_cap:
            self.shortened_cap.release()

        if not self.shortened_video_path:
            return

        self.shortened_cap = cv2.VideoCapture(self.shortened_video_path)
        self.shortened_play_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

        # Extract audio from shortened video
        self.shortened_audio_path = self.extract_audio_from_video(
            self.shortened_video_path
        )

        # Display first frame
        if self.shortened_cap.isOpened():
            ret, frame = self.shortened_cap.read()
            if ret:
                self.display_frame(frame, self.shortened_video_label)

            # Reset to beginning
            self.shortened_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def display_frame(self, frame, label):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to fit the display area
        height, width = rgb_frame.shape[:2]
        max_width, max_height = 380, 280

        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize frame
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height))

        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(resized_frame)
        photo = ImageTk.PhotoImage(pil_image)

        # Update label
        label.config(image=photo, text="")
        label.image = photo  # Keep a reference

    def toggle_original_playback(self):
        if self.original_playing:
            self.original_playing = False
            self.original_play_button.config(text="Play")
            # Stop audio
            pygame.mixer.music.stop()
        else:
            self.original_playing = True
            self.original_play_button.config(text="Pause")
            # Start audio if available
            if self.original_audio_path and os.path.exists(self.original_audio_path):
                pygame.mixer.music.load(self.original_audio_path)
                pygame.mixer.music.play()
                self.audio_start_time = time.time()
            threading.Thread(target=self.play_original_video, daemon=True).start()

    def toggle_shortened_playback(self):
        if self.shortened_playing:
            self.shortened_playing = False
            self.shortened_play_button.config(text="Play")
            # Stop audio
            pygame.mixer.music.stop()
        else:
            self.shortened_playing = True
            self.shortened_play_button.config(text="Pause")
            # Start audio if available
            if self.shortened_audio_path and os.path.exists(self.shortened_audio_path):
                pygame.mixer.music.load(self.shortened_audio_path)
                pygame.mixer.music.play()
                self.audio_start_time = time.time()
            threading.Thread(target=self.play_shortened_video, daemon=True).start()

    def play_original_video(self):
        while self.original_playing and self.original_cap:
            ret, frame = self.original_cap.read()
            if not ret:
                # End of video, reset to beginning and restart audio
                self.original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self.original_audio_path and os.path.exists(
                    self.original_audio_path
                ):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(self.original_audio_path)
                    pygame.mixer.music.play()
                    self.audio_start_time = time.time()
                ret, frame = self.original_cap.read()

            if ret:
                self.display_frame(frame, self.original_video_label)
                time.sleep(1 / 30)  # Approximate 30 FPS

        # Update button text when playback stops
        if not self.original_playing:
            self.root.after(0, lambda: self.original_play_button.config(text="Play"))

    def play_shortened_video(self):
        while self.shortened_playing and self.shortened_cap:
            ret, frame = self.shortened_cap.read()
            if not ret:
                # End of video, reset to beginning and restart audio
                self.shortened_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self.shortened_audio_path and os.path.exists(
                    self.shortened_audio_path
                ):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(self.shortened_audio_path)
                    pygame.mixer.music.play()
                    self.audio_start_time = time.time()
                ret, frame = self.shortened_cap.read()

            if ret:
                self.display_frame(frame, self.shortened_video_label)
                time.sleep(1 / 30)  # Approximate 30 FPS

        # Update button text when playback stops
        if not self.shortened_playing:
            self.root.after(0, lambda: self.shortened_play_button.config(text="Play"))

    def shorten_video(self):
        """Create shortened video using pipeline's important segments."""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please download a video first.")
            return

        # Check if we have pipeline results with important segments
        if (
            not hasattr(self.pipeline, "data")
            or not self.pipeline.data.important_segments
        ):
            messagebox.showerror(
                "Error",
                "No important segments found. Please download and process a video first.",
            )
            return

        # Use the shortened video path from the pipeline
        if self.pipeline.data.shortened_video_path:
            if os.path.exists(self.pipeline.data.shortened_video_path):
                self.shortened_video_path = self.pipeline.data.shortened_video_path
                self.load_shortened_video()
                messagebox.showinfo(
                    "Success",
                    f"Shortened video is ready! Segments: {len(self.pipeline.data.important_segments)}",
                )
            else:
                messagebox.showerror(
                    "Error",
                    f"Shortened video file not found at: {self.pipeline.data.shortened_video_path}",
                )
        else:
            messagebox.showerror(
                "Error", "Shortened video not found in pipeline results."
            )

    def on_video_processed_success(self):
        self.progress_var.set(
            f"Video shortened successfully! Saved to: {os.path.basename(self.shortened_video_path)}"
        )
        self.shorten_button.config(state=tk.NORMAL)
        self.load_shortened_video()
        messagebox.showinfo("Success", "Video has been shortened successfully!")

    def on_video_processed_error(self, error_msg):
        self.progress_var.set("Error occurred during processing.")
        self.shorten_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", f"Failed to process video: {error_msg}")

    def save_shortened_video(self):
        if not self.shortened_video_path or not os.path.exists(
            self.shortened_video_path
        ):
            messagebox.showerror("Error", "No shortened video to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Shortened Video",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*"),
            ],
        )

        if save_path:
            try:
                # Copy the shortened video to the selected location
                import shutil

                shutil.copy2(self.shortened_video_path, save_path)
                messagebox.showinfo("Success", f"Video saved to: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save video: {str(e)}")

    def on_closing(self):
        # Clean up video captures and audio
        self.original_playing = False
        self.shortened_playing = False

        # Stop audio
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        if self.original_cap:
            self.original_cap.release()
        if self.shortened_cap:
            self.shortened_cap.release()

        # Clean up temporary audio files
        for temp_file in self.temp_audio_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file}: {e}")

        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoShorteningGUI(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()
