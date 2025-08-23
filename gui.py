import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from cut import cut_and_stitch_video
import pygame
from moviepy import VideoFileClip
import tempfile


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

        # File selection frame
        file_frame = tk.Frame(self.root, bg="#f0f0f0")
        file_frame.pack(pady=10)

        select_button = tk.Button(
            file_frame,
            text="Select Video File",
            command=self.select_video_file,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="black",
            padx=20,
            pady=5,
            relief="raised",
            borderwidth=2,
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
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Action buttons frame
        action_frame = tk.Frame(self.root, bg="#f0f0f0")
        action_frame.pack(pady=20)

        # Timestamp input frame
        timestamp_frame = tk.Frame(action_frame, bg="#f0f0f0")
        timestamp_frame.pack(pady=10)

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

        self.timestamp_entry = tk.Entry(timestamp_frame, width=50, font=("Arial", 10))
        self.timestamp_entry.pack(pady=5)
        self.timestamp_entry.insert(0, "10,20;30,40")  # Default example

        # Shorten button
        self.shorten_button = tk.Button(
            action_frame,
            text="Shorten Video",
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

    def parse_timestamps(self, timestamp_text):
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

    def shorten_video(self):
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return

        timestamp_text = self.timestamp_entry.get().strip()
        if not timestamp_text:
            messagebox.showerror("Error", "Please enter timestamp pairs.")
            return

        try:
            timestamp_pairs = self.parse_timestamps(timestamp_text)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        # Create output path
        input_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        output_dir = os.path.join(os.path.dirname(self.current_video_path), "shortened")
        os.makedirs(output_dir, exist_ok=True)
        self.shortened_video_path = os.path.join(
            output_dir, f"{input_name}_shortened.mp4"
        )

        # Disable buttons during processing
        self.shorten_button.config(state=tk.DISABLED)
        self.progress_var.set("Processing video... Please wait.")

        # Run video shortening in a separate thread
        threading.Thread(
            target=self.process_video, args=(timestamp_pairs,), daemon=True
        ).start()

    def process_video(self, timestamp_pairs):
        try:
            if not self.current_video_path or not self.shortened_video_path:
                raise ValueError("Video paths not properly set")

            cut_and_stitch_video(
                timestamp_pairs=timestamp_pairs,
                input_video_path=self.current_video_path,
                output_video_path=self.shortened_video_path,
            )

            # Update UI in main thread
            self.root.after(0, self.on_video_processed_success)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.on_video_processed_error(error_msg))

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
