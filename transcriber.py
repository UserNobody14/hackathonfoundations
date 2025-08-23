# pip install yt-dlp youtube-transcript-api openai
import os, json, math, tempfile, subprocess, shutil
from pathlib import Path
import yt_dlp
from openai import OpenAI
from typing import List, Dict, Any, Optional

# ---- Config ----
# ---- Config ----
AUDIO_CODEC = "mp3"
AUDIO_ABR   = "48k"
AUDIO_AR    = "16000"
AUDIO_AC    = "1"

# Model hard limits
MAX_MODEL_SECONDS = 1400          # error shows 1400s max
CHUNK_SEC         = 1200          # keep a safety buffer (< 1400)
UPLOAD_CAP        = 25 * 1024 * 1024


class YouTubeTranscriber:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    # ---------- Try YouTube captions first ----------
    def try_youtube_captions(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
            from urllib.parse import parse_qs, urlparse
            vid = parse_qs(urlparse(url).query).get("v", [None])[0]
            if not vid:
                return None
            tracks = YouTubeTranscriptApi.list_transcripts(vid)
            # Prefer manually created, then auto
            track = (tracks.find_manually_created_transcript(tracks._languages) 
                     if hasattr(tracks, "find_manually_created_transcript") else None)
            if track is None:
                track = tracks.find_transcript(tracks._languages)  # may be auto
            entries = track.fetch()  # [{text,start,duration}, ...]
            # Normalize to "words" array of whole phrases (one per entry)
            words = []
            for e in entries:
                start = float(e["start"])
                end   = start + float(e.get("duration", 0.0))
                words.append({"word": e["text"], "start": start, "end": end})
            return {"text": " ".join(e["text"] for e in entries), "words": words}
        except Exception:
            return None

    # ---------- Download compressed audio only ----------
    def download_audio(self, url: str, out_base: str) -> str:
        out_path = f"{out_base}.%(ext)s"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": AUDIO_CODEC,  # e.g. "mp3" or "m4a"
                    "preferredquality": "0",
                },
            ],
            # extra ffmpeg params to force mono/16k/bitrate
            "postprocessor_args": [
                "-ac", AUDIO_AC,
                "-ar", AUDIO_AR,
                "-b:a", AUDIO_ABR,
            ],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # resolve actual file
        for p in Path(".").glob(f"{out_base}.*"):
            return str(p)
        raise RuntimeError("Audio download failed")

    def get_duration_seconds(self, path: str) -> float:
        # requires ffmpeg/ffprobe installed
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        return float(out.strip())

    # ---------- Split with ffmpeg if needed ----------
    def maybe_split(self, src_path: str, chunk_seconds: int = CHUNK_SEC) -> List[str]:
        if os.path.getsize(src_path) < UPLOAD_CAP:
            return [src_path]
        tmpdir = tempfile.mkdtemp(prefix="chunks_")
        outpat = os.path.join(tmpdir, "part_%03d." + AUDIO_CODEC)
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", AUDIO_AC, "-ar", AUDIO_AR, "-b:a", AUDIO_ABR,
            "-f", "segment", "-segment_time", str(chunk_seconds),
            outpat
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        parts = sorted(str(p) for p in Path(tmpdir).glob(f"part_*.{AUDIO_CODEC}"))
        return parts

    def split_by_duration(self, src_path: str, chunk_seconds: int = CHUNK_SEC) -> list[str]:
        dur = self.get_duration_seconds(src_path)

        # If within model duration and size limits, use as-is
        if dur <= MAX_MODEL_SECONDS and os.path.getsize(src_path) < UPLOAD_CAP:
            return [src_path]

        # Otherwise, segment by time into a temp dir
        tmpdir = tempfile.mkdtemp(prefix="chunks_")
        outpat = os.path.join(tmpdir, "part_%03d." + AUDIO_CODEC)
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", AUDIO_AC, "-ar", AUDIO_AR, "-b:a", AUDIO_ABR,
            "-f", "segment", "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            outpat
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        parts = sorted(str(p) for p in Path(tmpdir).glob(f"part_*.{AUDIO_CODEC}"))
        if not parts:
            raise RuntimeError("Segmentation produced no parts")
        return parts


    # ---------- Transcribe one file (chunk) ----------
    def transcribe_file(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            # Use current STT model; returns text + (for some variants) structured segments
            tx = self.client.audio.transcriptions.create(
                # model="gpt-4o-transcribe",
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]  # ask for words + segments
                # Some models don’t yet expose reliable per-word timestamps.
                # We’ll gracefully handle missing word granularity below.
            )
        # tx is a pydantic object; tx.text exists. Try to pull words if present.
        data = tx.dict() if hasattr(tx, "dict") else dict(tx)  # tolerate SDK changes
        # normalize: expect data.get("words") or reconstruct from segments if available
        words = []
        if "words" in data and data["words"]:
            words = data["words"]
        elif "segments" in data and data["segments"]:
            for seg in data["segments"]:
                s_start = float(seg.get("start", 0.0))
                s_end   = float(seg.get("end", s_start))
                text    = seg.get("text", "").strip()
                if text:
                    words.append({"word": text, "start": s_start, "end": s_end})
        else:
            # Fallback: no timing info, return a single span 0..0 (usable text, unusable timings)
            words = [{"word": data.get("text", ""), "start": 0.0, "end": 0.0}]
        return {"text": data.get("text", ""), "words": words}

    # ---------- Merge chunk transcripts with offsets ----------
    def merge_transcripts(self, parts: list[dict], part_files: list[str]) -> dict:
        # compute cumulative offsets from real file durations
        offsets = []
        total = 0.0
        for pf in part_files:
            offsets.append(total)
            total += self.get_duration_seconds(pf)

        merged_words, merged_texts = [], []
        for i, part in enumerate(parts):
            off = offsets[i]
            merged_texts.append(part.get("text", ""))
            for w in part.get("words", []):
                start = float(w.get("start", 0.0)) + off
                end   = float(w.get("end", start)) + off
                merged_words.append({"word": w.get("word", ""), "start": start, "end": end})
        return {"text": " ".join(t for t in merged_texts if t), "words": merged_words}


    # ---------- Save helpers ----------
    @staticmethod
    def _srt_time(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def save_all(self, data: Dict[str, Any], out_base: str, include_json=False):
        # TXT (word-per-line with spans)
        txt = []
        for w in data.get("words", []):
            txt.append(f"[{self._srt_time(w['start'])} - {self._srt_time(w['end'])}] {w['word']}")
        Path(f"{out_base}.txt").write_text("\n".join(txt) or data.get("text", ""), encoding="utf-8")

        # SRT (phrase up to 5s or end punctuation)
        srt_lines, idx, phrase, cur_start = [], 1, [], None
        for w in data.get("words", []):
            if cur_start is None:
                cur_start = w["start"]
            phrase.append(w["word"])
            end = w["end"]
            ends_sentence = any(phrase[-1].rstrip().endswith(p) for p in [".", "!", "?", ","])
            too_long = (end - cur_start) > 5.0
            if ends_sentence or too_long:
                text = " ".join(phrase).strip()
                if text:
                    srt_lines += [str(idx),
                                  f"{self._srt_time(cur_start)} --> {self._srt_time(end)}",
                                  text, ""]
                    idx += 1
                phrase, cur_start = [], None
        if phrase:
            end = data["words"][-1]["end"] if data.get("words") else 0.0
            text = " ".join(phrase).strip()
            if text:
                srt_lines += [str(idx),
                              f"{self._srt_time(cur_start or 0.0)} --> {self._srt_time(end)}",
                              text, ""]
        Path(f"{out_base}.srt").write_text("\n".join(srt_lines), encoding="utf-8")

        if include_json:
            Path(f"{out_base}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Public entry ----------
    def transcribe_youtube(self, youtube_url: str, out_base: str, include_raw_json=False) -> Dict[str, Any]:
        # 1) Try official captions (fast + already timestamped)
        captions = self.try_youtube_captions(youtube_url)
        if captions and captions.get("words"):
            self.save_all(captions, out_base, include_json=include_raw_json)
            return captions

        # 2) Pull compressed audio
        # audio_path = self.download_audio(youtube_url, out_base + "_audio")
        # parts = self.maybe_split(audio_path, CHUNK_SEC)

        # # 3) Transcribe each part
        # results = []
        # for p in parts:
        #     results.append(self.transcribe_file(p))

        # # 4) Merge (apply offsets)
        # merged = self.merge_transcripts(results, CHUNK_SEC)

        audio_path = self.download_audio(youtube_url, out_base + "_audio")
        part_files = self.split_by_duration(audio_path, CHUNK_SEC)
        results = [self.transcribe_file(p) for p in part_files]
        merged = self.merge_transcripts(results, part_files)

        self.save_all(merged, out_base, include_json=include_raw_json)

        # 5) Cleanup temp split dir if we created one
        # split_dir = Path(parts[0]).parent
        # if "chunks_" in str(split_dir):
        #     shutil.rmtree(split_dir, ignore_errors=True)

        split_dir = Path(part_files[0]).parent
        if "chunks_" in str(split_dir):
            shutil.rmtree(split_dir, ignore_errors=True)

        return merged


if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY.")
    url = input("Enter YouTube URL: ").strip()
    out = input("Enter output filename (default: transcript): ").strip() or "transcript"
    include = input("Include raw JSON? (y/n) [n]: ").strip().lower() == "y"
    tx = YouTubeTranscriber(key).transcribe_youtube(url, out, include_raw_json=include)
    print("✅ done:", out, f"({len(tx.get('words', []))} words)")
