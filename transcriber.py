# pip install yt-dlp youtube-transcript-api openai
import os, json, math, tempfile, subprocess, shutil
from pathlib import Path
import yt_dlp
from openai import OpenAI
from typing import List, Dict, Any, Optional

# ---- Config ----
AUDIO_CODEC = "mp3"
AUDIO_ABR   = "48k"
AUDIO_AR    = "16000"
AUDIO_AC    = "1"

INPUT_DIR   = Path("input")   # where we store the original mp4 and derived mp3
OUTPUT_DIR  = Path("output")  # where we store srt/txt/json

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

    # ---------- Download full MP4 video into input/<name>/video.mp4 ----------
    def download_video_mp4(self, url: str, dest_dir: Path) -> str:
        dest_dir.mkdir(parents=True, exist_ok=True)
        # outtmpl WITHOUT extension; yt-dlp will add .mp4 after merge
        outtmpl = str(dest_dir / "video.%(ext)s")
        ydl_opts = {
            # best video + best audio; if not possible, fallback to best single stream
            "format": "bv*+ba/b",
            # Ensure we get MP4 container on merge when possible
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Resolve actual .mp4 path (could be mp4/m4a/etc; we forced mp4 merge)
        # If merge not possible, best effort: pick the mp4 if present, else any video file.
        cand = list(dest_dir.glob("video.mp4"))
        if cand:
            return str(cand[0])

        # Fallback in rare cases
        vids = sorted(dest_dir.glob("video.*"))
        if not vids:
            raise RuntimeError("Video download failed")
        return str(vids[0])

    # ---------- Extract mono/16k/48kbit MP3 from MP4 into input/<name>/audio.mp3 ----------
    def extract_audio_from_video(self, video_path: str, dest_dir: Path) -> str:
        dest_dir.mkdir(parents=True, exist_ok=True)
        audio_path = dest_dir / f"audio.{AUDIO_CODEC}"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", AUDIO_AC, "-ar", AUDIO_AR, "-b:a", AUDIO_ABR,
            str(audio_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(audio_path)

    def get_duration_seconds(self, path: str) -> float:
        # requires ffmpeg/ffprobe installed
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        return float(out.strip())

    # ---------- Split with ffmpeg if needed (size threshold only) ----------
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

    # ---------- Split by model duration OR size ----------
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
            )
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

    # ---------- Save helpers (to OUTPUT_DIR/<name>/...) ----------
    @staticmethod
    def _srt_time(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def save_all(self, data: Dict[str, Any], out_base: Path, include_json=False):
        out_base.parent.mkdir(parents=True, exist_ok=True)

        # TXT (word-per-line with spans)
        txt_lines = []
        for w in data.get("words", []):
            txt_lines.append(f"[{self._srt_time(w['start'])} - {self._srt_time(w['end'])}] {w['word']}")
        (out_base.with_suffix(".txt")).write_text("\n".join(txt_lines) or data.get("text", ""), encoding="utf-8")

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
        (out_base.with_suffix(".srt")).write_text("\n".join(srt_lines), encoding="utf-8")

        if include_json:
            (out_base.with_suffix(".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Public entry ----------
    def transcribe_youtube(self, youtube_url: str, name: str, include_raw_json=False) -> Dict[str, Any]:
        """
        name: used as folder name under input/ and output/
        - MP4 at:   input/<name>/video.mp4
        - MP3 at:   input/<name>/audio.mp3
        - Outputs:  output/<name>/<name>.srt/.txt/(.json)
        """
        input_subdir  = INPUT_DIR / name
        output_subdir = OUTPUT_DIR / name

        # 0) Try official captions first (fast)
        captions = self.try_youtube_captions(youtube_url)
        if captions and captions.get("words"):
            out_base = output_subdir / name
            self.save_all(captions, out_base, include_json=include_raw_json)
            return captions

        # 1) Download MP4 into input/<name>/
        mp4_path = self.download_video_mp4(youtube_url, input_subdir)

        # 2) Extract MP3 (mono/16k/48k) next to MP4
        audio_path = self.extract_audio_from_video(mp4_path, input_subdir)

        # 3) Segment if needed (duration/size)
        part_files = self.split_by_duration(audio_path, CHUNK_SEC)

        # 4) Transcribe each part
        results = [self.transcribe_file(p) for p in part_files]

        # 5) Merge
        merged = self.merge_transcripts(results, part_files)

        # 6) Save SRT/TXT/(JSON) in output/<name>/
        out_base = output_subdir / name
        self.save_all(merged, out_base, include_json=include_raw_json)

        # 7) Cleanup temp split dir if we created one
        if part_files:
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
    print(f"✅ done:\n  Video:  {INPUT_DIR/out/'video.mp4'}\n  Audio:  {INPUT_DIR/out/'audio.mp3'}\n  Output: {OUTPUT_DIR/out/(out+'.srt')}, {OUTPUT_DIR/out/(out+'.txt')}" + (f", {OUTPUT_DIR/out/(out+'.json')}" if include else ""))