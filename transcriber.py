# pip install yt-dlp openai
import os, json, tempfile, subprocess, shutil
from pathlib import Path
import yt_dlp
from openai import OpenAI
from typing import List, Dict, Any

# ---- Config ----
AUDIO_CODEC = "mp3"
AUDIO_ABR   = "48k"
AUDIO_AR    = "16000"
AUDIO_AC    = "1"

INPUT_DIR   = Path("input")   # stores original mp4 and derived mp3
OUTPUT_DIR  = Path("output")  # stores srt/txt/json

# Model hard limits
MAX_MODEL_SECONDS = 1400          # whisper-1 limit
CHUNK_SEC         = 1200          # safety buffer under 1400s
UPLOAD_CAP        = 25 * 1024 * 1024  # ~25MB

class YouTubeTranscriber:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    # ---------- Download full MP4 video into input/<name>/video.mp4 ----------
    def download_video_mp4(self, url: str, dest_dir: Path) -> str:
        dest_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(dest_dir / "video.%(ext)s")
        ydl_opts = {
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        cand = list(dest_dir.glob("video.mp4"))
        if cand:
            return str(cand[0])

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
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        return float(out.strip())

    # ---------- Split by model duration OR size ----------
    def split_by_duration(self, src_path: str, chunk_seconds: int = CHUNK_SEC) -> list[str]:
        dur = self.get_duration_seconds(src_path)
        if dur <= MAX_MODEL_SECONDS and os.path.getsize(src_path) < UPLOAD_CAP:
            return [src_path]

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

    def transcribe_file(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            tx = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        data = tx.model_dump() if hasattr(tx, "model_dump") else (tx.dict() if hasattr(tx, "dict") else dict(tx))

        # Keep native segments if the API provided them
        segments_raw = data.get("segments") or []

        # Normalize to words (for SRT-building / minute-bins, etc.)
        words = []
        if data.get("words"):
            words = data["words"]
        elif segments_raw:
            for seg in segments_raw:
                s_start = float(seg.get("start", 0.0))
                s_end   = float(seg.get("end", s_start))
                text    = (seg.get("text") or "").strip()
                if text:
                    words.append({"word": text, "start": s_start, "end": s_end})
        else:
            return {"text": data.get("text", ""), "words": [], "segments": []}

        return {"text": data.get("text", ""), "words": words, "segments": segments_raw}

    def merge_transcripts(self, parts: list[dict], part_files: list[str]) -> dict:
        # compute cumulative offsets from real file durations
        offsets, total = [], 0.0
        for pf in part_files:
            offsets.append(total)
            total += self.get_duration_seconds(pf)

        merged_words, merged_texts, merged_segments = [], [], []
        for i, part in enumerate(parts):
            off = offsets[i]
            merged_texts.append(part.get("text", ""))

            for w in part.get("words", []):
                start = float(w.get("start", 0.0)) + off
                end   = float(w.get("end", start)) + off
                merged_words.append({"word": w.get("word", ""), "start": start, "end": end})

            for seg in part.get("segments", []) or []:
                s = float(seg.get("start", 0.0)) + off
                e = float(seg.get("end", s)) + off
                merged_segments.append({"start": s, "end": e, "text": seg.get("text", "")})

        return {
            "text": " ".join(t for t in merged_texts if t),
            "words": merged_words,
            "segments": merged_segments,
        }


    # ---------- Utility: make sentence/phrase segments from words ----------
    @staticmethod
    def _words_to_segments(words: list[dict], max_duration: float = 5.0) -> list[dict]:
        segments = []
        phrase, cur_start = [], None
        for w in words:
            if cur_start is None:
                cur_start = float(w["start"])
            phrase.append(w["word"])
            end = float(w["end"])
            ends_sentence = phrase[-1].rstrip().endswith((".", "!", "?", ","))
            too_long = (end - cur_start) > max_duration
            if ends_sentence or too_long:
                text = " ".join(phrase).strip()
                if text:
                    segments.append({"start": cur_start, "end": end, "text": text})
                phrase, cur_start = [], None
        if phrase:
            end = float(words[-1]["end"]) if words else 0.0
            text = " ".join(phrase).strip()
            if text:
                segments.append({"start": cur_start or 0.0, "end": end, "text": text})
        return segments

    # ---------- Save helpers (to OUTPUT_DIR/<name>/...) ----------
    @staticmethod
    def _srt_time(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


    def save_all(self, data: Dict[str, Any], out_base: Path):
        out_base.parent.mkdir(parents=True, exist_ok=True)

        # Prefer native segments if present; else build from words
        segments = data.get("segments")
        if not segments:
            segments = self._words_to_segments(data.get("words", []))

        # ---- TXT (word-per-line) unchanged ----
        txt_lines = [
            f"[{self._srt_time(w['start'])} - {self._srt_time(w['end'])}] {w['word']}"
            for w in data.get("words", [])
        ]
        (out_base.with_suffix(".txt")).write_text(
            "\n".join(txt_lines) or data.get("text", ""), encoding="utf-8"
        )

        # ---- SRT from segments (unchanged) ----
        srt_lines, idx = [], 1
        for seg in segments:
            srt_lines += [
                str(idx),
                f"{self._srt_time(seg['start'])} --> {self._srt_time(seg['end'])}",
                seg["text"],
                ""
            ]
            idx += 1
        (out_base.with_suffix(".srt")).write_text("\n".join(srt_lines), encoding="utf-8")

        # ---- Format output JSON ----
        pretty = [
            {
                "start": self._srt_time(seg["start"]),
                "end":   self._srt_time(seg["end"]),
                "text":  seg["text"],
            }
            for seg in segments
        ]
        (out_base.with_suffix(".json")).write_text(
            json.dumps(pretty, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Optional per word JSON
        by_word_json_path = out_base.parent / f"{out_base.name}_by_word.json"
        by_word_json_path.write_text(
            json.dumps({"text": data.get("text", ""), "words": data.get("words", [])}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


    # ---------- Public entry ----------
    def transcribe_youtube(self, youtube_url: str, name: str) -> Dict[str, Any]:
        """
        name: used as folder name under input/ and output/
        - MP4 at:   input/<name>/video.mp4
        - MP3 at:   input/<name>/audio.mp3
        - Outputs:  output/<name>/<name>.srt/.txt/.json + <name>_by_word.json
        """
        input_subdir  = INPUT_DIR / name
        output_subdir = OUTPUT_DIR / name

        # 1) Download MP4
        mp4_path = self.download_video_mp4(youtube_url, input_subdir)

        # 2) Extract MP3 (mono/16k/48k)
        audio_path = self.extract_audio_from_video(mp4_path, input_subdir)

        # 3) Segment if needed (duration/size)
        part_files = self.split_by_duration(audio_path, CHUNK_SEC)

        # 4) Transcribe each part with Whisper 
        results = [self.transcribe_file(p) for p in part_files]

        # 5) Merge
        merged = self.merge_transcripts(results, part_files)

        # 6) Save artifacts
        out_base = output_subdir / name
        self.save_all(merged, out_base)

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
    out = input("Enter output name (folder & basename) [default: transcript]: ").strip() or "transcript"

    tx = YouTubeTranscriber(key).transcribe_youtube(url, out)
    print(
        "✅ done:\n"
        f"  Video:  {INPUT_DIR/out/'video.mp4'}\n"
        f"  Audio:  {INPUT_DIR/out/'audio.mp3'}\n"
        f"  Output: {OUTPUT_DIR/out/(out+'.srt')}, {OUTPUT_DIR/out/(out+'.txt')}, "
        f"{OUTPUT_DIR/out/(out+'.json')} (segments), {OUTPUT_DIR/out/(out+'_by_word.json')} (per-word)"
    )
