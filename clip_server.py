# clip_server.py
"""
Clip + Caption server (updated)

Changes:
 - Accept optional `resolution` (e.g. "1080x1920") in requests for /clip and /clip_with_caption
 - Resize/reencode output to requested resolution (scale+pad to preserve aspect ratio)
 - Resize before burning/embedding subtitles so final output matches requested size

Usage example request JSON:
{
  "url":"https://www.youtube.com/watch?v=v52S3LBFZJs",
  "start":10,
  "end":40,
  "burn":true,
  "resolution":"1080x1920",
  "words_per_line":3,
  "fontsize":64
}

Requirements: same as before (ffmpeg, yt-dlp, optional ASR backends)
"""

import os
import shlex
import math
import subprocess
import tempfile
import asyncio
import logging
import importlib
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse

# load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clip_server")

# Optional OpenAI API key for remote Whisper
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Clip + Caption Server (OpenAI-first, fallback local) - resizable")

# ----------------- Request models -----------------
class ClipRequest(BaseModel):
    url: HttpUrl
    start: float
    end: float
    resolution: Optional[str] = None  # e.g. "1920x1080" or "1080x1920"

class ClipCaptionRequest(ClipRequest):
    model: Optional[str] = "small"   # local model size (tiny,base,small,medium,large) or "whisper-1" to prefer OpenAI
    burn: Optional[bool] = False
    language: Optional[str] = None
    words_per_line: Optional[int] = 3
    fontsize: Optional[int] = 56
    margin_v: Optional[int] = 50

# ----------------- Subprocess runner -----------------

def run_cmd(cmd: List[str], timeout: int = None) -> subprocess.CompletedProcess:
    logger.debug("Run command: %s", " ".join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return cp

# ----------------- Helpers for resolution parsing and resizing -----------------

def parse_resolution(res: Optional[str]) -> Optional[Tuple[int, int]]:
    if not res:
        return None
    if isinstance(res, str) and "x" in res:
        try:
            w_s, h_s = res.lower().split("x")
            w = int(w_s)
            h = int(h_s)
            if w <= 0 or h <= 0:
                raise ValueError("width/height must be > 0")
            return (w, h)
        except Exception as e:
            raise ValueError(f"Invalid resolution format '{res}', expected WIDTHxHEIGHT")
    raise ValueError(f"Invalid resolution format '{res}', expected WIDTHxHEIGHT")


def resize_video(input_path: str, out_path: str, resolution: Tuple[int, int], mode: str = "fill"):
    """
    Resize video to exact resolution.
    mode:
      - "fit": scale preserving aspect, then pad (no cropping) -> may produce black bars
      - "fill": scale preserving aspect then center-crop to fill (zoom & crop)
    """
    w, h = resolution

    if mode == "fit":
        # scale down/up to fit and pad to exact size (kept from earlier)
        vf = (
            f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),"
            f"pad={w}:{h}:({w}-iw*min({w}/iw\\,{h}/ih))/2:({h}-ih*min({w}/iw\\,{h}/ih))/2"
        )
    else:
        # fill (zoom & center-crop)
        vf = (
            f"scale=iw*max({w}/iw\\,{h}/ih):ih*max({w}/iw\\,{h}/ih),"
            f"crop={w}:{h}"
        )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out_path
    ]
    cp = run_cmd(cmd, timeout=600)
    if cp.returncode != 0:
        logger.error("resize ffmpeg failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("resize ffmpeg failed: " + (cp.stderr or cp.stdout or ""))
    logger.info("Resized video to %dx%d (mode=%s) -> %s", w, h, mode, out_path)


# ----------------- yt-dlp direct URLs -----------------
def get_direct_urls(youtube_url: str) -> List[str]:
    try:
        cp = run_cmd(["yt-dlp", "-f", "bestvideo+bestaudio/best", "-g", youtube_url], timeout=30)
    except Exception as e:
        logger.exception("yt-dlp error")
        raise HTTPException(status_code=500, detail=f"yt-dlp error: {e}")
    if cp.returncode != 0:
        msg = (cp.stderr or cp.stdout or "").strip()
        logger.error("yt-dlp failed: %s", msg)
        raise HTTPException(status_code=500, detail=f"yt-dlp failed: {msg}")
    lines = [l.strip() for l in (cp.stdout or "").splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=500, detail="yt-dlp returned no direct url")
    logger.info("yt-dlp returned %d direct url(s)", len(lines))
    return lines

# ----------------- FFmpeg cutting helpers -----------------
def run_ffmpeg_for_single_input(input_url: str, start_sec: float, duration_sec: float, out_path: str):
    cmd_copy = [
        "ffmpeg", "-i", input_url,
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-c", "copy", "-fflags", "+genpts", "-avoid_negative_ts", "make_zero", "-y", out_path
    ]
    cp = run_cmd(cmd_copy, timeout=300)
    if cp.returncode == 0:
        logger.info("ffmpeg single copy ok")
        return
    logger.warning("ffmpeg single copy failed: %s; trying reencode", cp.stderr or cp.stdout)
    cmd_re = [
        "ffmpeg", "-i", input_url,
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero", "-y", out_path
    ]
    cp2 = run_cmd(cmd_re, timeout=600)
    if cp2.returncode == 0:
        logger.info("ffmpeg single reencode ok")
        return
    logger.error("ffmpeg single both methods failed: %s", cp2.stderr or cp2.stdout)
    raise RuntimeError("ffmpeg single input failed: " + (cp2.stderr or cp2.stdout or ""))


def run_ffmpeg_for_two_inputs(video_url: str, audio_url: str, start_sec: float, duration_sec: float, out_path: str):
    cmd_map_copy = [
        "ffmpeg",
        "-i", video_url,
        "-i", audio_url,
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-ar", "48000", "-af", "aresample=48000:async=1",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-shortest", "-y", out_path
    ]
    cp = run_cmd(cmd_map_copy, timeout=600)
    if cp.returncode == 0:
        logger.info("ffmpeg combine copy succeeded")
        return
    logger.warning("ffmpeg combine copy failed: %s; trying reencode", cp.stderr or cp.stdout)
    cmd_re = [
        "ffmpeg",
        "-i", video_url,
        "-i", audio_url,
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-ar", "48000", "-af", "aresample=48000:async=1",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        "-shortest", "-y", out_path
    ]
    cp2 = run_cmd(cmd_re, timeout=900)
    if cp2.returncode == 0:
        logger.info("ffmpeg combine reencode succeeded")
        return
    logger.error("ffmpeg combine both methods failed: %s", cp2.stderr or cp2.stdout)
    raise RuntimeError("ffmpeg combine failed: " + (cp2.stderr or cp2.stdout or ""))


def run_ffmpeg_with_possible_audio(urls: List[str], start_sec: float, duration_sec: float, out_path: str):
    if len(urls) == 1:
        return run_ffmpeg_for_single_input(urls[0], start_sec, duration_sec, out_path)
    return run_ffmpeg_for_two_inputs(urls[0], urls[1], start_sec, duration_sec, out_path)

# ----------------- audio extraction -----------------
def extract_audio_wav(input_video: str, out_wav: str):
    cmd = ["ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", out_wav]
    cp = run_cmd(cmd, timeout=120)
    if cp.returncode != 0:
        logger.error("extract audio failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("extract audio failed: " + (cp.stderr or cp.stdout or ""))
    logger.info("extracted audio: %s", out_wav)

# ----------------- Local ASR helpers -----------------
_HAS_FASTER = importlib.util.find_spec("faster_whisper") is not None
_HAS_WHISPER = importlib.util.find_spec("whisper") is not None

def transcribe_audio_local(audio_path: str, model_size: str = "small", language: Optional[str] = None) -> List[dict]:
    if _HAS_FASTER:
        from faster_whisper import WhisperModel
        logger.info("Transcribing with faster_whisper model=%s", model_size)
        model = WhisperModel(model_size, device="auto", compute_type="float16")
        segments = []
        for seg in model.transcribe(audio_path, language=language, beam_size=5):
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": str(seg.text).strip()})
        return segments
    elif _HAS_WHISPER:
        import whisper
        logger.info("Transcribing with openai-whisper model=%s", model_size)
        model = whisper.load_model(model_size)
        kwargs = {}
        if language:
            kwargs["language"] = language
        res = model.transcribe(audio_path, **kwargs)
        segments = []
        for s in res.get("segments", []):
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"]).strip()})
        return segments
    else:
        raise RuntimeError("No local ASR backend installed (install faster-whisper or openai-whisper)")

# ----------------- OpenAI API ASR (fallback) -----------------
import requests

def transcribe_audio_openai(audio_path: str, model: str = "whisper-1", language: Optional[str] = None) -> List[dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": model, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    if resp.status_code != 200:
        logger.error("OpenAI ASR error: %s", resp.text)
        raise RuntimeError(f"OpenAI ASR failed: {resp.status_code} {resp.text}")
    result = resp.json()
    segments = []
    if "segments" in result:
        for seg in result["segments"]:
            segments.append({"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)), "text": seg.get("text", "").strip()})
    else:
        segments.append({"start": 0.0, "end": 1.0, "text": result.get("text", "").strip()})
    return segments

# ----------------- Prefer OpenAI then local wrapper -----------------
def transcribe_audio_prefer_openai(audio_path: str, openai_model: str = "whisper-1", local_model_size: str = "small", language: Optional[str] = None) -> List[dict]:
    # Try OpenAI first if key present
    if OPENAI_API_KEY:
        try:
            logger.info("Trying OpenAI ASR first")
            segments = transcribe_audio_openai(audio_path, model=openai_model, language=language)
            if segments and isinstance(segments, list) and len(segments) > 0:
                logger.info("OpenAI ASR success")
                return segments
            logger.warning("OpenAI ASR returned empty segments; falling back to local")
        except Exception as e:
            logger.warning("OpenAI ASR failed or quota issue: %s. Falling back to local ASR.", str(e))

    # Fallback local
    try:
        logger.info("Trying local ASR model=%s", local_model_size)
        segments = transcribe_audio_local(audio_path, model_size=local_model_size, language=language)
        if segments and isinstance(segments, list) and len(segments) > 0:
            logger.info("Local ASR success")
            return segments
        logger.warning("Local ASR returned no segments")
    except Exception as le:
        logger.exception("Local ASR failed: %s", le)

    raise RuntimeError("Transcription failed: both OpenAI API and local ASR unavailable or failed")

# ----------------- ASS generation (karaoke-style) -----------------
import html

def seconds_to_ass_time(s: float) -> str:
    if s < 0: s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    cs = int(round((s - math.floor(s)) * 100))
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

def generate_ass_from_segments(segments: List[dict],
                               ass_out_path: str,
                               words_per_line: int = 3,
                               fontname: str = "Arial",
                               fontsize: int = 56,
                               margin_v: int = 50,
                               outline: int = 3,
                               shadow: int = 1,
                               primary_color: str = "&H00FFFFFF"):
    def safe_text(t: str) -> str:
        t = t.replace("-->", "->")
        t = t.replace("{", "\\{").replace("}", "\\}")
        return html.escape(t)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: MyStyle,{fontname},{fontsize},{primary_color},&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    dialogues: List[str] = []

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 0.5))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        words = text.split()
        nwords = len(words)
        if nwords == 0:
            continue
        total_dur = max(0.05, seg_end - seg_start)
        per_word = total_dur / nwords
        for i in range(0, nwords, words_per_line):
            chunk_words = words[i:i+words_per_line]
            chunk_text = " ".join(chunk_words)
            chunk_start = seg_start + (i * per_word)
            chunk_end = min(seg_start + ((i + len(chunk_words)) * per_word), seg_end)
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + max(0.05, per_word)
            ass_text = safe_text(chunk_text)
            start_ts = seconds_to_ass_time(chunk_start)
            end_ts = seconds_to_ass_time(chunk_end)
            dialogue = f"Dialogue: 0,{start_ts},{end_ts},MyStyle,,0,0,0,,{ass_text}"
            dialogues.append(dialogue)

    with open(ass_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + dialogues))
    logger.info("Wrote ASS file: %s (%d dialogues)", ass_out_path, len(dialogues))

# ----------------- subtitle embed/burn robust -----------------
def embed_subtitles_soft(input_video: str, srt_or_ass: str, out_path: str):
    norm = srt_or_ass.replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", norm,
        "-map", "0:v",
        "-map", "0:a?",
        "-map", "1:0",
        # re-encode video to ensure compatibility with boxed resolution if needed
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        out_path
    ]
    cp = run_cmd(cmd, timeout=180)
    if cp.returncode == 0:
        logger.info("embed subtitles success: %s", out_path)
        return
    logger.warning("embed subtitles failed, trying MKV fallback: %s", cp.stderr or cp.stdout)
    out_mkv = out_path.rsplit(".", 1)[0] + ".mkv"
    cmd2 = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", norm,
        "-map", "0",
        "-map", "1:0",
        "-c", "copy",
        "-c:s", "srt",
        out_mkv
    ]
    cp2 = run_cmd(cmd2, timeout=120)
    if cp2.returncode == 0:
        os.replace(out_mkv, out_path)
        logger.info("embed fallback MKV success: %s", out_path)
        return
    logger.error("embed fallback failed: %s", cp2.stderr or cp2.stdout)
    raise RuntimeError("embed subtitles failed: " + (cp.stderr or cp.stdout or ""))


def burn_subtitles_hard(input_video: str, ass_file: str, out_path: str):
    ass_path = None
    try:
        ass_norm = ass_file.replace("\\", "/")
        # if user passed .srt, convert to ass
        if ass_norm.lower().endswith(".srt"):
            fd, tmp_ass = tempfile.mkstemp(suffix=".ass")
            os.close(fd)
            cp_conv = run_cmd(["ffmpeg", "-y", "-f", "srt", "-i", ass_norm, tmp_ass], timeout=40)
            if cp_conv.returncode != 0:
                logger.error("SRT->ASS conversion failed: %s", cp_conv.stderr or cp_conv.stdout)
                raise RuntimeError("SRT->ASS conversion failed")
            ass_norm = tmp_ass
            ass_path = tmp_ass
        # escape ":" and spaces and single quotes
        ass_esc = ass_norm.replace(":", r"\:").replace(" ", r"\ ").replace("'", r"\'")
        vf = f"ass='{ass_esc}'"
        cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", vf, "-c:a", "copy", out_path]
        cp = run_cmd(cmd, timeout=300)
        if cp.returncode == 0:
            logger.info("burn ASS success: %s", out_path)
            return
        logger.error("burn ASS failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("burn ASS failed: " + (cp.stderr or cp.stdout or ""))
    finally:
        if ass_path and os.path.exists(ass_path):
            try: os.remove(ass_path)
            except Exception: pass

# ----------------- cleanup -----------------
def remove_file_later(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("removed temp file %s", path)
    except Exception as e:
        logger.warning("failed to remove temp file %s: %s", path, e)

# ----------------- API endpoints -----------------
@app.post("/clip")
async def clip_endpoint(req: ClipRequest, background: BackgroundTasks):
    start = float(req.start); end = float(req.end)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    duration = end - start

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out_path = tmp_out.name; tmp_out.close()

    loop = asyncio.get_event_loop()
    try:
        urls = await loop.run_in_executor(None, get_direct_urls, str(req.url))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_direct_urls failed")
        raise HTTPException(status_code=500, detail=f"yt-dlp error: {e}")

    try:
        await loop.run_in_executor(None, run_ffmpeg_with_possible_audio, urls, start, duration, tmp_out_path)
    except Exception as e:
        try: os.remove(tmp_out_path)
        except Exception: pass
        logger.exception("ffmpeg cut failed")
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")

    # optional resize
    resized_path = None
    try:
        res = parse_resolution(req.resolution)
        if res:
            fd, resized_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            await loop.run_in_executor(None, resize_video, tmp_out_path, resized_path, res)
            # schedule cleanup of original
            try: os.remove(tmp_out_path)
            except Exception: pass
            tmp_out_path = resized_path
    except ValueError as ve:
        try: os.remove(tmp_out_path)
        except Exception: pass
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        try: os.remove(tmp_out_path)
        except Exception: pass
        logger.exception("resize failed")
        raise HTTPException(status_code=500, detail=f"resize failed: {e}")

    background.add_task(remove_file_later, tmp_out_path)
    filename = f"clip-{Path(tmp_out_path).stem}.mp4"
    return FileResponse(tmp_out_path, media_type="video/mp4", filename=filename)

@app.post("/clip_with_caption")
async def clip_with_caption_endpoint(req: ClipCaptionRequest, background: BackgroundTasks):
    start = float(req.start); end = float(req.end)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    duration = end - start

    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp_video_path = tmp_video.name; tmp_video.close()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav"); tmp_wav_path = tmp_wav.name; tmp_wav.close()
    tmp_ass = tempfile.NamedTemporaryFile(delete=False, suffix=".ass"); tmp_ass_path = tmp_ass.name; tmp_ass.close()
    out_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); out_final_path = out_final.name; out_final.close()

    tmp_video_resized = None

    loop = asyncio.get_event_loop()
    try:
        urls = await loop.run_in_executor(None, get_direct_urls, str(req.url))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_direct_urls failed")
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        raise HTTPException(status_code=500, detail=f"yt-dlp error: {e}")

    # 1) cut clip
    try:
        await loop.run_in_executor(None, run_ffmpeg_with_possible_audio, urls, start, duration, tmp_video_path)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("ffmpeg cut failed")
        raise HTTPException(status_code=500, detail=f"ffmpeg cut failed: {e}")

    # 2) extract audio
    try:
        await loop.run_in_executor(None, extract_audio_wav, tmp_video_path, tmp_wav_path)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("extract audio failed")
        raise HTTPException(status_code=500, detail=f"extract audio failed: {e}")

    # 3) transcribe (prefer OpenAI API then local)
    try:
        local_model_size = req.model if req.model in ("tiny","base","small","medium","large") else "small"
        segments = await loop.run_in_executor(None, transcribe_audio_prefer_openai, tmp_wav_path, "whisper-1", local_model_size, req.language)
        if not segments:
            logger.warning("transcription returned no segments")
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")

    # 4) generate ASS (karaoke style)
    try:
        words_per_line = int(req.words_per_line) if req.words_per_line else 3
        fontsize = int(req.fontsize) if req.fontsize else 56
        margin_v = int(req.margin_v) if req.margin_v else 50
        await loop.run_in_executor(None, generate_ass_from_segments, segments, tmp_ass_path, words_per_line, "Arial", fontsize, margin_v)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("generate ASS failed")
        raise HTTPException(status_code=500, detail=f"generate ASS failed: {e}")

    # optional resize before burn/embed so final matches requested resolution
    try:
        res = parse_resolution(req.resolution)
        if res:
            fd, tmp_video_resized = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            await loop.run_in_executor(None, resize_video, tmp_video_path, tmp_video_resized, res)
            # keep resized path as input for subtitle steps
            working_video_for_sub = tmp_video_resized
        else:
            working_video_for_sub = tmp_video_path
    except ValueError as ve:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else ""):
            try: os.remove(p)
            except Exception: pass
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else ""):
            try: os.remove(p)
            except Exception: pass
        logger.exception("resize failed")
        raise HTTPException(status_code=500, detail=f"resize failed: {e}")

    # 5) burn or embed onto working_video_for_sub
    try:
        if req.burn:
            await loop.run_in_executor(None, burn_subtitles_hard, working_video_for_sub, tmp_ass_path, out_final_path)
        else:
            await loop.run_in_executor(None, embed_subtitles_soft, working_video_for_sub, tmp_ass_path, out_final_path)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else ""):
            try: os.remove(p)
            except Exception: pass
        logger.exception("subtitle processing failed")
        raise HTTPException(status_code=500, detail=f"subtitle processing failed: {e}")

    # schedule cleanup
    for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, tmp_video_resized) :
        if p:
            background.add_task(remove_file_later, p)
    background.add_task(remove_file_later, out_final_path)

    filename = f"clip-caption-{Path(out_final_path).stem}.mp4"
    return FileResponse(out_final_path, media_type="video/mp4", filename=filename)

@app.get("/health")
async def health():
    return {"ok": True}
