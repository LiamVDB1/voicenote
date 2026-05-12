from __future__ import annotations
import asyncio
import json
from pathlib import Path


async def probe_duration_sec(path: Path) -> float | None:
    """Use ffprobe to get audio duration, in seconds."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(out)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


async def to_wav_16k_mono(src: Path, dst: Path) -> None:
    """Convert any audio to 16 kHz mono PCM s16le wav, suitable for both engines."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode('utf-8', 'replace')[:500]}")


async def detect_silences(
    wav_path: Path,
    noise_db: float = -30.0,
    min_duration_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Return all silences in the file as (start_sec, end_sec) pairs.

    Uses ffmpeg's `silencedetect` filter, which prints `silence_start: T` and
    `silence_end: T | silence_duration: D` to stderr. We parse those lines.
    Returns an empty list on any failure — callers should treat that as
    "no silences known" and fall back to fixed-time boundaries.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostats", "-hide_banner",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration_sec}",
        "-f", "null", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err_bytes = await proc.communicate()
    if proc.returncode != 0:
        return []
    out = err_bytes.decode("utf-8", "replace")

    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in out.splitlines():
        if "silence_start:" in line:
            try:
                pending_start = float(line.split("silence_start:", 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                pending_start = None
        elif "silence_end:" in line and pending_start is not None:
            try:
                end_str = line.split("silence_end:", 1)[1].strip().split()[0]
                # silencedetect prints "silence_end: T | silence_duration: D";
                # the split on whitespace already gives us T but strip "|" safety:
                end = float(end_str.rstrip("|"))
                if end > pending_start:
                    silences.append((pending_start, end))
            except (IndexError, ValueError):
                pass
            pending_start = None
    return silences
