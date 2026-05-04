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
