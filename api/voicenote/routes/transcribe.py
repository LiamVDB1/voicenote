from __future__ import annotations
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..audio import probe_duration_sec, to_wav_16k_mono
from ..auth import current_user
from ..config import settings
from ..db import Transcript, User, get_session
from ..engines import ENGINE_NAMES, EngineResult, get_engine

log = logging.getLogger("voicenote.transcribe")

router = APIRouter()

# Global concurrency gate: prevents CPU thrash when multiple uploads land at once.
# Tunable via VN_MAX_CONCURRENT_JOBS.
_job_gate = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))


def _safe_filename(name: str) -> str:
    base = Path(name).name or "audio"
    return base[:255]


@router.post("/v1/transcribe")
async def transcribe(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    audio: Annotated[UploadFile, File(...)],
    engine: Annotated[str, Form()] = "auto",
    language: Annotated[str, Form()] = "auto",
) -> dict:
    if engine == "auto" or not engine:
        engine = settings.default_engine
    if engine not in ENGINE_NAMES:
        raise HTTPException(status_code=400, detail=f"Onbekende engine: {engine}")
    if language not in ("auto", "nl", "en", "fr", "de", "es", "it", "pt"):
        language = "auto"

    filename = _safe_filename(audio.filename or "audio.bin")

    cap = settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="vn_") as td_str:
        td = Path(td_str)
        src = td / filename
        size = 0
        with src.open("wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bestand groter dan {settings.max_upload_mb} MB",
                    )
                f.write(chunk)

        duration = await probe_duration_sec(src)
        if duration is not None and duration > settings.max_audio_minutes * 60:
            raise HTTPException(
                status_code=413,
                detail=f"Audio langer dan {settings.max_audio_minutes} minuten",
            )

        wav = td / "audio.wav"
        try:
            await to_wav_16k_mono(src, wav)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Kon audio niet lezen: {e}")

        async with _job_gate:
            result, used_engine, fallback, attempts = await _run_with_cascade(
                engine, wav, language
            )

    rec = Transcript(
        user_id=user.id,
        original_filename=filename,
        audio_size_bytes=size,
        duration_sec=duration,
        language=language,
        detected_language=result.detected_language,
        engine=used_engine,
        fallback_used=fallback,
        text=result.text,
        segments_json=json.dumps([
            {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
        ]) if result.segments else None,
    )
    session.add(rec)
    await session.commit()
    await session.refresh(rec)

    return {
        "id": rec.id,
        "text": result.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
        ],
        "language": result.detected_language or language,
        "engine": used_engine,
        "requested_engine": engine,
        "fallback_used": fallback,
        "attempts": attempts,
        "duration_sec": duration,
        "original_filename": filename,
        "created_at": rec.created_at.isoformat(),
    }


async def _run_with_cascade(
    requested: str, wav: Path, language: str
) -> tuple[EngineResult, str, bool, list[dict]]:
    """
    Try the requested engine first, then walk the configured fallback chain
    (skipping the requested engine to avoid loops). Each attempt is logged.
    """
    chain: list[str] = [requested]
    for name in settings.fallback_engines:
        if name in ENGINE_NAMES and name not in chain:
            chain.append(name)

    attempts: list[dict] = []
    last_error: Exception | None = None

    for idx, name in enumerate(chain):
        engine = get_engine(name)
        t0 = time.time()
        try:
            if not await engine.is_ready():
                attempts.append({"engine": name, "ok": False, "error": "not configured"})
                log.info("skip %s — not configured", name)
                continue
            result = await engine.transcribe(
                wav, language=language, timeout=settings.transcribe_timeout_sec
            )
            elapsed = time.time() - t0
            attempts.append({"engine": name, "ok": True, "elapsed_sec": round(elapsed, 2)})
            log.info("transcribe ok engine=%s elapsed=%.1fs", name, elapsed)
            return result, name, idx > 0, attempts
        except Exception as e:
            elapsed = time.time() - t0
            attempts.append({
                "engine": name, "ok": False,
                "error": str(e)[:200],
                "elapsed_sec": round(elapsed, 2),
            })
            log.warning("engine %s failed err=%s", name, e)
            last_error = e

    detail = "Geen beschikbare engine. " + ("; ".join(
        f"{a['engine']}: {a.get('error', 'ok')}" for a in attempts
    ) or str(last_error or "unknown"))
    raise HTTPException(status_code=500, detail=detail)
