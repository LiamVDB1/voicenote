from __future__ import annotations
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .. import jobs
from ..audio import probe_duration_sec
from ..auth import current_user
from ..config import settings
from ..db import User
from ..engines import ENGINE_NAMES

router = APIRouter()


def _safe_filename(name: str) -> str:
    base = Path(name).name or "audio"
    return base[:255]


@router.post("/v1/transcribe", status_code=202)
async def transcribe(
    user: Annotated[User, Depends(current_user)],
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

    job_id = uuid.uuid4().hex
    filename = _safe_filename(audio.filename or "audio.bin")
    upload_dir = settings.data_dir / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    src = upload_dir / filename

    try:
        size = await _stream_upload(audio, src)
        duration = await probe_duration_sec(src)
        if duration is not None and duration > settings.max_audio_minutes * 60:
            raise HTTPException(
                status_code=413,
                detail=f"Audio langer dan {settings.max_audio_minutes} minuten",
            )

        job = await jobs.enqueue(
            user_id=user.id,
            filename=filename,
            engine=engine,
            language=language,
            audio_path=src,
            audio_size=size,
            job_id=job_id,
        )
        return job.to_dict()
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Kon upload niet verwerken: {e}") from e


async def _stream_upload(audio: UploadFile, dst: Path) -> int:
    cap = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dst.open("wb") as f:
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
    return size
