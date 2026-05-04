from __future__ import annotations
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import Transcript, User, get_session

router = APIRouter()


def _summary(t: Transcript) -> dict:
    return {
        "id": t.id,
        "original_filename": t.original_filename,
        "language": t.detected_language or t.language,
        "engine": t.engine,
        "fallback_used": t.fallback_used,
        "duration_sec": t.duration_sec,
        "audio_size_bytes": t.audio_size_bytes,
        "created_at": t.created_at.isoformat(),
        "snippet": (t.text or "")[:160],
    }


@router.get("/v1/transcripts")
async def list_transcripts(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    total_q = await session.execute(
        select(func.count(Transcript.id)).where(Transcript.user_id == user.id)
    )
    total = total_q.scalar_one()
    result = await session.execute(
        select(Transcript)
        .where(Transcript.user_id == user.id)
        .order_by(Transcript.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [_summary(t) for t in result.scalars().all()]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/v1/transcripts/{transcript_id}")
async def get_transcript(
    transcript_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    t = await session.get(Transcript, transcript_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="Transcript niet gevonden")
    segments = json.loads(t.segments_json) if t.segments_json else []
    return {
        "id": t.id,
        "original_filename": t.original_filename,
        "text": t.text,
        "segments": segments,
        "language": t.detected_language or t.language,
        "engine": t.engine,
        "fallback_used": t.fallback_used,
        "duration_sec": t.duration_sec,
        "audio_size_bytes": t.audio_size_bytes,
        "created_at": t.created_at.isoformat(),
    }


@router.delete("/v1/transcripts/{transcript_id}")
async def delete_transcript(
    transcript_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    t = await session.get(Transcript, transcript_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status_code=404, detail="Transcript niet gevonden")
    await session.delete(t)
    await session.commit()
    return {"ok": True}
