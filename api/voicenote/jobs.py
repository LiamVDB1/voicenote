"""
In-memory job system: each transcription request becomes a background job that
the user can leave behind. Survives within one process lifetime; completed jobs
also produce a persistent Transcript row, so the result is durable even after
restart. Active jobs are lost on restart (acceptable trade-off for a personal
deployment — the audio file remains on disk under /data/uploads until cleaned).
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from .audio import probe_duration_sec, to_wav_16k_mono
from .config import settings
from .db import _SessionLocal, Transcript
from .engines import ENGINE_NAMES, EngineResult, get_engine

log = logging.getLogger("voicenote.jobs")

# Concurrency gate. On a 4-core box, 1 is the right default.
_job_gate = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))

_jobs: dict[str, "Job"] = {}
_jobs_lock = asyncio.Lock()


class Job:
    __slots__ = (
        "id", "user_id", "filename", "engine_requested", "language",
        "status", "progress", "error", "transcript_id",
        "engine_used", "fallback_used", "attempts",
        "duration_sec", "audio_size_bytes",
        "created_at", "started_at", "finished_at",
        "_audio_path", "_task",
    )

    def __init__(
        self, *, user_id: int, filename: str, engine: str, language: str,
        audio_path: Path, audio_size: int, job_id: str | None = None,
    ):
        self.id = job_id or uuid.uuid4().hex
        self.user_id = user_id
        self.filename = filename
        self.engine_requested = engine
        self.language = language
        self.status: str = "queued"      # queued | running | done | failed | cancelled
        self.progress: float = 0.0
        self.error: str | None = None
        self.transcript_id: int | None = None
        self.engine_used: str | None = None
        self.fallback_used: bool = False
        self.attempts: list[dict] = []
        self.duration_sec: float | None = None
        self.audio_size_bytes = audio_size
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._audio_path = audio_path
        self._task: asyncio.Task | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "engine_requested": self.engine_requested,
            "engine_used": self.engine_used,
            "fallback_used": self.fallback_used,
            "language": self.language,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "transcript_id": self.transcript_id,
            "duration_sec": self.duration_sec,
            "audio_size_bytes": self.audio_size_bytes,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": (
                (self.finished_at or time.time()) - (self.started_at or self.created_at)
                if self.started_at else None
            ),
            "attempts": self.attempts,
        }


async def enqueue(
    *, user_id: int, filename: str, engine: str, language: str,
    audio_path: Path, audio_size: int, job_id: str | None = None,
) -> Job:
    job = Job(
        user_id=user_id, filename=filename, engine=engine, language=language,
        audio_path=audio_path, audio_size=audio_size, job_id=job_id,
    )
    async with _jobs_lock:
        _jobs[job.id] = job
    job._task = asyncio.create_task(_run(job))
    return job


async def _run(job: Job) -> None:
    # The CPU-bound gate is acquired per-attempt INSIDE the cascade so a Voxtral
    # job (network-only) doesn't queue behind a slow Whisper run, and the WAV
    # conversion / DB writes don't hold the gate either.
    try:
        job.status = "running"
        job.started_at = time.time()
        job.progress = 0.05

        wav_path = job._audio_path.with_suffix(".wav")
        try:
            await to_wav_16k_mono(job._audio_path, wav_path)
        except Exception as e:
            raise RuntimeError(f"Kon audio niet lezen: {e}") from e
        job.duration_sec = await probe_duration_sec(job._audio_path)
        job.progress = 0.10

        result, used_engine, fallback = await _run_cascade(job, wav_path)
        job.engine_used = used_engine
        job.fallback_used = fallback

        # Persist as Transcript so it shows up in history
        async with _SessionLocal() as session:
            rec = Transcript(
                user_id=job.user_id,
                original_filename=job.filename,
                audio_size_bytes=job.audio_size_bytes,
                duration_sec=job.duration_sec,
                language=job.language,
                detected_language=result.detected_language,
                engine=used_engine,
                fallback_used=fallback,
                text=result.text,
                segments_json=json.dumps([
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in result.segments
                ]) if result.segments else None,
            )
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            job.transcript_id = rec.id

        job.status = "done"
        job.progress = 1.0
        log.info("job %s done engine=%s fallback=%s elapsed=%.1fs",
                 job.id, used_engine, fallback,
                 time.time() - (job.started_at or time.time()))
    except asyncio.CancelledError:
        log.info("job %s cancelled", job.id)
        job.status = "cancelled"
        job.error = "Geannuleerd"
        raise
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.status = "failed"
        job.error = str(e)[:500]
    finally:
        job.finished_at = time.time()
        _cleanup_audio_files(job._audio_path)


async def _run_cascade(job: Job, wav: Path) -> tuple[EngineResult, str, bool]:
    requested = job.engine_requested
    chain: list[str] = [requested]
    for name in settings.fallback_engines:
        if name in ENGINE_NAMES and name not in chain:
            chain.append(name)

    last_error: Exception | None = None
    saw_failed_ready_engine = False
    for idx, name in enumerate(chain):
        engine = get_engine(name)
        t0 = time.time()
        try:
            if not await engine.is_ready():
                job.attempts.append({"engine": name, "ok": False, "error": "not configured"})
                continue
            # Engines call this with [0.0, 1.0]. We map the engine's reported
            # progress into the 0.10–0.95 band of the overall job (the first
            # 10% is upload+convert, the last 5% is DB persistence).
            def _on_engine_progress(p: float, _job=job) -> None:
                try:
                    _job.progress = max(_job.progress, 0.10 + 0.85 * float(p))
                except Exception:
                    pass

            job.progress = 0.10
            cpu_bound = bool(getattr(engine, "cpu_bound", True))
            if cpu_bound:
                # Local CPU engines must serialise — running two would just
                # halve each other's speed on a 4-core box.
                async with _job_gate:
                    result = await engine.transcribe(
                        wav,
                        language=job.language,
                        timeout=settings.transcribe_timeout_sec,
                        progress=_on_engine_progress,
                    )
            else:
                # Network-only engines (Voxtral via Mistral API) skip the gate
                # entirely so they can run alongside CPU work.
                result = await engine.transcribe(
                    wav,
                    language=job.language,
                    timeout=settings.transcribe_timeout_sec,
                    progress=_on_engine_progress,
                )
            elapsed = time.time() - t0
            job.attempts.append({"engine": name, "ok": True, "elapsed_sec": round(elapsed, 2)})
            return result, name, saw_failed_ready_engine and name != requested
        except Exception as e:
            elapsed = time.time() - t0
            job.attempts.append({
                "engine": name, "ok": False,
                "error": str(e)[:200], "elapsed_sec": round(elapsed, 2),
            })
            last_error = e
            saw_failed_ready_engine = True

    detail = "Geen beschikbare engine. " + "; ".join(
        f"{a['engine']}: {a.get('error', 'ok')}" for a in job.attempts
    )
    raise RuntimeError(detail) from last_error


def _cleanup_audio_files(audio_path: Path) -> None:
    try:
        if audio_path.exists():
            audio_path.unlink()
        wav = audio_path.with_suffix(".wav")
        if wav.exists():
            wav.unlink()
        parent = audio_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        log.warning("cleanup failed for %s", audio_path, exc_info=True)


async def get_job(job_id: str) -> Job | None:
    async with _jobs_lock:
        return _jobs.get(job_id)


async def cancel_job(job: Job) -> Job:
    if job.status not in ("queued", "running"):
        return job
    if job._task is not None and not job._task.done():
        job._task.cancel()
    job.status = "cancelled"
    job.error = "Geannuleerd"
    job.finished_at = time.time()
    _cleanup_audio_files(job._audio_path)
    return job


async def list_user_jobs(user_id: int, max_age_sec: int = 24 * 3600) -> list[Job]:
    """Return the user's recent jobs (active + finished within window)."""
    cutoff = time.time() - max_age_sec
    async with _jobs_lock:
        jobs = [
            j for j in _jobs.values()
            if j.user_id == user_id and j.created_at >= cutoff
        ]
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


async def cleanup_old_jobs(max_age_sec: int = 7 * 24 * 3600) -> None:
    cutoff = time.time() - max_age_sec
    async with _jobs_lock:
        for jid in list(_jobs.keys()):
            j = _jobs[jid]
            if j.created_at < cutoff and j.status in ("done", "failed", "cancelled"):
                del _jobs[jid]
