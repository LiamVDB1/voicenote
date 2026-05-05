"""
Voxtral Mini via Mistral's hosted API. Roughly $1 per 1,000 minutes of audio
(voxtral-mini-transcribe pricing). Best Dutch ASR quality of the three engines
in this project — and zero local compute.

For audio longer than the per-request cap (Mistral: 25 min), the engine splits
the wav into chunks of `voxtral_chunk_minutes` minutes, transcribes them in
parallel through the API (bounded by `voxtral_max_parallel_chunks`), then
concatenates the results in time order. Segment timestamps are offset back to
absolute positions in the original recording.
"""
from __future__ import annotations
import asyncio
import contextlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..audio import probe_duration_sec
from ..config import settings
from .base import EngineResult, Segment


@dataclass
class _Chunk:
    path: Path
    start_sec: float
    duration_sec: float


class VoxtralEngine:
    name = "voxtral"
    cpu_bound = False

    async def is_ready(self) -> bool:
        return bool(settings.mistral_api_key)

    async def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        timeout: int = 3600,
        progress=None,
    ) -> EngineResult:
        if not settings.mistral_api_key:
            raise RuntimeError("Mistral API-sleutel niet geconfigureerd (VN_MISTRAL_API_KEY)")

        duration = await probe_duration_sec(wav_path) or 0.0
        cap_sec = settings.voxtral_max_minutes * 60

        if duration <= cap_sec:
            return await self._transcribe_one(
                wav_path, language=language, timeout=timeout,
                progress=progress, duration_hint=duration,
            )

        return await self._transcribe_chunked(
            wav_path, language=language, timeout=timeout,
            progress=progress, total_duration=duration,
        )

    # ------------------------------------------------------------------ split
    async def _split_audio(self, wav_path: Path, chunk_sec: int) -> list[_Chunk]:
        """Split wav into ≤chunk_sec chunks via ffmpeg's segment muxer.
        Stream-copy (no re-encode) so it's fast and lossless."""
        out_dir = wav_path.parent / "chunks"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        pattern = str(out_dir / "chunk_%03d.wav")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-f", "segment",
            "-segment_time", str(chunk_sec),
            "-reset_timestamps", "1",
            "-c", "copy",
            pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RuntimeError(f"ffmpeg split failed: {err.decode('utf-8', 'replace')[:300]}")

        chunk_files = sorted(out_dir.glob("chunk_*.wav"))
        if not chunk_files:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RuntimeError("ffmpeg split produced no chunks")

        chunks: list[_Chunk] = []
        for i, p in enumerate(chunk_files):
            d = await probe_duration_sec(p) or float(chunk_sec)
            chunks.append(_Chunk(path=p, start_sec=i * float(chunk_sec), duration_sec=d))
        return chunks

    # -------------------------------------------------------------- chunked
    async def _transcribe_chunked(
        self,
        wav_path: Path,
        language: str,
        timeout: int,
        progress,
        total_duration: float,
    ) -> EngineResult:
        chunk_sec = settings.voxtral_chunk_minutes * 60
        chunks = await self._split_audio(wav_path, chunk_sec)
        n = len(chunks)

        # Per-chunk progress; the overall progress is the average across chunks.
        chunk_progress = [0.0] * n

        def make_cb(idx: int):
            def cb(p: float) -> None:
                # Clamp + avoid going backwards
                if p > chunk_progress[idx]:
                    chunk_progress[idx] = max(0.0, min(1.0, p))
                if progress is not None:
                    progress(sum(chunk_progress) / n)
            return cb

        sem = asyncio.Semaphore(max(1, settings.voxtral_max_parallel_chunks))

        async def run_chunk(idx: int, chunk: _Chunk) -> EngineResult:
            async with sem:
                return await self._transcribe_one(
                    chunk.path,
                    language=language,
                    timeout=timeout,
                    progress=make_cb(idx),
                    duration_hint=chunk.duration_sec,
                )

        try:
            tasks = [asyncio.create_task(run_chunk(i, c)) for i, c in enumerate(chunks)]
            results = await asyncio.gather(*tasks)
        finally:
            # Always clean up chunk files, even on cancel/error.
            chunks_dir = wav_path.parent / "chunks"
            shutil.rmtree(chunks_dir, ignore_errors=True)

        # Stitch text + segments back together with absolute timestamps.
        text_parts: list[str] = []
        all_segments: list[Segment] = []
        detected_language: str | None = None
        for chunk, res in zip(chunks, results):
            if res.text.strip():
                text_parts.append(res.text.strip())
            if not detected_language and res.detected_language:
                detected_language = res.detected_language
            for s in res.segments:
                all_segments.append(Segment(
                    start=s.start + chunk.start_sec,
                    end=s.end + chunk.start_sec,
                    text=s.text,
                ))

        if progress is not None:
            progress(1.0)

        full_text = "\n\n".join(text_parts).strip()
        return EngineResult(
            text=full_text,
            segments=all_segments,
            detected_language=detected_language or (language if language != "auto" else None),
            engine=self.name,
        )

    # ---------------------------------------------------------------- single
    async def _transcribe_one(
        self,
        wav_path: Path,
        language: str,
        timeout: int,
        progress,
        duration_hint: float | None = None,
    ) -> EngineResult:
        url = f"{settings.mistral_api_url.rstrip('/')}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}
        data: dict[str, str] = {"model": settings.voxtral_model_name}
        if language and language != "auto":
            data["language"] = language

        if progress:
            progress(0.05)

        loop = asyncio.get_event_loop()

        def _read_bytes() -> bytes:
            return wav_path.read_bytes()

        body_bytes = await loop.run_in_executor(None, _read_bytes)
        files = {"file": (wav_path.name, body_bytes, "audio/wav")}

        if progress:
            progress(0.10)

        # Time-based progress estimate while the request is in flight (Mistral
        # doesn't stream progress back). Conservative ~10× realtime.
        async def _tick_progress() -> None:
            if progress is None:
                return
            t0 = time.monotonic()
            est_total = max(3.0, (duration_hint or 60.0) / 10.0 + 3.0)
            while True:
                await asyncio.sleep(0.5)
                frac = min(0.95, (time.monotonic() - t0) / est_total)
                progress(0.10 + 0.80 * frac)

        ticker = asyncio.create_task(_tick_progress())
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(float(timeout), connect=30.0)) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)
        except httpx.HTTPError as e:
            ticker.cancel()
            with contextlib.suppress(BaseException):
                await ticker
            raise RuntimeError(f"Voxtral API onbereikbaar: {e}") from e
        finally:
            ticker.cancel()
            with contextlib.suppress(BaseException):
                await ticker

        if progress:
            progress(0.95)

        if resp.status_code >= 400:
            err_body = resp.text[:400] if resp.text else f"HTTP {resp.status_code}"
            raise RuntimeError(f"Voxtral API gaf {resp.status_code}: {err_body}")

        try:
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"Voxtral API gaf onleesbare response: {e}") from e

        text = (payload.get("text") or "").strip()
        detected_language = (
            payload.get("language")
            or (language if language and language != "auto" else None)
        )

        segments: list[Segment] = []
        for seg in payload.get("segments") or []:
            t = (seg.get("text") or "").strip()
            if not t:
                continue
            segments.append(Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=t,
            ))

        if progress:
            progress(1.0)

        return EngineResult(
            text=text,
            segments=segments,
            detected_language=detected_language,
            engine=self.name,
            raw_stdout=json.dumps(payload)[:2000],
        )
