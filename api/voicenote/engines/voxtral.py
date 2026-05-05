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
    async def _split_audio(
        self, wav_path: Path, chunk_sec: int, overlap_sec: float, total_duration: float,
    ) -> list[_Chunk]:
        """Cut the wav into chunks with a small overlap at the start of every
        chunk after the first. The boundary at exactly i*chunk_sec stays the
        natural split point; chunk i+1 just starts `overlap_sec` earlier so a
        word that gets sliced in two is still fully present in one of the two
        chunks. We dedupe the duplicate at stitch time.

        Each chunk is cut with its own ffmpeg invocation (stream-copy → fast,
        lossless). The segment muxer can't do per-chunk overlap.
        """
        out_dir = wav_path.parent / "chunks"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        chunks: list[_Chunk] = []
        i = 0
        while True:
            cut = i * chunk_sec
            start = max(0.0, float(cut) - (overlap_sec if i > 0 else 0.0))
            end = min(total_duration, float((i + 1) * chunk_sec))
            if start >= total_duration:
                break

            out_path = out_dir / f"chunk_{i:03d}.wav"
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-t", f"{(end - start):.3f}",
                "-i", str(wav_path),
                "-c", "copy",
                str(out_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                shutil.rmtree(out_dir, ignore_errors=True)
                raise RuntimeError(
                    f"ffmpeg chunk {i} failed: {err.decode('utf-8', 'replace')[:300]}"
                )

            chunks.append(_Chunk(path=out_path, start_sec=start, duration_sec=end - start))
            if end >= total_duration:
                break
            i += 1

        if not chunks:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RuntimeError("split produced no chunks")
        return chunks

    # ------------------------------------------------------------ stitcher
    @staticmethod
    def _merge_with_overlap(prev_text: str, next_text: str, max_overlap_words: int = 30) -> str:
        """Concatenate two adjacent chunk transcripts, stripping the leading
        words of `next_text` that already appeared at the tail of `prev_text`.

        Compares word-by-word, lowercased + punctuation-stripped, walking
        suffix lengths from largest to smallest until we find a match. If
        nothing matches we just join with a paragraph break.
        """
        if not prev_text:
            return next_text
        if not next_text:
            return prev_text

        prev_words = prev_text.split()
        next_words = next_text.split()
        if not prev_words or not next_words:
            return (prev_text + "\n\n" + next_text).strip()

        def norm(w: str) -> str:
            return w.lower().strip(".,!?;:\"'()…—–-")

        max_n = min(max_overlap_words, len(prev_words), len(next_words))
        best_n = 0
        for n in range(max_n, 1, -1):
            tail = [norm(w) for w in prev_words[-n:]]
            head = [norm(w) for w in next_words[:n]]
            if tail == head and any(t for t in tail):  # avoid matching empty/punctuation-only
                best_n = n
                break

        if best_n == 0:
            return (prev_text + "\n\n" + next_text).strip()
        merged_tail = " ".join(next_words[best_n:]).strip()
        if not merged_tail:
            return prev_text
        # Decide separator: same paragraph if the dedup was substantial, else newline
        sep = " " if best_n >= 3 else "\n"
        return prev_text + sep + merged_tail

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
        overlap_sec = settings.voxtral_chunk_overlap_sec
        chunks = await self._split_audio(wav_path, chunk_sec, overlap_sec, total_duration)
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

        # Stitch text with word-level dedup at the overlap seam, and rebuild
        # segments in absolute time, dropping the overlap region of every
        # chunk after the first (those seconds already exist in the previous
        # chunk's segments).
        full_text = ""
        all_segments: list[Segment] = []
        detected_language: str | None = None
        for idx, (chunk, res) in enumerate(zip(chunks, results)):
            if not detected_language and res.detected_language:
                detected_language = res.detected_language

            if idx == 0:
                full_text = res.text.strip()
            else:
                full_text = self._merge_with_overlap(full_text, res.text.strip())

            # For chunk i>=1, the first overlap_sec of internal time was already
            # transcribed in chunk i-1's tail — skip those segments.
            internal_skip = overlap_sec if idx > 0 else 0.0
            for s in res.segments:
                if s.start < internal_skip:
                    continue
                all_segments.append(Segment(
                    start=s.start + chunk.start_sec,
                    end=s.end + chunk.start_sec,
                    text=s.text,
                ))

        if progress is not None:
            progress(1.0)

        return EngineResult(
            text=full_text.strip(),
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
