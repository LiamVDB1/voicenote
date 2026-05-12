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
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..audio import detect_silences, probe_duration_sec
from ..config import settings
from .base import EngineResult, Segment

log = logging.getLogger("voicenote.engines.voxtral")


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

    # -------------------------------------------------------------- boundaries
    @staticmethod
    def _pick_boundaries(
        target_chunk_sec: float,
        total_duration: float,
        silences: list[tuple[float, float]],
        search_window_sec: float,
    ) -> list[float]:
        """Pick chunk cut points. Starts with ideal targets at i*target_chunk_sec
        and snaps each to the centre of the longest silence within
        +/- search_window_sec. If no silence is within the window, the ideal
        target stands (the merge dedup handles a mid-utterance cut).

        Returns a strictly increasing list of cut times in (0, total_duration);
        the chunks are [0, cuts[0], cuts[1], ..., total_duration].
        """
        cuts: list[float] = []
        target = target_chunk_sec
        last_cut = 0.0
        while target < total_duration:
            best_mid: float | None = None
            best_dur = 0.0
            for s, e in silences:
                if e <= last_cut:
                    continue
                mid = (s + e) / 2.0
                if abs(mid - target) > search_window_sec:
                    continue
                dur = e - s
                if dur > best_dur:
                    best_dur = dur
                    best_mid = mid
            cut = best_mid if best_mid is not None else target
            # Guard against silences/targets we've already passed.
            if cut <= last_cut + 1.0:
                cut = max(last_cut + 1.0, target)
            cuts.append(cut)
            last_cut = cut
            target = cut + target_chunk_sec
        return cuts

    # ------------------------------------------------------------------ split
    async def _split_audio(
        self,
        wav_path: Path,
        boundaries: list[float],
        overlap_sec: float,
        total_duration: float,
    ) -> list[_Chunk]:
        """Cut the wav at the given boundary times. Each chunk after the first
        starts `overlap_sec` earlier so a word sliced at the boundary is fully
        present in one of the two chunks; the merge step dedupes the seam.

        Each chunk is cut with its own ffmpeg invocation (stream-copy → fast,
        lossless). The segment muxer can't do per-chunk overlap.
        """
        out_dir = wav_path.parent / "chunks"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        # boundaries are the *internal* cut points; build [(start,end), ...]:
        # first chunk runs 0..boundaries[0], next chunks each run
        # boundaries[i-1]..boundaries[i], last runs boundaries[-1]..total.
        edges = [0.0, *boundaries, total_duration]
        ranges = list(zip(edges, edges[1:]))

        chunks: list[_Chunk] = []
        for i, (cut, next_cut) in enumerate(ranges):
            start = max(0.0, cut - (overlap_sec if i > 0 else 0.0))
            end = min(total_duration, next_cut)
            if end - start < 0.5:
                continue

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

        if not chunks:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RuntimeError("split produced no chunks")
        return chunks

    # -------------------------------------------------------------- loop check
    @staticmethod
    def _detect_repetition_loop(
        text: str,
        min_ngram: int = 3,
        max_ngram: int = 12,
        min_repeats: int = 8,
    ) -> tuple[bool, str | None]:
        """Detect runaway ASR repetition: any word n-gram of length [min_ngram,
        max_ngram] that appears `min_repeats` or more times back-to-back.

        Word-level with light normalization (lowercase, strip trailing
        punctuation) — token-exact loops are the dominant Voxtral/Whisper
        failure mode, so character-level fuzz isn't needed.

        Returns (is_loop, sample_phrase). sample_phrase is one instance of the
        offending n-gram, useful for logging.
        """
        words = text.split()
        if len(words) < min_ngram * min_repeats:
            return False, None

        _PUNCT = ".,!?;:\"'()…—–-"

        def norm(w: str) -> str:
            return w.lower().strip(_PUNCT)

        nwords = [norm(w) for w in words]

        for n in range(min_ngram, max_ngram + 1):
            if len(nwords) < n * min_repeats:
                break
            i = 0
            while i + 2 * n <= len(nwords):
                ngram = nwords[i:i + n]
                if not any(ngram):  # all-punctuation/empty — skip
                    i += 1
                    continue
                if nwords[i + n:i + 2 * n] != ngram:
                    i += 1
                    continue
                # Found a doubled n-gram at i; count how many times it repeats.
                repeats = 2
                j = i + 2 * n
                while j + n <= len(nwords) and nwords[j:j + n] == ngram:
                    repeats += 1
                    j += n
                if repeats >= min_repeats:
                    return True, " ".join(words[i:i + n])
                i = j  # skip past this run
        return False, None

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

        # Snap boundaries to silences when enabled; this dramatically reduces
        # the chance a chunk starts mid-utterance (the dominant Voxtral loop
        # trigger). If silence detection finds nothing useful in a window,
        # _pick_boundaries falls back to the fixed-time target.
        if settings.voxtral_vad_align_chunks:
            silences = await detect_silences(
                wav_path,
                noise_db=settings.voxtral_silence_threshold_db,
                min_duration_sec=settings.voxtral_silence_min_duration_sec,
            )
            boundaries = self._pick_boundaries(
                target_chunk_sec=float(chunk_sec),
                total_duration=total_duration,
                silences=silences,
                search_window_sec=settings.voxtral_silence_search_window_sec,
            )
            log.info(
                "voxtral: %d silences, %d boundaries (target=%ds)",
                len(silences), len(boundaries), chunk_sec,
            )
        else:
            boundaries = [
                float(i * chunk_sec)
                for i in range(1, int(total_duration // chunk_sec) + 1)
                if i * chunk_sec < total_duration
            ]

        chunks = await self._split_audio(wav_path, boundaries, overlap_sec, total_duration)
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
                return await self._transcribe_chunk_with_retry(
                    idx=idx,
                    chunk=chunk,
                    language=language,
                    timeout=timeout,
                    progress_cb=make_cb(idx),
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

    # ----------------------------------------------------------- retry+fallback
    async def _transcribe_chunk_with_retry(
        self,
        idx: int,
        chunk: _Chunk,
        language: str,
        timeout: int,
        progress_cb,
    ) -> EngineResult:
        """Transcribe one chunk with Voxtral, retrying on detected runaway
        repetition loops. After `voxtral_max_chunk_retries` Voxtral attempts
        all fail (whether by exception or by loop output), fall back to
        Whisper for this chunk only — the cheap, slower, but loop-free path.
        """
        retries = max(1, settings.voxtral_max_chunk_retries)
        last_err: str | None = None

        for attempt in range(1, retries + 1):
            try:
                result = await self._transcribe_one(
                    chunk.path,
                    language=language,
                    timeout=timeout,
                    progress=progress_cb,
                    duration_hint=chunk.duration_sec,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                log.warning(
                    "voxtral: chunk %d attempt %d/%d raised: %s",
                    idx, attempt, retries, last_err,
                )
                continue

            is_loop, phrase = self._detect_repetition_loop(
                result.text,
                min_ngram=settings.voxtral_loop_min_ngram,
                max_ngram=settings.voxtral_loop_max_ngram,
                min_repeats=settings.voxtral_loop_min_repeats,
            )
            if not is_loop:
                return result
            last_err = f"repetition loop: {phrase!r}"
            log.warning(
                "voxtral: chunk %d attempt %d/%d returned %s",
                idx, attempt, retries, last_err,
            )

        # All Voxtral attempts failed — fall back to Whisper for this chunk.
        log.warning(
            "voxtral: chunk %d falling back to whisper after %d failed attempts (last=%s)",
            idx, retries, last_err,
        )
        # Reset progress so the fallback's progress bar starts clean.
        progress_cb(0.0)
        from .whisper import WhisperEngine  # local import: avoid import cycles
        return await WhisperEngine().transcribe(
            chunk.path,
            language=language,
            timeout=timeout,
            progress=progress_cb,
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
