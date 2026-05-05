"""
Voxtral Mini via Mistral's hosted API. Roughly $1 per 1,000 minutes of audio
(voxtral-mini-transcribe pricing). Best Dutch ASR quality of the three engines
in this project — and zero local compute, so it doesn't fight the ARM box for
CPU. Falls through quickly if VN_MISTRAL_API_KEY isn't set.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

import httpx

from ..audio import probe_duration_sec
from ..config import settings
from .base import EngineResult, Segment


class VoxtralEngine:
    name = "voxtral"

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

        # Mistral has a per-request audio length cap — fail clearly upstream
        # rather than getting a vague 4xx from the API.
        duration = await probe_duration_sec(wav_path)
        if duration is not None and duration > settings.voxtral_max_minutes * 60:
            raise RuntimeError(
                f"Audio is langer dan {settings.voxtral_max_minutes} minuten — "
                "Voxtral-API ondersteunt geen langere bestanden in één request."
            )

        url = f"{settings.mistral_api_url.rstrip('/')}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}
        data: dict[str, str] = {"model": settings.voxtral_model_name}
        if language and language != "auto":
            data["language"] = language

        if progress:
            progress(0.05)

        # httpx handles streamed multipart upload; we open the file in a thread
        # to avoid blocking the event loop on large files.
        loop = asyncio.get_event_loop()

        def _read_bytes() -> bytes:
            return wav_path.read_bytes()

        body_bytes = await loop.run_in_executor(None, _read_bytes)
        files = {"file": (wav_path.name, body_bytes, "audio/wav")}

        if progress:
            progress(0.15)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(float(timeout), connect=30.0)) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Voxtral API onbereikbaar: {e}") from e

        if progress:
            progress(0.95)

        if resp.status_code >= 400:
            # Surface a useful error string to the UI without leaking the auth header.
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
