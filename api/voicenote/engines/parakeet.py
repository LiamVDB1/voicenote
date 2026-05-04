from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any

from ..config import settings
from .base import EngineResult, Segment


class ParakeetEngine:
    """
    NVIDIA Parakeet TDT 0.6B V3, INT8, via sherpa-onnx.
    - 25 European languages including Dutch, English, French, German.
    - ~640 MB total weights.
    - Uses Silero VAD to chunk long audio when available; falls back to one-shot
      decode for short clips.
    """
    name = "parakeet"

    _recognizer: Any = None  # sherpa_onnx.OfflineRecognizer
    _vad_config: Any = None  # sherpa_onnx.VadModelConfig

    async def is_ready(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return all([
            settings.parakeet_encoder.exists(),
            settings.parakeet_decoder.exists(),
            settings.parakeet_joiner.exists(),
            settings.parakeet_tokens.exists(),
        ])

    def _load(self):
        import sherpa_onnx
        if self._recognizer is None:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(settings.parakeet_encoder),
                decoder=str(settings.parakeet_decoder),
                joiner=str(settings.parakeet_joiner),
                tokens=str(settings.parakeet_tokens),
                num_threads=settings.inference_threads,
                model_type="nemo_transducer",
                provider="cpu",
                debug=False,
            )
        if self._vad_config is None and settings.use_vad and settings.silero_vad_path.exists():
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(settings.silero_vad_path)
            cfg.silero_vad.threshold = settings.vad_threshold
            cfg.silero_vad.min_silence_duration = settings.vad_min_silence_sec
            cfg.silero_vad.min_speech_duration = settings.vad_min_speech_sec
            cfg.sample_rate = 16000
            self._vad_config = cfg

    async def transcribe(
        self, wav_path: Path, language: str = "auto", timeout: int = 3600
    ) -> EngineResult:
        # sherpa-onnx is synchronous CPU work — push to a thread, but enforce
        # the timeout so a pathological decode can't tie up workers indefinitely.
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._transcribe_sync, wav_path),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"parakeet inference timed out after {timeout}s")

    def _transcribe_sync(self, wav_path: Path) -> EngineResult:
        import sherpa_onnx
        self._load()
        samples, sr = sherpa_onnx.read_wave(str(wav_path))
        # Use VAD-based chunking for anything longer than ~30s.
        duration = len(samples) / float(sr) if sr else 0.0
        if self._vad_config is not None and duration > 30.0:
            return self._transcribe_with_vad(samples, sr)
        return self._transcribe_oneshot(samples, sr)

    def _transcribe_oneshot(self, samples, sr) -> EngineResult:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        self._recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        return EngineResult(
            text=text,
            segments=[Segment(start=0.0, end=len(samples) / sr, text=text)] if text else [],
            engine=self.name,
        )

    def _transcribe_with_vad(self, samples, sr) -> EngineResult:
        import sherpa_onnx
        vad = sherpa_onnx.VoiceActivityDetector(
            self._vad_config, buffer_size_in_seconds=100
        )
        window_size = self._vad_config.silero_vad.window_size  # 512 typically

        segments: list[Segment] = []
        pending_streams: list[Any] = []
        pending_starts: list[float] = []
        pending_ends: list[float] = []

        def drain(flush: bool = False):
            if flush:
                vad.flush()
            while not vad.empty():
                seg = vad.front
                start_sec = seg.start / float(sr)
                end_sec = (seg.start + len(seg.samples)) / float(sr)
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sr, seg.samples)
                pending_streams.append(stream)
                pending_starts.append(start_sec)
                pending_ends.append(end_sec)
                vad.pop()
                # Decode in small batches to keep memory bounded
                if len(pending_streams) >= 8:
                    self._recognizer.decode_streams(pending_streams)
                    for s, st, et in zip(pending_streams, pending_starts, pending_ends):
                        t = (s.result.text or "").strip()
                        if t:
                            segments.append(Segment(start=st, end=et, text=t))
                    pending_streams.clear()
                    pending_starts.clear()
                    pending_ends.clear()

        i = 0
        n = len(samples)
        while i + window_size < n:
            vad.accept_waveform(samples[i : i + window_size])
            i += window_size
            drain(flush=False)

        drain(flush=True)
        if pending_streams:
            self._recognizer.decode_streams(pending_streams)
            for s, st, et in zip(pending_streams, pending_starts, pending_ends):
                t = (s.result.text or "").strip()
                if t:
                    segments.append(Segment(start=st, end=et, text=t))

        full = " ".join(seg.text for seg in segments).strip()
        return EngineResult(text=full, segments=segments, engine=self.name)
