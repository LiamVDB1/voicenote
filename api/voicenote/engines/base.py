from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


# Engines call this with a fraction in [0.0, 1.0] reflecting how much of the
# audio has been processed. Optional — engines that can't report fine-grained
# progress simply don't call it.
ProgressCallback = Callable[[float], None]


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class EngineResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    detected_language: str | None = None
    engine: str = ""
    raw_stdout: str = ""
    raw_stderr: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in self.segments
            ],
            "detected_language": self.detected_language,
            "engine": self.engine,
        }


class TranscriptionEngine(Protocol):
    name: str

    async def is_ready(self) -> bool: ...

    async def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        timeout: int = 3600,
        progress: ProgressCallback | None = None,
    ) -> EngineResult: ...
