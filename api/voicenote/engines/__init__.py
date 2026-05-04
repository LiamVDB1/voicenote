from __future__ import annotations
import threading

from .base import EngineResult, Segment, TranscriptionEngine
from .parakeet import ParakeetEngine
from .voxtral import VoxtralEngine
from .whisper import WhisperEngine

__all__ = [
    "EngineResult",
    "Segment",
    "TranscriptionEngine",
    "ParakeetEngine",
    "VoxtralEngine",
    "WhisperEngine",
    "get_engine",
    "ENGINE_NAMES",
]

ENGINE_NAMES = ("parakeet", "whisper", "voxtral")

# Engine instances are reused across requests so the Parakeet ONNX recognizer
# (and any other future per-engine state) is loaded once per process.
_lock = threading.Lock()
_instances: dict[str, TranscriptionEngine] = {}


def get_engine(name: str) -> TranscriptionEngine:
    if name not in ENGINE_NAMES:
        raise ValueError(f"Unknown engine: {name}")
    with _lock:
        inst = _instances.get(name)
        if inst is None:
            if name == "parakeet":
                inst = ParakeetEngine()
            elif name == "whisper":
                inst = WhisperEngine()
            else:  # voxtral
                inst = VoxtralEngine()
            _instances[name] = inst
        return inst
