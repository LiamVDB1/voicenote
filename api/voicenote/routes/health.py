from __future__ import annotations
from fastapi import APIRouter

from ..config import settings
from ..engines import ENGINE_NAMES, get_engine

router = APIRouter()


_ENGINE_MODEL_LABEL = {
    "parakeet": lambda: settings.parakeet_dir,
    "whisper":  lambda: settings.whisper_model,
    "voxtral":  lambda: f"{settings.voxtral_model_name} (Mistral API)",
}


@router.get("/v1/health")
async def health() -> dict:
    engines = {}
    for name in ENGINE_NAMES:
        engine = get_engine(name)
        engines[name] = {
            "ready": await engine.is_ready(),
            "model": _ENGINE_MODEL_LABEL[name](),
        }
    return {
        "ok": True,
        "default_engine": settings.default_engine,
        "fallback_chain": settings.fallback_engines,
        "engines": engines,
    }
