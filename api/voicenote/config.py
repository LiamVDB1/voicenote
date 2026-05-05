from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VN_", env_file=".env", extra="ignore")

    # Network
    host: str = "0.0.0.0"
    port: int = 8732

    # Storage
    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    db_url: str | None = None

    # Web
    web_dir: Path = Path("/app/web")
    serve_web: bool = True

    # Auth
    secret_key: str = "change-me-in-production"
    session_cookie_name: str = "vn_session"
    session_max_age_days: int = 30
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"

    # Engines — Parakeet is default. Whisper & Voxtral are optional fallbacks.
    default_engine: Literal["parakeet", "whisper", "voxtral"] = "parakeet"
    # Fallback chain (comma-separated): tried in order if the chosen engine fails.
    fallback_chain: str = "whisper,voxtral"

    # Parakeet (sherpa-onnx INT8)
    parakeet_dir: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    silero_vad_filename: str = "silero_vad.onnx"
    use_vad: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_sec: float = 0.25
    vad_min_speech_sec: float = 0.2

    # Whisper (whisper.cpp). Defaults to large-v3-turbo Q5: ~5× faster than
    # full large-v3 on CPU, near-identical multilingual quality. Override via
    # VN_WHISPER_MODEL if you want to swap in a different ggml file.
    whisper_model: str = "ggml-large-v3-turbo-q5_0.bin"
    whisper_bin: str = "whisper-cli"

    # Voxtral (llama.cpp / mtmd-cli)
    voxtral_model: str = "Voxtral-Mini-3B-2507-Q4_K_M.gguf"
    voxtral_mmproj: str = "mmproj-Voxtral-Mini-3B-2507-f16.gguf"
    llama_mtmd_bin: str = "llama-mtmd-cli"

    # Inference
    inference_threads: int = 3
    # How many transcribe jobs may run in parallel. On a 4-core box, 1 is
    # safest — Parakeet/Whisper already saturate the cores via inference_threads.
    max_concurrent_jobs: int = 1

    # Limits
    max_upload_mb: int = 200
    max_audio_minutes: int = 180
    transcribe_timeout_sec: int = 60 * 60 * 4

    # Misc
    cors_origins: list[str] = Field(default_factory=list)

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite+aiosqlite:///{self.data_dir}/voicenote.db"

    @property
    def parakeet_path(self) -> Path:
        return self.models_dir / self.parakeet_dir

    @property
    def parakeet_encoder(self) -> Path:
        return self.parakeet_path / "encoder.int8.onnx"

    @property
    def parakeet_decoder(self) -> Path:
        return self.parakeet_path / "decoder.int8.onnx"

    @property
    def parakeet_joiner(self) -> Path:
        return self.parakeet_path / "joiner.int8.onnx"

    @property
    def parakeet_tokens(self) -> Path:
        return self.parakeet_path / "tokens.txt"

    @property
    def silero_vad_path(self) -> Path:
        return self.models_dir / self.silero_vad_filename

    @property
    def whisper_model_path(self) -> Path:
        return self.models_dir / self.whisper_model

    @property
    def voxtral_model_path(self) -> Path:
        return self.models_dir / self.voxtral_model

    @property
    def voxtral_mmproj_path(self) -> Path:
        return self.models_dir / self.voxtral_mmproj

    @property
    def fallback_engines(self) -> list[str]:
        return [s.strip() for s in self.fallback_chain.split(",") if s.strip()]


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
