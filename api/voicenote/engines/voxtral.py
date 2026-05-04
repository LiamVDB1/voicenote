from __future__ import annotations
import asyncio
import re
from pathlib import Path

from ..config import settings
from .base import EngineResult


_LANG_PROMPTS = {
    "nl": "Transcribeer deze audio woord voor woord in het Nederlands. Geef alleen de tekst terug.",
    "en": "Transcribe this audio verbatim in English. Output only the transcript.",
    "fr": "Transcrivez cet audio mot pour mot en français. Renvoyez uniquement le transcript.",
    "de": "Transkribiere diesen Audio wortgetreu auf Deutsch. Gib nur das Transkript zurück.",
    "auto": "Transcribe this audio verbatim in its source language. Output only the transcript text, no commentary.",
}


class VoxtralEngine:
    """
    Mistral Voxtral Mini 3B via llama.cpp's mtmd-cli.
    Optional second-line fallback. llama.cpp's audio support is upstream-flagged
    experimental — keep the model files out of the default download to save space.
    """
    name = "voxtral"

    async def is_ready(self) -> bool:
        return (
            settings.voxtral_model_path.exists()
            and settings.voxtral_mmproj_path.exists()
            and bool(self._which(settings.llama_mtmd_bin))
        )

    @staticmethod
    def _which(bin_name: str) -> str | None:
        import shutil
        return shutil.which(bin_name)

    async def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        timeout: int = 3600,
        progress=None,  # llama-mtmd-cli has no fine-grained progress; ignored.
    ) -> EngineResult:
        if not settings.voxtral_model_path.exists():
            raise RuntimeError(f"Voxtral model not found: {settings.voxtral_model_path}")
        if not settings.voxtral_mmproj_path.exists():
            raise RuntimeError(f"Voxtral mmproj not found: {settings.voxtral_mmproj_path}")
        bin_path = self._which(settings.llama_mtmd_bin)
        if not bin_path:
            raise RuntimeError(f"{settings.llama_mtmd_bin} not in PATH")

        prompt = _LANG_PROMPTS.get(language, _LANG_PROMPTS["auto"])

        cmd = [
            bin_path,
            "-m", str(settings.voxtral_model_path),
            "--mmproj", str(settings.voxtral_mmproj_path),
            "--audio", str(wav_path),
            "-p", prompt,
            "-t", str(settings.inference_threads),
            "-c", "16384",
            "--temp", "0.0",
            "-no-cnv",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("voxtral inference timed out")

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-mtmd-cli exited {proc.returncode}: "
                f"{err.decode('utf-8', 'replace')[-800:]}"
            )

        raw = out.decode("utf-8", "replace")
        text = self._extract_assistant_text(raw)
        return EngineResult(
            text=text,
            segments=[],
            detected_language=language if language != "auto" else None,
            engine=self.name,
            raw_stdout=raw,
            raw_stderr=err.decode("utf-8", "replace"),
        )

    @staticmethod
    def _extract_assistant_text(stdout: str) -> str:
        lines = stdout.splitlines()
        kept: list[str] = []
        skip_prefixes = (
            "main:", "load_", "llama_", "ggml_", "system_info", "sampler",
            "build:", "encoding image slice", "image slice encoded",
            "log start", "log end", "model:", "n_threads", "n_predict",
        )
        for ln in lines:
            s = ln.rstrip()
            if not s:
                continue
            low = s.strip().lower()
            if any(low.startswith(p) for p in skip_prefixes):
                continue
            if s.startswith(">") or s.startswith("<"):
                continue
            kept.append(s)
        text = "\n".join(kept).strip()
        text = re.sub(r"\n[^\n]*tokens?/s.*$", "", text, flags=re.S).strip()
        text = re.sub(r"^Transcribe[^\n]*\n", "", text, flags=re.I)
        return text.strip()
