from __future__ import annotations
import asyncio
import contextlib
import json
import re
import tempfile
from pathlib import Path

from ..config import settings
from .base import EngineResult, Segment


class WhisperEngine:
    name = "whisper"

    async def is_ready(self) -> bool:
        return settings.whisper_model_path.exists() and bool(self._which(settings.whisper_bin))

    @staticmethod
    def _which(bin_name: str) -> str | None:
        import shutil
        return shutil.which(bin_name)

    async def transcribe(
        self,
        wav_path: Path,
        language: str = "auto",
        timeout: int = 3600,
        progress=None,
    ) -> EngineResult:
        if not settings.whisper_model_path.exists():
            raise RuntimeError(f"Whisper model not found: {settings.whisper_model_path}")
        bin_path = self._which(settings.whisper_bin)
        if not bin_path:
            raise RuntimeError(f"{settings.whisper_bin} not in PATH")

        with tempfile.TemporaryDirectory() as td:
            out_prefix = Path(td) / "out"
            cmd = [
                bin_path,
                "-m", str(settings.whisper_model_path),
                "-f", str(wav_path),
                "-t", str(settings.inference_threads),
                "-p", "1",
                "-l", language if language and language != "auto" else "auto",
                "-oj",  # JSON output
                "-of", str(out_prefix),
                "-pp",  # print progress to stderr — drives the progress bar
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_buf: list[bytes] = []
            stderr_buf: list[bytes] = []
            progress_re = re.compile(rb"progress\s*=\s*(\d+)")

            async def _drain(stream, buf, with_progress: bool) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    buf.append(line)
                    if with_progress and progress is not None:
                        m = progress_re.search(line)
                        if m:
                            try:
                                pct = int(m.group(1)) / 100.0
                                progress(max(0.0, min(0.99, pct)))
                            except Exception:
                                pass

            stdout_task = asyncio.create_task(_drain(proc.stdout, stdout_buf, False))
            stderr_task = asyncio.create_task(_drain(proc.stderr, stderr_buf, True))

            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task), timeout=timeout
                )
                await proc.wait()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                for t in (stdout_task, stderr_task):
                    t.cancel()
                    with contextlib.suppress(BaseException):
                        await t
                raise RuntimeError("whisper-cli timed out")

            out = b"".join(stdout_buf)
            err = b"".join(stderr_buf)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"whisper-cli exited {proc.returncode}: "
                    f"{err.decode('utf-8', 'replace')[:600]}"
                )

            if progress is not None:
                progress(1.0)

            json_path = Path(str(out_prefix) + ".json")
            if not json_path.exists():
                # fallback: parse from stdout
                text = out.decode("utf-8", "replace")
                return EngineResult(
                    text=text.strip(),
                    segments=[],
                    engine=self.name,
                    raw_stdout=text,
                    raw_stderr=err.decode("utf-8", "replace"),
                )
            data = json.loads(json_path.read_text())

        # Tolerate whisper.cpp schema drift: items may live under "transcription"
        # or "segments"; timestamps may be under "offsets.{from,to}" (ms),
        # "t0/t1" (10ms units), or "start/end" (seconds).
        items = data.get("transcription") or data.get("segments") or []
        segments: list[Segment] = []
        full_text_parts: list[str] = []
        for item in items:
            t = (item.get("text") or "").strip()
            if not t:
                continue
            start = end = 0.0
            if "offsets" in item:
                off = item["offsets"] or {}
                start = float(off.get("from", 0)) / 1000.0
                end = float(off.get("to", 0)) / 1000.0
            elif "t0" in item and "t1" in item:
                start = float(item["t0"]) / 100.0
                end = float(item["t1"]) / 100.0
            elif "start" in item and "end" in item:
                start = float(item["start"])
                end = float(item["end"])
            segments.append(Segment(start=start, end=end, text=t))
            full_text_parts.append(t)

        full = (
            " ".join(full_text_parts).strip()
            or (data.get("text") or "").strip()
        )
        detected = (data.get("result") or {}).get("language") or data.get("language")
        return EngineResult(
            text=full,
            segments=segments,
            detected_language=detected,
            engine=self.name,
            raw_stderr=err.decode("utf-8", "replace"),
        )
