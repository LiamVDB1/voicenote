from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from voicenote import jobs
from voicenote.engines.base import EngineResult


class FakeEngine:
    def __init__(self, ready=True, fail=False):
        self.ready = ready
        self.fail = fail

    async def is_ready(self):
        return self.ready

    async def transcribe(self, wav_path, language="auto", timeout=3600):
        if self.fail:
            raise RuntimeError("boom")
        return EngineResult(text="ok", detected_language=language if language != "auto" else None)


class JobCascadeTests(unittest.TestCase):
    def test_skipped_engines_do_not_mark_fallback(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                job = jobs.Job(
                    user_id=1, filename="a.wav", engine="parakeet", language="nl",
                    audio_path=Path(td) / "a.wav", audio_size=10,
                )
                old_get_engine = jobs.get_engine
                old_fallback = jobs.settings.fallback_chain
                old_names = jobs.ENGINE_NAMES
                try:
                    jobs.ENGINE_NAMES = ("parakeet", "whisper", "voxtral")
                    jobs.settings.fallback_chain = "whisper"
                    engines = {
                        "parakeet": FakeEngine(ready=False),
                        "whisper": FakeEngine(ready=True),
                    }
                    jobs.get_engine = lambda name: engines[name]
                    result, used, fallback = await jobs._run_cascade(job, Path(td) / "a.wav")
                    self.assertEqual(result.text, "ok")
                    self.assertEqual(used, "whisper")
                    self.assertFalse(fallback)
                finally:
                    jobs.get_engine = old_get_engine
                    jobs.settings.fallback_chain = old_fallback
                    jobs.ENGINE_NAMES = old_names
        asyncio.run(scenario())

    def test_failed_ready_engine_marks_fallback(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                job = jobs.Job(
                    user_id=1, filename="a.wav", engine="parakeet", language="nl",
                    audio_path=Path(td) / "a.wav", audio_size=10,
                )
                old_get_engine = jobs.get_engine
                old_fallback = jobs.settings.fallback_chain
                old_names = jobs.ENGINE_NAMES
                try:
                    jobs.ENGINE_NAMES = ("parakeet", "whisper", "voxtral")
                    jobs.settings.fallback_chain = "whisper"
                    engines = {
                        "parakeet": FakeEngine(ready=True, fail=True),
                        "whisper": FakeEngine(ready=True),
                    }
                    jobs.get_engine = lambda name: engines[name]
                    result, used, fallback = await jobs._run_cascade(job, Path(td) / "a.wav")
                    self.assertEqual(result.text, "ok")
                    self.assertEqual(used, "whisper")
                    self.assertTrue(fallback)
                finally:
                    jobs.get_engine = old_get_engine
                    jobs.settings.fallback_chain = old_fallback
                    jobs.ENGINE_NAMES = old_names
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
