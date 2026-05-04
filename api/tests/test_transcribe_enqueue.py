from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicenote.auth import current_user
from voicenote.db import User, get_session
from voicenote.routes import transcribe


def fake_user():
    return User(id=3, username="tester", password_hash="x", display_name="Tester")


class TranscribeEnqueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_data_dir = transcribe.settings.data_dir
        self.old_default = transcribe.settings.default_engine
        self.old_max_mb = transcribe.settings.max_upload_mb
        self.old_max_minutes = transcribe.settings.max_audio_minutes
        self.old_probe = transcribe.probe_duration_sec
        self.old_enqueue = transcribe.jobs.enqueue
        self.old_names = transcribe.ENGINE_NAMES
        transcribe.settings.data_dir = Path(self.tmp.name)
        transcribe.settings.default_engine = "parakeet"
        transcribe.settings.max_upload_mb = 1
        transcribe.settings.max_audio_minutes = 180
        transcribe.ENGINE_NAMES = ("parakeet", "whisper", "voxtral")
        transcribe.probe_duration_sec = lambda path: self._probe(path)
        transcribe.jobs.enqueue = self._enqueue
        self.enqueued = []
        self.app = FastAPI()
        self.app.include_router(transcribe.router)
        self.app.dependency_overrides[current_user] = fake_user
        self.app.dependency_overrides[get_session] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        transcribe.settings.data_dir = self.old_data_dir
        transcribe.settings.default_engine = self.old_default
        transcribe.settings.max_upload_mb = self.old_max_mb
        transcribe.settings.max_audio_minutes = self.old_max_minutes
        transcribe.probe_duration_sec = self.old_probe
        transcribe.jobs.enqueue = self.old_enqueue
        transcribe.ENGINE_NAMES = self.old_names
        self.app.dependency_overrides.clear()
        self.tmp.cleanup()

    async def _probe(self, path):
        return 12.5

    async def _enqueue(self, **kwargs):
        self.enqueued.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"id": kwargs["job_id"], "status": "queued", "filename": kwargs["filename"]})

    def test_upload_is_streamed_to_job_storage_and_returns_accepted_job(self):
        res = self.client.post(
            "/v1/transcribe",
            data={"engine": "whisper", "language": "nl"},
            files={"audio": ("memo.wav", b"fake-audio", "audio/wav")},
        )

        self.assertEqual(res.status_code, 202)
        self.assertEqual(res.json()["status"], "queued")
        self.assertEqual(len(self.enqueued), 1)
        call = self.enqueued[0]
        self.assertEqual(call["user_id"], 3)
        self.assertEqual(call["filename"], "memo.wav")
        self.assertEqual(call["engine"], "whisper")
        self.assertEqual(call["language"], "nl")
        self.assertEqual(call["audio_size"], len(b"fake-audio"))
        self.assertTrue(Path(call["audio_path"]).exists())
        self.assertIn(call["job_id"], str(call["audio_path"]))


if __name__ == "__main__":
    unittest.main()
