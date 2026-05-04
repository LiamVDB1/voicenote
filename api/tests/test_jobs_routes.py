from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicenote.auth import current_user
from voicenote.db import User
from voicenote.routes import jobs as jobs_routes
from voicenote import jobs as job_mod


def fake_user(user_id=7):
    return User(id=user_id, username="tester", password_hash="x", display_name="Tester")


class JobRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(jobs_routes.router)
        self.app.dependency_overrides[current_user] = lambda: fake_user()
        self.client = TestClient(self.app)
        asyncio.run(self._clear_jobs())

    def tearDown(self):
        self.app.dependency_overrides.clear()
        asyncio.run(self._clear_jobs())

    async def _clear_jobs(self):
        async with job_mod._jobs_lock:
            job_mod._jobs.clear()

    async def _put_job(self, job):
        async with job_mod._jobs_lock:
            job_mod._jobs[job.id] = job

    def test_list_returns_only_current_users_recent_jobs(self):
        mine = job_mod.Job(
            user_id=7, filename="mine.wav", engine="parakeet", language="nl",
            audio_path=Path("/tmp/mine.wav"), audio_size=1, job_id="mine",
        )
        other = job_mod.Job(
            user_id=8, filename="other.wav", engine="parakeet", language="nl",
            audio_path=Path("/tmp/other.wav"), audio_size=1, job_id="other",
        )
        asyncio.run(self._put_job(mine))
        asyncio.run(self._put_job(other))

        res = self.client.get("/v1/jobs")

        self.assertEqual(res.status_code, 200)
        ids = [item["id"] for item in res.json()["items"]]
        self.assertEqual(ids, ["mine"])

    def test_get_rejects_jobs_owned_by_other_users(self):
        other = job_mod.Job(
            user_id=8, filename="other.wav", engine="parakeet", language="nl",
            audio_path=Path("/tmp/other.wav"), audio_size=1, job_id="other",
        )
        asyncio.run(self._put_job(other))

        res = self.client.get("/v1/jobs/other")

        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
