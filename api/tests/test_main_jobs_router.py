from __future__ import annotations

import unittest

from voicenote.main import create_app


class MainRouterTests(unittest.TestCase):
    def test_jobs_router_is_registered(self):
        app = create_app()
        paths = {route.path for route in app.routes if hasattr(route, "path")}

        self.assertIn("/v1/jobs", paths)
        self.assertIn("/v1/jobs/{job_id}", paths)


if __name__ == "__main__":
    unittest.main()
