from __future__ import annotations
import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from . import jobs as job_mod
from .db import init_db
from .routes import auth, health, transcribe, transcripts
from .routes import jobs as jobs_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("voicenote")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    cleanup_task = asyncio.create_task(_periodic_job_cleanup())
    log.info(
        "voicenote ready · db=%s · models=%s · default_engine=%s",
        settings.database_url, settings.models_dir, settings.default_engine,
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task


async def _periodic_job_cleanup() -> None:
    while True:
        await asyncio.sleep(3600)
        await job_mod.cleanup_old_jobs()


def create_app() -> FastAPI:
    app = FastAPI(
        title="voicenote",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(transcribe.router)
    app.include_router(transcripts.router)
    app.include_router(jobs_routes.router)

    @app.exception_handler(404)
    async def _404(_request, exc):
        # API requests get JSON; web requests fall through to the SPA shell.
        # Match BOTH the bare path ("/v1") and any sub-path ("/v1/...") so we
        # don't accidentally serve HTML to API clients on a missing endpoint.
        from starlette.requests import Request
        if isinstance(_request, Request):
            path = _request.url.path
            for prefix in ("/v1", "/api"):
                if path == prefix or path.startswith(prefix + "/"):
                    return JSONResponse({"detail": "Not found"}, status_code=404)
        if settings.serve_web and (settings.web_dir / "index.html").exists():
            return FileResponse(settings.web_dir / "index.html")
        return JSONResponse({"detail": "Not found"}, status_code=404)

    if settings.serve_web and settings.web_dir.exists():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run(
        "voicenote.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
