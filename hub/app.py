"""FastAPI app for the hub (PRD §9.3). ``uvicorn hub.app:app --host 127.0.0.1``.

``create_app`` builds everything but starts nothing: model clients connect lazily, Kokoro
loads on first render. Importing this module only builds the app when ``app`` is first
accessed (uvicorn does so after the server is configured). Settings refuse a non-loopback
bind (NFR-P2) before anything else runs.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from hub.api.deps import AppDeps, build_deps
from hub.api.errors import HubMiddleware, configure_logging, install_handlers
from hub.api.routes import router
from hub.settings import Settings, get_settings
from hub.workers import Workers, startup

__all__ = ["create_app"]  # plus the lazy module attribute ``app``

log = structlog.get_logger("hub.app")


def _init_tracing(settings: Settings) -> None:
    """Dev/eval tracing (NFR-P7), only if the module exists and enables itself."""
    if importlib.util.find_spec("hub.tracing") is None:
        return
    mod: Any = importlib.import_module("hub.tracing")
    init = getattr(mod, "init_tracing", None)
    if init is not None:
        init(settings)


def create_app(settings: Settings | None = None, deps: AppDeps | None = None) -> FastAPI:
    settings = settings or (deps.settings if deps else get_settings())
    deps = deps or build_deps(settings)
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(_init_tracing, deps.settings)
        await asyncio.to_thread(startup, deps)
        workers = Workers(deps)
        workers.start()
        try:
            yield
        finally:
            await workers.stop()
            await asyncio.to_thread(deps.close)

    app = FastAPI(
        title="Sair hub",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.deps = deps
    app.state.walk_tasks = set()
    app.add_middleware(HubMiddleware)
    install_handlers(app)
    app.include_router(router)
    if deps.static_dir is not None and deps.static_dir.is_dir():
        # After the API routes, so they win; unknown paths fall through to the PWA.
        app.mount("/", StaticFiles(directory=deps.static_dir, html=True), name="pwa")
    return app


_default: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Lazy module-level ``app`` for ``uvicorn hub.app:app``."""
    global _default
    if name == "app":
        if _default is None:
            _default = create_app()
        return _default
    raise AttributeError(name)
