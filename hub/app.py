"""FastAPI app for the hub (PRD §9.3). Start it with ``python -m hub`` (host/port from settings).

``create_app`` builds everything but starts nothing: model clients connect lazily, Kokoro
loads on first render. Importing this module only builds the app when ``app`` is first
accessed (uvicorn does so after the server is configured). Settings refuse a non-loopback
bind (NFR-P2) before anything else runs; the launcher binds exactly ``SAIR_HOST``. If the app
is started some other way (``uvicorn hub.app:app --host 0.0.0.0``), the lifespan refuses to
start when the server config shows a non-loopback bind, and :class:`LoopbackGuard` answers
503 ``non_loopback_bind`` to any request arriving on a non-loopback socket.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from hub.api.deps import AppDeps, build_deps
from hub.api.errors import HubMiddleware, LoopbackGuard, configure_logging, install_handlers
from hub.api.routes import router
from hub.settings import LOOPBACK_HOSTS, Settings, get_settings
from hub.workers import Workers, startup

__all__ = [
    "NonLoopbackBind",
    "create_app",
    "server_binds",
]  # plus the lazy module attribute ``app``

log = structlog.get_logger("hub.app")


def _init_tracing(settings: Settings) -> None:
    """Dev/eval tracing (NFR-P7), only if the module exists and enables itself."""
    if importlib.util.find_spec("hub.tracing") is None:
        return
    mod: Any = importlib.import_module("hub.tracing")
    init = getattr(mod, "init_tracing", None)
    if init is not None:
        init(settings)


class NonLoopbackBind(RuntimeError):
    """NFR-P2: the server is configured to listen on a non-loopback address."""

    def __init__(self) -> None:
        super().__init__("non_loopback_bind")


def server_binds() -> list[str]:
    """Hosts of every live ``uvicorn.Config`` in this process (empty if uvicorn isn't used).

    A TCP bind with ``fd``/``uds`` set is not a host bind and is skipped (``fd`` is refused by
    the per-request guard if it turns out not to be loopback).
    """
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a hard dependency
        return []
    return [
        str(o.host)
        for o in gc.get_objects()
        if isinstance(o, uvicorn.Config) and o.uds is None and o.fd is None
    ]


def check_bind() -> None:
    if any(h not in LOOPBACK_HOSTS for h in server_binds()):
        log.error("non_loopback_bind")
        raise NonLoopbackBind


def create_app(settings: Settings | None = None, deps: AppDeps | None = None) -> FastAPI:
    settings = settings or (deps.settings if deps else get_settings())
    deps = deps or build_deps(settings)
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        check_bind()  # before anything starts (uvicorn binds only after the lifespan startup)
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
    app.add_middleware(LoopbackGuard)  # outermost: a non-loopback socket never reaches the app
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
