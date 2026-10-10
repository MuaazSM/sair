"""Error bodies, request-log hygiene and response headers (NFR-P4).

* Every error body is ``{"error": "<fixed code>"}``; validation errors add only field
  locations (never input values). No exception text ever reaches a response or a log.
* The ASGI middleware logs method, route template, status, latency and walk id; never a body
  or a query string. It also swallows unhandled exceptions after logging the type only, so
  the server's own traceback logger never sees an exception message.
* Security headers on every response (CSP without external origins, no referrer, no
  geolocation).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import structlog
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hub.settings import LOOPBACK_HOSTS

__all__ = [
    "SECURITY_HEADERS",
    "ApiError",
    "HubMiddleware",
    "LoopbackGuard",
    "configure_logging",
    "install_handlers",
]

log = structlog.get_logger("hub.api")

SECURITY_HEADERS: dict[str, str] = {
    "content-security-policy": (
        "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "referrer-policy": "no-referrer",
    "permissions-policy": "geolocation=()",
    "x-content-type-options": "nosniff",
}

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
}
_ID = re.compile(r"^[0-9a-f]{32}$")
_FIELD = re.compile(r"^[a-z_]{1,32}$")


class ApiError(Exception):
    """A deliberate API failure with a fixed code (the message is never user content)."""

    def __init__(self, status: int, code: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.headers = headers or {}


def error_response(
    status: int, code: str, headers: dict[str, str] | None = None, **extra: Any
) -> JSONResponse:
    return JSONResponse({"error": code, **extra}, status_code=status, headers=headers)


async def _api_error(_: Request, exc: Exception) -> Response:
    assert isinstance(exc, ApiError)
    return error_response(exc.status, exc.code, exc.headers)


async def _http_error(_: Request, exc: Exception) -> Response:
    assert isinstance(exc, StarletteHTTPException)
    return error_response(
        exc.status_code, _STATUS_CODES.get(exc.status_code, "error"), dict(exc.headers or {})
    )


async def _validation_error(_: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    fields: list[list[str | int]] = []
    for err in exc.errors():
        loc: list[str | int] = []
        for part in err.get("loc", ()):
            if isinstance(part, int) or (isinstance(part, str) and _FIELD.match(part)):
                loc.append(part)
        if loc not in fields:
            fields.append(loc)
    return error_response(422, "invalid_request", fields=fields)


async def _unhandled(_: Request, exc: Exception) -> Response:
    log.error("unhandled_exception", exc_type=type(exc).__name__)
    return error_response(500, "internal_error")


def install_handlers(app: Any) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)


def configure_logging() -> None:
    """Structured JSON to stdout; the server's access log (which prints query strings) is off."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("msg"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=False,
    )
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.disabled = True


class LoopbackGuard:
    """NFR-P2, per request: serve only on a loopback server socket.

    ``scope["server"]`` is the local address of the accepted socket. Anything that is not
    loopback (a wildcard or LAN bind, or an unknown server) gets a fixed 503 code and the app
    is never reached. One code-only error line is logged per process.
    """

    BODY = b'{"error":"non_loopback_bind"}'

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._logged = False

    @staticmethod
    def is_loopback(server: object) -> bool:
        if not isinstance(server, list | tuple) or not server:
            return False
        host = server[0]
        return isinstance(host, str) and host in LOOPBACK_HOSTS

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self.is_loopback(scope.get("server")):
            await self.app(scope, receive, send)
            return
        if not self._logged:
            self._logged = True
            log.error("non_loopback_bind")
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(self.BODY)).encode()),
                    *((k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": self.BODY})


class HubMiddleware:
    """Pure ASGI: security headers, request log line, last-resort 500 without exception text."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        status = 500
        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                headers = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    headers[key] = value
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            log.error("unhandled_exception", exc_type=type(exc).__name__)
            if not started:
                await error_response(500, "internal_error")(scope, receive, send_wrapper)
            status = 500
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None)
            wid = (scope.get("path_params") or {}).get("walk_id")
            if not (isinstance(wid, str) and _ID.match(wid)):
                wid = (scope.get("state") or {}).get("walk_id")
            log.info(
                "request",
                method=scope.get("method"),
                route=template if isinstance(template, str) and template else "static",
                status=status,
                ms=round((time.perf_counter() - t0) * 1000.0, 1),
                walk_id=wid if isinstance(wid, str) and _ID.match(wid) else None,
            )
