"""§T NFR-P2: the real bind is loopback only, enforced by the launcher and per request."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hub.settings import MSG_HOST
from tests.integration.conftest import Env, bearer

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _asgi_get(app: Any, path: str, server: tuple[str, int] | None, token: str | None) -> int:
    """Drive one HTTP request through the ASGI app with an explicit ``scope["server"]``."""
    headers = [(b"host", b"hub")]
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": headers, "client": ("127.0.0.1", 50000), "server": server,
    }  # fmt: skip
    status: list[int] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] == "http.response.start":
            status.append(int(msg["status"]))

    asyncio.run(app(scope, receive, send))
    return status[0]


@pytest.mark.parametrize("server", [("0.0.0.0", 8787), ("192.168.1.20", 8787), ("::", 8787), None])
def test_non_loopback_server_socket_gets_503(env: Env, server: tuple[str, int] | None) -> None:
    app = env.client.app
    tok = bearer("mom")["Authorization"].split()[1]
    for path in ("/healthz", "/mission/today", "/diary"):
        assert _asgi_get(app, path, server, tok) == 503


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_server_socket_is_served(env: Env, host: str) -> None:
    app = env.client.app
    tok = bearer("mom")["Authorization"].split()[1]
    assert _asgi_get(app, "/healthz", (host, 8787), None) == 200
    assert _asgi_get(app, "/diary", (host, 8787), tok) == 200
    assert _asgi_get(app, "/diary", (host, 8787), None) == 401


def test_503_body_is_a_code(env: Env) -> None:
    from hub.api.errors import LoopbackGuard

    assert LoopbackGuard.BODY == b'{"error":"non_loopback_bind"}'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _env(tmp: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp),
        "SAIR_ENV": "test",
        "SAIR_DATA_DIR": str(tmp / "data"),
        "SAIR_LOG_DIR": str(tmp / "logs"),
        **extra,
    }


def test_python_m_hub_refuses_non_loopback(tmp_path: Path) -> None:
    port = _free_port()
    r = subprocess.run(
        [sys.executable, "-m", "hub"],
        env=_env(tmp_path, SAIR_HOST="0.0.0.0", SAIR_PORT=str(port)),
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode != 0
    assert MSG_HOST in r.stderr
    assert "0.0.0.0" not in r.stderr  # the rejected value is not echoed
    assert "Uvicorn running" not in r.stderr + r.stdout
    assert not (tmp_path / "data" / "sair.db").exists()  # nothing started


def test_uvicorn_cli_with_wildcard_host_fails_startup(tmp_path: Path) -> None:
    port = _free_port()
    r = subprocess.run(
        [sys.executable, "-m", "uvicorn", "hub.app:app", "--host", "0.0.0.0", "--port", str(port)],
        env=_env(tmp_path), cwd=REPO, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode != 0
    assert "Uvicorn running" not in r.stderr + r.stdout
    assert "non_loopback_bind" in r.stderr + r.stdout


def test_launchers_use_python_m_hub() -> None:
    up = (REPO / "scripts" / "up.sh").read_text()
    plist = (REPO / "ops" / "launchd" / "com.sair.hub.plist").read_text()
    assert "python -m hub" in up and "uvicorn" not in up
    assert "<string>-m</string>" in plist and "<string>hub</string>" in plist
    assert "uvicorn" not in plist and "0.0.0.0" not in plist
