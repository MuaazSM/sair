"""Fixtures for HTTP API tests: tmp DB/dirs, stub models, TestClient with lifespan."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hub.api.deps import AppDeps, build_deps
from hub.app import create_app
from tests.unit._graphfix import (
    FIXTURES,
    FakeClock,
    RecNotifier,
    StubLlama,
    StubOllama,
    StubTts,
    make_settings,
)

TOKENS = {"mom": "tok-mom-0123456789", "dad": "tok-dad-0123456789", "owner": "tok-own-0123456789"}


def bearer(device: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[device]}"}


@dataclass
class Env:
    client: TestClient
    deps: AppDeps
    ollama: StubOllama
    tts: StubTts
    notifier: RecNotifier
    clock: FakeClock

    @property
    def tmp_dir(self) -> Path:
        return self.deps.settings.tmp_dir

    def post_walk(
        self,
        name: str = "en_20s.webm",
        device: str = "mom",
        content_type: str | None = None,
        data: dict[str, str] | None = None,
    ) -> Any:
        src = FIXTURES / name
        ctype = content_type or {
            ".webm": "audio/webm",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
        }.get(src.suffix, "application/octet-stream")
        return self.client.post(
            "/walk",
            headers=bearer(device),
            files={"audio": (name, src.read_bytes(), ctype)},
            data=data or {},
        )

    def wait_walk(self, walk_id: str, device: str = "mom", timeout: float = 30.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            r = self.client.get(f"/walk/{walk_id}", headers=bearer(device))
            assert r.status_code == 200
            body: dict[str, Any] = r.json()
            if body["status"] != "received" or time.monotonic() > end:
                return body
            time.sleep(0.05)


@pytest.fixture
def make_env(tmp_path: Path) -> Iterator[Callable[..., Env]]:
    """Factory: ``make_env(ollama=..., llama=..., **settings_overrides)``; closed at teardown."""
    clients: list[TestClient] = []

    def factory(
        *,
        ollama: StubOllama | None = None,
        llama: StubLlama | None = None,
        tts: StubTts | None = None,
        options: dict[str, Any] | None = None,
        before_start: Callable[[AppDeps], None] | None = None,
        start: bool = True,
        **settings: Any,
    ) -> Env:
        settings.setdefault("sair_token_mom", TOKENS["mom"])
        settings.setdefault("sair_token_dad", TOKENS["dad"])
        settings.setdefault("sair_token_owner", TOKENS["owner"])
        s = make_settings(tmp_path, **settings)
        clock = FakeClock()
        stub = ollama or StubOllama()
        stub_tts = tts or StubTts()
        notifier = RecNotifier()
        opts = {"run_workers": False, "plan_on_startup": False, **(options or {})}
        deps = build_deps(
            s, ollama=stub, llama=llama, tts=stub_tts, notifier=notifier, clock=clock, **opts
        )
        deps.static_dir = None
        if before_start:
            before_start(deps)
        app = create_app(s, deps)
        client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
        if start:
            client.__enter__()
            clients.append(client)
        return Env(client, deps, stub, stub_tts, notifier, clock)

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def env(make_env: Callable[..., Env]) -> Env:
    return make_env()
