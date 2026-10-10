"""Privacy canary harness (PRD §11.1, NFR-P2/P3/P4/P7). Stub models only, synthetic data only.

``canary_env`` builds the real FastAPI app (``create_app``) in ``SAIR_ENV=eval`` with every
sink the PRD names wired to a capture:

* Sentry: ``init_tracing(settings, transport=CaptureTransport())`` (fake transport; the DSN
  points at a loopback discard port, so even a misrouted envelope could not leave the host);
* ntfy: the real :class:`hub.notify.NtfyNotifier` over an ``httpx.MockTransport`` (URL,
  headers and body of every request are kept);
* logs: a DEBUG handler on the root logger with every named logger lowered to DEBUG
  (uvicorn, httpx, langgraph, fastapi, ...); structlog prints to stdout, which ``capfd`` holds;
* outbound: ``socket.socket.connect``/``connect_ex`` and ``socket.create_connection`` record
  every destination;
* HTTP responses: every response the test makes is recorded with whether it is an allowed
  content endpoint (see ``ALLOWED_CONTENT_ROUTES``).

Nothing in this file reads real data: settings use ``tmp_path`` only, never ``~/.sair``.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

import hub.app as hub_app
from hub import db, tracing
from hub.api import routes
from hub.api.deps import AppDeps, build_deps
from hub.app import create_app
from hub.graphs.journal import run_journal
from hub.models._http import CallStats
from hub.notify import NtfyNotifier
from hub.prompts.builders import ChatMessage
from hub.schemas import WalkStatus
from tests.unit._graphfix import FIXTURES, FakeClock, StubTts, make_settings

MARKER = "zebra-ultramarine-4471"
MARKER_B = MARKER.encode("utf-8")
GROUP = "Mom and Dad"
TOKENS = {"mom": "tok-mom-0123456789", "dad": "tok-dad-0123456789", "owner": "tok-own-0123456789"}
STATS = CallStats(latency_s=0.01, prompt_tokens=11, completion_tokens=7)

# A loopback discard port: the DSN host must be in ALLOWED_OUTBOUND_HOSTS for init_tracing to
# enable itself, and nothing listens there. The fake transport means nothing is ever sent.
FAKE_DSN = "http://canarypublickey@127.0.0.1:9/1"
NTFY_URL = "http://127.0.0.1:8090"
NTFY_TOPIC = "sair-canary"

# Responses that are the product itself and may carry content to the family/owner (PRD §9.3):
# the walk view (entry, facts), the diary and today's mission text. Every other response,
# and the headers of every response, must be marker-free.
ALLOWED_CONTENT_ROUTES = frozenset({"GET /walk/{walk_id}", "GET /diary", "GET /mission/today"})

# DB places the PRD §11.1 allows the marker in: walks.transcript, facts.*, entries.*.
ALLOWED_DB: frozenset[tuple[str, str | None]] = frozenset(
    {("walks", "transcript"), ("facts", None), ("entries", None)}
)


# The test's own HTTP client (starlette's TestClient uses httpx2) stands in for the phone; its
# request log lines are client-side, not hub logs. The hub's own httpx (ntfy) is kept.
CLIENT_LOGGERS = frozenset({"httpx2"})


def bearer(device: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[device]}"}


# ----------------------------------------------------------------------- captures


class CaptureTransport(Transport):
    """Fake Sentry transport: keeps every serialised envelope, sends nothing."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.envelopes: list[bytes] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope.serialize())

    def flush(self, timeout: float, callback: Any | None = None) -> None:
        return None

    def kill(self) -> None:
        return None


class RecordAll(logging.Handler):
    """Keeps every record as one string: message, args, every attribute, formatted exc_info."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []
        self._fmt = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.split(".")[0] in CLIENT_LOGGERS:
            return
        parts = [record.name, str(record.msg)]
        try:
            parts.append(record.getMessage())
        except Exception:
            parts.append(repr(record.args))
        for key, value in vars(record).items():
            if key not in ("msg", "args", "exc_info"):
                parts.append(f"{key}={value!r}")
        if record.exc_info:
            parts.append(self._fmt.formatException(record.exc_info))
        if record.stack_info:
            parts.append(str(record.stack_info))
        self.records.append(" | ".join(parts))


@dataclass
class Resp:
    route: str
    status: int
    headers: str
    body: bytes

    @property
    def allowed_content(self) -> bool:
        return self.route in ALLOWED_CONTENT_ROUTES and self.status == 200


Step = str | BaseException | Callable[..., str]


def _run(step: Step, *args: Any) -> str:
    if isinstance(step, BaseException):
        raise step
    if callable(step):
        return step(*args)
    return step


class CanaryModels:
    """Transcriber + writer + planner stub; each path sets what the next calls return."""

    def __init__(self, transcript: Step, writer: Sequence[Step], planner: Step) -> None:
        self.transcript: Step = transcript
        self.writer: list[Step] = list(writer)
        self.planner: Step = planner
        self.writer_calls = 0

    def set_writer(self, *steps: Step) -> None:
        self.writer = list(steps)
        self.writer_calls = 0

    def transcribe_with_stats(self, wav: Path) -> tuple[str, CallStats]:
        return _run(self.transcript, wav), STATS

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        role: Any = "writer",
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> tuple[str, CallStats]:
        if role == "planner":
            return _run(self.planner, messages), STATS
        step = self.writer[min(self.writer_calls, len(self.writer) - 1)]
        self.writer_calls += 1
        return _run(step, messages), STATS

    def warm(self) -> bool:
        return True


# ----------------------------------------------------------------------- the env


@dataclass
class CanaryEnv:
    client: TestClient
    deps: AppDeps
    models: CanaryModels
    sentry: CaptureTransport
    logs: RecordAll
    ntfy: list[tuple[str, str, bytes]]
    connections: list[tuple[int, str]]
    responses: list[Resp] = field(default_factory=list)
    tmp_path: Path = Path()

    # -- requests (every response is recorded)
    def request(self, method: str, url: str, route: str, **kw: Any) -> httpx.Response:
        r = self.client.request(method, url, **kw)
        headers = json.dumps(sorted(r.headers.multi_items()))
        self.responses.append(Resp(f"{method} {route}", r.status_code, headers, r.content))
        return r

    def post_walk(
        self,
        fixture: str,
        filename: str,
        content_type: str,
        *,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        query: str = "",
    ) -> str:
        r = self.request(
            "POST",
            f"/walk{query}",
            "/walk",
            headers={**bearer("mom"), **(headers or {})},
            files={"audio": (filename, (FIXTURES / fixture).read_bytes(), content_type)},
            data=data or {},
        )
        assert r.status_code == 202, r.status_code
        walk_id: str = r.json()["walk_id"]
        return walk_id

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Wait for every background walk task and the planner queue to drain."""
        end = time.monotonic() + timeout
        tasks: set[Any] = self.client.app.state.walk_tasks  # type: ignore[attr-defined]
        while tasks and time.monotonic() < end:
            time.sleep(0.05)
        self.deps.planner.submit(lambda: None).result(timeout=timeout)

    def walk_status(self, walk_id: str) -> str:
        conn = self.deps.connect()
        try:
            row = db.get_walk_row(conn, walk_id)
        finally:
            conn.close()
        assert row is not None
        return str(row["status"])

    def wait_done(self, walk_id: str) -> None:
        self.wait_idle()
        assert self.walk_status(walk_id) == WalkStatus.DONE.value

    # -- sink scans; each returns violation labels (never content)
    def sentry_hits(self) -> list[str]:
        envs = self.sentry.envelopes
        return [f"sentry envelope #{i}" for i, e in enumerate(envs) if MARKER_B in e]

    def ntfy_hits(self) -> list[str]:
        out = []
        for i, (url, headers, body) in enumerate(self.ntfy):
            if MARKER in url or MARKER in headers or MARKER_B in body:
                out.append(f"ntfy request #{i}")
        return out

    def log_hits(self) -> list[str]:
        return [f"log record #{i}" for i, rec in enumerate(self.logs.records) if MARKER in rec]

    def response_hits(self) -> list[str]:
        out = []
        for r in self.responses:
            if MARKER in r.headers:
                out.append(f"response headers {r.route} {r.status}")
            if not r.allowed_content and MARKER_B in r.body:
                out.append(f"response body {r.route} {r.status}")
        return out

    def connection_hits(self) -> list[str]:
        conns = self.connections
        return [f"outbound {fam}:{dest}" for fam, dest in conns if not _loopback(fam, dest)]

    def file_hits(self) -> list[str]:
        """No path under the test's dirs is named after content."""
        return [f"file name {p.name[:8]}..." for p in self.tmp_path.rglob("*") if MARKER in p.name]

    def db_hits(self, extra_allowed: frozenset[tuple[str, str | None]] = frozenset()) -> list[str]:
        allowed = ALLOWED_DB | extra_allowed
        return [
            f"db {t}.{c}"
            for t, c in db_marker_columns(self.deps.settings.db_path)
            if (t, c) not in allowed and (t, None) not in allowed
        ]


def db_marker_columns(path: Path) -> list[tuple[str, str]]:
    """Every (table, column) whose value (as text or bytes) contains the marker."""
    conn = db.connect(path)
    try:
        tables = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        hits: list[tuple[str, str]] = []
        for t in tables:
            cols = [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{t}")')]
            for c in cols:
                n = conn.execute(
                    f'SELECT COUNT(*) FROM "{t}" WHERE instr(CAST("{c}" AS BLOB), ?) > 0',
                    (MARKER_B,),
                ).fetchone()[0]
                if n:
                    hits.append((t, c))
        return hits
    finally:
        conn.close()


def _loopback(family: int, dest: str) -> bool:
    if family == socket.AF_UNIX:
        return True
    host = dest.rsplit(":", 1)[0].strip("[]")
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def _fmt_addr(address: Any) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


@pytest.fixture
def canary_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CanaryEnv]:
    # --- outbound monitor (installed first so it covers app startup)
    connections: list[tuple[int, str]] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def connect(self: socket.socket, address: Any) -> None:
        connections.append((self.family, _fmt_addr(address)))
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        connections.append((self.family, _fmt_addr(address)))
        return real_connect_ex(self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        connections.append((socket.AF_INET, _fmt_addr(address)))
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)

    # --- every logger at DEBUG into one handler
    handler = RecordAll()
    root = logging.getLogger()
    saved = {"root": root.level}
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    names = ["uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore", "langgraph",
             "langchain", "fastapi", "starlette", "asyncio", "sentry_sdk", "sentry_sdk.errors",
             "hub", "multipart", "python_multipart"]  # fmt: skip
    names += list(logging.root.manager.loggerDict)
    levels: dict[str, int] = {}
    for name in set(names):
        lg = logging.getLogger(name)
        levels[name] = lg.level
        lg.setLevel(logging.DEBUG)

    # --- settings: eval env, tmp dirs only, Sentry + ntfy on (both captured)
    settings = make_settings(
        tmp_path,
        sair_env="eval",
        sentry_dsn=FAKE_DSN,
        ntfy_url=NTFY_URL,
        ntfy_topic=NTFY_TOPIC,
        sair_log_dir=tmp_path / "hub-logs",
        sair_audio_cache_dir=tmp_path / "cache",
        sair_checkpoint_dir=tmp_path / "checkpoints",
        sair_token_mom=TOKENS["mom"],
        sair_token_dad=TOKENS["dad"],
        sair_token_owner=TOKENS["owner"],
    )

    # --- Sentry: the app's own startup hook, with the fake transport
    sentry = CaptureTransport()
    monkeypatch.setattr(
        hub_app, "_init_tracing", lambda s: tracing.init_tracing(s, transport=sentry)
    )

    # Tracing integration as documented in hub/tracing.py (the app does not wrap the graph
    # itself yet): one transaction per journal run, one span per node.
    def traced_run_journal(walk_id: str, *args: Any, **kwargs: Any) -> Any:
        with tracing.graph_transaction("journal", walk_id=walk_id):
            return run_journal(walk_id, *args, **kwargs)

    monkeypatch.setattr(routes, "run_journal", traced_run_journal)

    # --- ntfy: the real notifier over a mock transport
    ntfy: list[tuple[str, str, bytes]] = []

    def ntfy_handler(request: httpx.Request) -> httpx.Response:
        ntfy.append((str(request.url), json.dumps(sorted(request.headers.multi_items())),
                     request.content))  # fmt: skip
        return httpx.Response(200, json={"id": "x"})

    ntfy_client = httpx.Client(transport=httpx.MockTransport(ntfy_handler))
    notifier = NtfyNotifier(settings, client=ntfy_client)

    models = CanaryModels(transcript="", writer=[""], planner="")
    deps = build_deps(
        settings,
        ollama=models,
        llama=None,
        tts=StubTts(),
        notifier=notifier,
        clock=FakeClock(),
        run_workers=False,
        plan_on_startup=False,
    )
    deps.static_dir = None
    deps.journal.span = tracing.node_span
    deps.mission.span = tracing.node_span

    app = create_app(settings, deps)
    client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    client.__enter__()
    env = CanaryEnv(client, deps, models, sentry, handler, ntfy, connections, tmp_path=tmp_path)
    assert tracing.is_enabled(), "tracing must be on in eval so the Sentry sink is exercised"
    try:
        yield env
    finally:
        try:
            client.__exit__(None, None, None)
        finally:
            tracing.shutdown_tracing()
            root.removeHandler(handler)
            root.setLevel(saved["root"])
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)
