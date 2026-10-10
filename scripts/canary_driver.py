"""Eval-only canary driver for ``make canary`` (PRD §11.1; NFR-P3, NFR-P4, NFR-P7).

Runs the journal and mission graphs in this process against the REAL local model servers
(Ollama, and llama-server if it is up) with a stub transcriber whose transcript carries the
canary marker (a spoken marker transcribes unpredictably). Sinks captured in-process: Sentry
(fake transport), ntfy (real notifier over a mock transport), outbound sockets, the eval DB.
The calling script greps this process's stdout/stderr and the model-server log files.

Prints nothing but counts. Writes a JSON summary of counts to ``--summary``. Refuses to run
unless ``SAIR_ENV=eval`` and the data dir is a scratch dir (never ``~/.sair``). Never reads an
env file.

    uv run python scripts/canary_driver.py run --summary <file>
    uv run python scripts/canary_driver.py write-meta passed|failed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hub import db, tracing  # noqa: E402
from hub.graphs.journal import JournalDeps, run_journal  # noqa: E402
from hub.graphs.mission import MissionDeps, plan_mission  # noqa: E402
from hub.models._http import CallStats  # noqa: E402
from hub.models.llama_server import LlamaServerClient  # noqa: E402
from hub.models.ollama import OllamaClient  # noqa: E402
from hub.notify import NtfyNotifier  # noqa: E402
from hub.prompts.builders import ChatMessage  # noqa: E402
from hub.schemas import WalkStatus, WriterBackend  # noqa: E402
from hub.settings import Settings  # noqa: E402

DEFAULT_MARKER = "zebra-ultramarine-4471"
FIXTURE = REPO / "tests" / "fixtures" / "audio" / "en_20s.webm"
GROUP = "Mom and Dad"
FAKE_DSN = "http://canarypublickey@127.0.0.1:9/1"
STATS = CallStats(latency_s=0.0, prompt_tokens=None, completion_tokens=None)
# PRD §11.1 allow-list, plus missions.text (PRD §9.4 content column; open PRD question).
ALLOWED = {("walks", "transcript"), ("facts", "*"), ("entries", "*"), ("missions", "text")}


def _marker() -> str:
    return os.environ.get("CANARY_MARKER") or DEFAULT_MARKER


def _settings(**update: Any) -> Settings:
    if os.environ.get("SAIR_ENV", "").lower() != "eval":
        raise SystemExit("canary_driver: SAIR_ENV must be eval")
    s = Settings(_env_file=None, **update)  # type: ignore[call-arg]
    home_sair = (Path.home() / ".sair").resolve()
    for p in (s.data_dir, s.db_path, s.tmp_dir, s.audio_dir):
        r = p.resolve()
        if r == home_sair or home_sair in r.parents:
            raise SystemExit("canary_driver: refusing to use ~/.sair (live data)")
    return s


class Capture:
    """Fake Sentry transport."""

    def __init__(self) -> None:
        from sentry_sdk.transport import Transport

        marker = _marker().encode()

        class _T(Transport):
            def __init__(self) -> None:
                super().__init__(None)
                self.total = 0
                self.hits = 0

            def capture_envelope(self, envelope: Any) -> None:
                self.total += 1
                if marker in envelope.serialize():
                    self.hits += 1

            def flush(self, timeout: float, callback: Any | None = None) -> None:
                return None

            def kill(self) -> None:
                return None

        self.transport = _T()


class MarkerTranscriber:
    """Stub transcription (marker transcript) + the real Ollama client for everything else."""

    def __init__(self, real: OllamaClient, transcript: str) -> None:
        self.real = real
        self.transcript: str | BaseException = transcript

    def transcribe_with_stats(self, wav: Path) -> tuple[str, CallStats]:
        if isinstance(self.transcript, BaseException):
            raise self.transcript
        return self.transcript, STATS

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        role: Any = "writer",
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> tuple[str, CallStats]:
        return self.real.chat_json(
            messages, schema, role=role, temperature=temperature, num_predict=num_predict
        )

    def warm(self) -> bool:
        return self.real.warm()


def _loopback(family: int, addr: Any) -> bool:
    if family == socket.AF_UNIX:
        return True
    host = str(addr[0]) if isinstance(addr, tuple) and addr else str(addr)
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def run(summary_path: Path) -> int:
    marker = _marker()
    marker_b = marker.encode()
    # --- outbound monitor
    outbound = {"total": 0, "non_loopback": 0}
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        outbound["total"] += 1
        if not _loopback(self.family, address):
            outbound["non_loopback"] += 1
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        outbound["total"] += 1
        if not _loopback(self.family, address):
            outbound["non_loopback"] += 1
        return real_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]

    base = _settings()
    llama_up = LlamaServerClient(base).health()
    backend = WriterBackend.LLAMA if llama_up else WriterBackend.GEMMA
    s = _settings(
        writer_backend=backend,
        sentry_dsn=FAKE_DSN,
        ntfy_topic="sair-canary",
        allowed_outbound_hosts="127.0.0.1,localhost",
    )
    s.tmp_dir.mkdir(parents=True, exist_ok=True)
    db.init_db(s.db_path).close()

    sentry = Capture().transport
    tracing.init_tracing(s, transport=sentry)

    ntfy = {"total": 0, "hits": 0}

    def ntfy_handler(request: httpx.Request) -> httpx.Response:
        ntfy["total"] += 1
        blob = str(request.url).encode() + json.dumps(request.headers.multi_items()).encode()
        if marker_b in blob + request.content:
            ntfy["hits"] += 1
        return httpx.Response(200, json={"id": "x"})

    notifier = NtfyNotifier(s, client=httpx.Client(transport=httpx.MockTransport(ntfy_handler)))
    real_ollama = OllamaClient(s)
    ollama_up = real_ollama.health()
    models = MarkerTranscriber(
        real_ollama,
        f"We walked to the park. We saw a {marker} bird sitting on the wall near the gate. "
        "It was cool and windy, and Dad felt a bit dizzy on the way back.",
    )
    llama = LlamaServerClient(s) if llama_up else None

    def connect_db() -> Any:
        return db.connect(s.db_path)

    mission = MissionDeps(settings=s, connect=connect_db, ollama=models, tts=None,
                          scheduler=lambda g, d: None, span=tracing.node_span)  # fmt: skip
    journal = JournalDeps(settings=s, connect=connect_db, ollama=models, llama=llama,
                          notifier=notifier, mission_trigger=lambda g: None,
                          span=tracing.node_span)  # fmt: skip

    statuses: dict[str, int] = {}
    errors: dict[str, int] = {}
    for i in range(4):  # 3 seed walks, then the planner sees the marker facts
        walk_id = db.new_id()
        if i == 3:
            models.transcript = RuntimeError(marker)  # exception path (transient)
        dst = s.tmp_dir / f"{walk_id}{FIXTURE.suffix}"
        shutil.copyfile(FIXTURE, dst)
        try:
            with tracing.graph_transaction("journal", walk_id=walk_id):
                out = run_journal(walk_id, dst, GROUP, None, journal)
            statuses[out.status.value] = statuses.get(out.status.value, 0) + 1
        except Exception as exc:
            errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
    mission_source = None
    if ollama_up:
        try:
            with tracing.graph_transaction("mission"):
                m = plan_mission(GROUP, dt.date.today(), mission, replace=True)
            mission_source = m.source.value if m else None
        except Exception as exc:
            errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
    tracing.shutdown_tracing()
    real_ollama.close()
    if llama is not None:
        llama.close()

    # --- eval DB: per-column marker counts (counts only)
    conn = db.connect(s.db_path)
    try:
        columns: dict[str, int] = {}
        for (t,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            for col in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
                c = str(col[1])
                n = conn.execute(
                    f'SELECT COUNT(*) FROM "{t}" WHERE instr(CAST("{c}" AS BLOB), ?) > 0',
                    (marker_b,),
                ).fetchone()[0]
                if n:
                    columns[f"{t}.{c}"] = int(n)
        done = conn.execute(
            "SELECT COUNT(*) FROM walks WHERE status = ?", (WalkStatus.DONE.value,)
        ).fetchone()[0]
    finally:
        conn.close()
    disallowed = {
        k: v for k, v in columns.items()
        if tuple(k.split(".", 1)) not in ALLOWED and (k.split(".", 1)[0], "*") not in ALLOWED
    }  # fmt: skip
    leftover_audio = sum(1 for p in s.tmp_dir.rglob("*") if p.is_file())
    summary = {
        "ollama_up": ollama_up,
        "llama_server_up": llama_up,
        "writer_backend": backend.value,
        "walk_statuses": statuses,
        "walks_done": int(done),
        "errors": errors,
        "mission_source": mission_source,
        "sentry_envelopes": sentry.total,
        "sentry_hits": sentry.hits,
        "ntfy_requests": ntfy["total"],
        "ntfy_hits": ntfy["hits"],
        "outbound_connections": outbound["total"],
        "outbound_non_loopback": outbound["non_loopback"],
        "db_marker_columns": columns,
        "db_disallowed": disallowed,
        "tmp_files_left": leftover_audio,  # the exception walk keeps its audio for retry
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return 0


def write_meta(result: str) -> int:
    if result not in ("passed", "failed"):
        raise SystemExit("canary_driver: result must be passed or failed")
    s = _settings()
    conn = db.init_db(s.db_path)
    try:
        db.set_meta(conn, "last_canary_result", result)
        db.set_meta(conn, "last_canary_date", dt.date.today().isoformat())
    finally:
        conn.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--summary", type=Path, required=True)
    w = sub.add_parser("write-meta")
    w.add_argument("result")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args.summary)
    return write_meta(args.result)


if __name__ == "__main__":
    raise SystemExit(main())
