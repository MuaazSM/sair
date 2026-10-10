"""Shared plumbing for the journal and mission graphs.

* Client protocols, so the graphs depend on behaviour, not on the concrete model clients
  (tests pass stubs; production passes ``hub.models.*``).
* ``timed_node``: one LangGraph node = one function = one metric row (+ one span in dev/eval).
* Small query helpers the graphs need that ``hub/db.py`` does not expose (reported upstream).
* File helpers for the raw-audio temp dir (FR-J6).

Nothing here logs or raises with content (NFR-P4): log calls carry IDs, counts and codes only.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol

import structlog

from hub import db, tracing
from hub.models._http import CallStats
from hub.prompts.builders import ChatMessage
from hub.schemas import WalkStatus

__all__ = [
    "graph_run",
    "REPO_ROOT",
    "Clock",
    "ConnectFn",
    "SpanFn",
    "TranscriberLike",
    "TtsLike",
    "WriterLike",
    "count_done_walks",
    "delete_path",
    "find_raw_audio",
    "mission_text",
    "null_span",
    "open_conn",
    "pending_walks",
    "repo_path",
    "set_duration",
    "timed_node",
    "utc_now",
    "work_dir_for",
]

log = structlog.get_logger("hub.graphs")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

Clock = Callable[[], dt.datetime]
ConnectFn = Callable[[], sqlite3.Connection]
SpanFn = Callable[[str], AbstractContextManager[None]]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def graph_run(
    name: str, *, walk_id: str | None = None, mission_id: str | None = None
) -> AbstractContextManager[None]:
    """One tracing transaction per graph run (dev/eval only; a no-op in prod).

    A caller that already opened a transaction (tests, eval harness) keeps it: no nesting.
    """
    if not tracing.is_enabled():
        return contextlib.nullcontext()
    import sentry_sdk  # only reached when dev/eval tracing is on

    if sentry_sdk.get_current_scope().transaction is not None:
        return contextlib.nullcontext()
    return tracing.graph_transaction(name, walk_id=walk_id, mission_id=mission_id)


def null_span(_name: str) -> AbstractContextManager[None]:
    return contextlib.nullcontext()


def repo_path(p: Path) -> Path:
    """Resolve a settings path: absolute or cwd-relative if it exists, else repo-relative."""
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


# --------------------------------------------------------------------- protocols


class TranscriberLike(Protocol):
    """The parts of ``hub.models.ollama.OllamaClient`` the graphs use."""

    def transcribe_with_stats(self, wav: Path) -> tuple[str, CallStats]: ...

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        role: Any = ...,
        temperature: float | None = ...,
        num_predict: int | None = ...,
    ) -> tuple[str, CallStats]: ...

    def warm(self) -> bool: ...


class WriterLike(Protocol):
    """The part of ``hub.models.llama_server.LlamaServerClient`` the journal graph uses."""

    def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        grammar: str | None = ...,
        max_tokens: int | None = ...,
    ) -> tuple[str, CallStats]: ...


class TtsLike(Protocol):
    def render(self, text: str, out_path: Path) -> Path: ...


# --------------------------------------------------------------------- DB helpers


@contextlib.contextmanager
def open_conn(connect: ConnectFn) -> Iterator[sqlite3.Connection]:
    """A fresh connection per node (graph nodes may run on executor threads)."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def mission_text(conn: sqlite3.Connection, mission_id: str | None) -> str | None:
    if mission_id is None:
        return None
    row = conn.execute("SELECT text FROM missions WHERE id = ?", (mission_id,)).fetchone()
    return None if row is None else str(row["text"])


def set_duration(conn: sqlite3.Connection, walk_id: str, duration: float) -> None:
    conn.execute("UPDATE walks SET duration_s = ? WHERE id = ?", (duration, walk_id))


def count_done_walks(conn: sqlite3.Connection, group: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM walks WHERE group_name = ? AND status = ?",
        (group, WalkStatus.DONE.value),
    ).fetchone()
    return int(row[0])


def pending_walks(conn: sqlite3.Connection, reasons: Sequence[str]) -> list[sqlite3.Row]:
    """Walks still ``received`` with one of ``reasons``: id, group, mission, upload time only."""
    marks = ",".join("?" for _ in reasons)
    rows: list[sqlite3.Row] = conn.execute(
        "SELECT id, group_name, mission_id, recorded_at, reason FROM walks "
        f"WHERE status = 'received' AND reason IN ({marks}) ORDER BY recorded_at",
        tuple(reasons),
    ).fetchall()
    return rows


# --------------------------------------------------------------------- files (FR-J6)


def work_dir_for(tmp_dir: Path, walk_id: str) -> Path:
    """Per-walk scratch dir for the decoded WAV and any chunks."""
    return tmp_dir / f"work-{walk_id}"


def find_raw_audio(tmp_dir: Path, walk_id: str) -> Path | None:
    """The upload convention is ``<tmp_dir>/<walk_id><ext>`` (the app saves it that way)."""
    if not tmp_dir.is_dir():
        return None
    for p in sorted(tmp_dir.glob(f"{walk_id}.*")):
        if p.is_file():
            return p
    return None


def delete_path(p: Path | str | None) -> None:
    """Delete a file or directory if present. Never raises."""
    if p is None:
        return
    path = Path(p)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        log.warning("audio_delete_failed", kind="oserror")


# --------------------------------------------------------------------- node timing


def timed_node[S: Mapping[str, object]](
    name: str,
    fn: Callable[[S], dict[str, Any]],
    *,
    connect: ConnectFn,
    span: SpanFn,
    walk_id_key: str | None = "walk_id",
) -> Callable[[S], dict[str, Any]]:
    """Wrap a node: span (dev/eval), wall time, one ``metrics`` row (local DB only, NFR-P7).

    A node may return ``_tokens_in`` / ``_tokens_out`` keys; they go to the metric row and are
    stripped from the state update. ``latency_ms`` in the state is extended with this node.
    """

    def run(state: S) -> dict[str, Any]:
        t0 = time.perf_counter()
        with span(name):
            update = fn(state)
        ms = (time.perf_counter() - t0) * 1000.0
        tin = update.pop("_tokens_in", None)
        tout = update.pop("_tokens_out", None)
        prev = state.get("latency_ms")
        lat: dict[str, float] = dict(prev) if isinstance(prev, dict) else {}
        lat[name] = round(lat.get(name, 0.0) + ms, 1)
        update["latency_ms"] = lat
        wid = state.get(walk_id_key) if walk_id_key else None
        try:
            with open_conn(connect) as conn:
                db.insert_metric(
                    conn,
                    name,
                    ms,
                    walk_id=wid if isinstance(wid, str) else None,
                    tokens_in=tin if isinstance(tin, int) else None,
                    tokens_out=tout if isinstance(tout, int) else None,
                )
        except sqlite3.Error:
            log.warning("metric_write_failed", node=name)
        return update

    run.__name__ = name
    return run
