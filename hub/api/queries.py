"""Read helpers the HTTP layer needs that ``hub/db.py`` does not expose.

Rules: dates only (NFR-P5); nothing that feeds ``/status`` returns text (FR-O2); the walks
``transcript`` column is never selected here.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

from hub import db
from hub.graphs.common import find_raw_audio
from hub.schemas import (
    CanaryResult,
    Certainty,
    FactChip,
    FactKind,
    LatencyStat,
    Mission,
    MissionResult,
    MissionSource,
    MissionType,
    ReasonCode,
    StatusResponse,
    WalkStatus,
    WalkStatusResponse,
)

__all__ = [
    "JOURNAL_TARGET_P50_MS",
    "JOURNAL_TARGET_P95_MS",
    "journal_total_latency",
    "local_date",
    "mission_by_id",
    "mission_group",
    "requeue_interrupted",
    "status_response",
    "walk_exists",
    "walk_view",
]

JOURNAL_TARGET_P50_MS = 15_000.0  # NFR-L2
JOURNAL_TARGET_P95_MS = 30_000.0


def local_date(stamp: str, tz: ZoneInfo) -> dt.date:
    """Calendar date of a stored UTC timestamp in the household time zone."""
    return db.local_date(stamp, tz)


def mission_group(conn: sqlite3.Connection, mission_id: str) -> str | None:
    row = conn.execute("SELECT group_name FROM missions WHERE id = ?", (mission_id,)).fetchone()
    return None if row is None else str(row["group_name"])


def mission_by_id(conn: sqlite3.Connection, mission_id: str) -> Mission | None:
    row = conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
    if row is None:
        return None
    return Mission(
        id=row["id"],
        date=db.iso_to_date(row["date"]),
        group=row["group_name"],
        type=MissionType(row["type"]),
        text=row["text"],
        references=json.loads(row["references_json"]),
        source=MissionSource(row["source"]),
    )


def walk_exists(conn: sqlite3.Connection, walk_id: str) -> bool:
    return conn.execute("SELECT 1 FROM walks WHERE id = ?", (walk_id,)).fetchone() is not None


def walk_view(
    conn: sqlite3.Connection, walk_id: str, tz: ZoneInfo
) -> tuple[str, WalkStatusResponse] | None:
    """(group, response) for ``GET /walk/{id}``. ``reason`` is set only when lost."""
    row = conn.execute(
        "SELECT group_name, recorded_at, status, mission_result, reason FROM walks WHERE id = ?",
        (walk_id,),
    ).fetchone()
    if row is None:
        return None
    status = WalkStatus(row["status"])
    date = local_date(row["recorded_at"], tz)
    if status is WalkStatus.LOST:
        reason: ReasonCode | None
        try:
            reason = ReasonCode(row["reason"]) if row["reason"] else None
        except ValueError:
            reason = None
        return row["group_name"], WalkStatusResponse(status=status, date=date, reason=reason)
    if status is WalkStatus.RECEIVED:
        return row["group_name"], WalkStatusResponse(status=status, date=date)
    entry_row = conn.execute("SELECT text FROM entries WHERE walk_id = ?", (walk_id,)).fetchone()
    facts = [
        FactChip(
            subject=f["subject"],
            kind=FactKind(f["kind"]),
            certainty=Certainty(f["certainty"]),
            place=f["place"],
        )
        for f in conn.execute(
            "SELECT kind, subject, place, certainty FROM facts "
            "WHERE walk_id = ? AND verified = 1 ORDER BY rowid",
            (walk_id,),
        )
    ]
    return row["group_name"], WalkStatusResponse(
        status=status,
        date=date,
        entry=None if entry_row is None else str(entry_row["text"]),
        facts=facts,
        mission_result=MissionResult(row["mission_result"] or "not_mentioned"),
    )


# ---------------------------------------------------------------- startup recovery


def requeue_interrupted(conn: sqlite3.Connection, tmp_dir: Path) -> int:
    """Startup: hand walks a crash left in ``received`` (no reason, no entry) to the retry worker.

    * transcript stored, no entry (crash between store steps) -> WRITER_TRANSIENT, so the
      retry worker re-runs the graph from ``write_entry`` (transcript from the DB);
    * raw upload still on disk -> TRANSCRIBE_TRANSIENT (retried inside the 10-minute window);
    * neither -> ``lost`` / RETRY_WINDOW_EXPIRED (nothing left to work from).

    Run after the stale-audio sweep. Returns the number of walks touched.
    """
    rows = conn.execute(
        "SELECT w.id, w.transcript IS NOT NULL AS has_t FROM walks w "
        "WHERE w.status = 'received' AND w.reason IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM entries e WHERE e.walk_id = w.id)"
    ).fetchall()
    for r in rows:
        wid = str(r["id"])
        if r["has_t"]:
            db.update_walk_status(conn, wid, WalkStatus.RECEIVED, ReasonCode.WRITER_TRANSIENT)
        elif find_raw_audio(tmp_dir, wid) is not None:
            db.update_walk_status(conn, wid, WalkStatus.RECEIVED, ReasonCode.TRANSCRIBE_TRANSIENT)
        else:
            db.update_walk_status(conn, wid, WalkStatus.LOST, ReasonCode.RETRY_WINDOW_EXPIRED)
    return len(rows)


# ------------------------------------------------------------------------ /status


def _nearest_rank(sorted_vals: list[float], pct: float) -> float:
    return sorted_vals[max(1, math.ceil(pct / 100 * len(sorted_vals))) - 1]


def journal_total_latency(conn: sqlite3.Connection) -> LatencyStat | None:
    """End-to-end journal time per finished walk (sum of node times), vs NFR-L2 targets."""
    totals: list[float] = []
    for r in conn.execute(
        "SELECT latency_ms_json FROM walks WHERE status = 'done' AND latency_ms_json IS NOT NULL"
    ):
        try:
            data = json.loads(r["latency_ms_json"])
        except ValueError:
            continue
        if isinstance(data, dict):
            vals = [float(v) for v in data.values() if isinstance(v, int | float)]
            if vals:
                totals.append(sum(vals))
    if not totals:
        return None
    totals.sort()
    return LatencyStat(
        n=len(totals),
        p50_ms=_nearest_rank(totals, 50),
        p95_ms=_nearest_rank(totals, 95),
        target_p50_ms=JOURNAL_TARGET_P50_MS,
        target_p95_ms=JOURNAL_TARGET_P95_MS,
    )


def _canary(conn: sqlite3.Connection, tz: ZoneInfo) -> tuple[CanaryResult | None, dt.date | None]:
    row = conn.execute(
        "SELECT value, updated_at FROM meta WHERE key = 'last_canary_result'"
    ).fetchone()
    if row is None:
        return None, None
    try:
        return CanaryResult(row["value"]), local_date(row["updated_at"], tz)
    except ValueError:
        return None, None


def status_response(
    conn: sqlite3.Connection, tmp_dir: Path, tz: ZoneInfo, model_health: dict[str, bool]
) -> StatusResponse:
    """FR-O2: metadata only. Group names are configuration; everything else is a count/code."""
    counts = db.status_counts(conn)
    latency = db.latency_percentiles(conn)
    total = journal_total_latency(conn)
    if total is not None:
        latency["journal_total"] = total
    canary, canary_date = _canary(conn, tz)
    return StatusResponse(
        model_health=model_health,
        latency=latency,
        per_group=counts.per_group,
        grounding_pass_rate=counts.grounding_pass_rate,
        grounding_n=counts.grounding_n,
        reason_counts=counts.reason_counts,
        fallback_count=counts.fallback_count,
        feedback_pending=counts.feedback_pending,
        raw_audio_on_disk=db.raw_audio_on_disk(tmp_dir),
        last_canary_result=canary,
        last_canary_date=canary_date,
    )
