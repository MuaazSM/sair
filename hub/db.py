"""SQLite storage (PRD §9.4): WAL, foreign keys, numbered migrations, typed repository.

Rules this module upholds:

* No BLOB columns, ever (FR-J6, §9.4). Raw audio is never stored; mission audio is a file at
  ``audio_dir / f"{mission_id}.mp3"`` (see :func:`mission_audio_path`).
* ``recorded_at`` is stored as a timezone-aware ISO timestamp (ordering). Everything the
  API exposes is a ``date`` (NFR-P5): use :func:`iso_to_date`.
* No helper that feeds ``/status`` returns text (FR-O2): only counts, rates and enum codes.
* Device tokens are stored as SHA-256 hashes and compared in constant time.
* Nothing here logs; errors raised never embed row content (NFR-P4).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import shutil
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from hub.schemas import (
    Certainty,
    DiaryEntry,
    DiaryFact,
    DiarySummary,
    FactKind,
    GroupCounts,
    LatencyStat,
    Mission,
    MissionResult,
    MissionSource,
    MissionType,
    ReasonCode,
    VerifiedFact,
    WalkStatus,
    WriterOutput,
)
from hub.settings import Settings

__all__ = [
    "GroundingSummary",
    "MIGRATIONS",
    "StatusCounts",
    "Walker",
    "connect",
    "diary",
    "diary_summary",
    "feedback_for_walk",
    "forget",
    "get_cached_mission",
    "get_meta",
    "get_walk_row",
    "hash_token",
    "init_db",
    "insert_feedback",
    "insert_metric",
    "insert_mission",
    "insert_walk",
    "iso_to_date",
    "latency_percentiles",
    "latest_missions",
    "mission_audio_path",
    "raw_audio_on_disk",
    "record_mission_refs",
    "schema_version",
    "set_meta",
    "set_transcript",
    "status_counts",
    "store_journal_result",
    "transaction",
    "update_walk_status",
    "upsert_walkers_from_settings",
    "verified_facts_for_group",
    "verify_token",
    "walker_for_token",
]

# ------------------------------------------------------------------ migrations

_V1 = """
CREATE TABLE walkers (
    id                TEXT PRIMARY KEY,
    device            TEXT NOT NULL UNIQUE,
    group_name        TEXT NOT NULL,
    device_token_hash TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE missions (
    id              TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    group_name      TEXT NOT NULL,
    type            TEXT NOT NULL CHECK (type IN ('return','change','sense','count','seed')),
    text            TEXT NOT NULL,
    references_json TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL CHECK (source IN ('model','seed','fallback')),
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_missions_group_date ON missions (group_name, date);

CREATE TABLE walks (
    id              TEXT PRIMARY KEY,
    mission_id      TEXT REFERENCES missions (id),
    group_name      TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,
    duration_s      REAL,
    transcript      TEXT,
    mission_result  TEXT CHECK (mission_result IN ('done','partly','not_done','not_mentioned')),
    mission_span    TEXT,
    writer          TEXT CHECK (writer IN ('gemma','llama','template')),
    check_passed    INTEGER CHECK (check_passed IN (0,1)),
    regen_count     INTEGER NOT NULL DEFAULT 0 CHECK (regen_count >= 0),
    status          TEXT NOT NULL CHECK (status IN ('received','done','lost')),
    latency_ms_json TEXT,
    reason          TEXT
);
CREATE INDEX idx_walks_group_recorded ON walks (group_name, recorded_at);

CREATE TABLE facts (
    id         TEXT PRIMARY KEY,
    walk_id    TEXT NOT NULL REFERENCES walks (id) ON DELETE CASCADE,
    span       TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN
               ('sight','sound','smell','person','animal','plant','weather','other')),
    subject    TEXT NOT NULL,
    place      TEXT,
    certainty  TEXT NOT NULL CHECK (certainty IN ('sure','unsure')),
    verified   INTEGER NOT NULL CHECK (verified IN (0,1)),
    created_at TEXT NOT NULL
);
CREATE INDEX idx_facts_walk ON facts (walk_id);

CREATE TABLE entries (
    walk_id    TEXT PRIMARY KEY REFERENCES walks (id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE mission_refs (
    mission_id TEXT NOT NULL REFERENCES missions (id) ON DELETE CASCADE,
    fact_id    TEXT NOT NULL REFERENCES facts (id) ON DELETE CASCADE,
    PRIMARY KEY (mission_id, fact_id)
);
CREATE INDEX idx_mission_refs_fact ON mission_refs (fact_id);

CREATE TABLE feedback (
    walk_id    TEXT PRIMARY KEY REFERENCES walks (id) ON DELETE CASCADE,
    correct    INTEGER NOT NULL CHECK (correct IN (0,1)),
    note       TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE metrics (
    walk_id    TEXT,
    node       TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    tokens_in  INTEGER,
    tokens_out INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_metrics_node ON metrics (node);
"""

# Append only. Index i holds the script that upgrades schema version i to i + 1.
MIGRATIONS: tuple[str, ...] = (_V1,)

_BUSY_TIMEOUT_MS = 5000


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection: WAL, foreign keys, busy timeout, autocommit (see ``transaction``)."""
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: Path | str) -> sqlite3.Connection:
    """Create the file (and parent dir) if needed, apply pending migrations, return a connection."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    current = schema_version(conn)
    for target, script in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in _split_script(script):
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version={target}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return conn


def _split_script(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``; rolls back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --------------------------------------------------------------------- helpers


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _iso(ts: dt.datetime | None) -> str:
    return (ts or _now()).isoformat()


def iso_to_date(value: str) -> dt.date:
    """Date part of a stored ISO timestamp or date, as a ``date`` (never a ``datetime``)."""
    return dt.datetime.fromisoformat(value).date() if "T" in value else dt.date.fromisoformat(value)


def new_id() -> str:
    return uuid.uuid4().hex


def mission_audio_path(audio_dir: Path, mission_id: str) -> Path:
    """FR-M8: mission audio is a file, derived from the mission id."""
    return audio_dir / f"{mission_id}.mp3"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, token_hash: str) -> bool:
    """Constant-time check; an empty token never verifies."""
    if not token:
        return False
    return hmac.compare_digest(hash_token(token), token_hash)


# --------------------------------------------------------------------- walkers


@dataclass(frozen=True, slots=True)
class Walker:
    id: str
    device: str
    group_name: str

    @property
    def is_owner(self) -> bool:
        return self.device == "owner"


OWNER_DEVICE = "owner"
OWNER_GROUP = "owner"


def upsert_walkers_from_settings(
    conn: sqlite3.Connection, settings: Settings, now: dt.datetime | None = None
) -> int:
    """Sync walkers from ``SAIR_DEVICE_GROUPS`` + tokens. Devices without a token are skipped.

    The owner device (``SAIR_TOKEN_OWNER``) is stored with group ``owner``. Returns the number of
    walkers written.
    """
    pairs = dict(settings.device_groups)
    if settings.token_for(OWNER_DEVICE):
        pairs[OWNER_DEVICE] = OWNER_GROUP
    n = 0
    stamp = _iso(now)
    with transaction(conn):
        for device, group in pairs.items():
            token = settings.token_for(device)
            if not token:
                continue
            conn.execute(
                "INSERT INTO walkers (id, device, group_name, device_token_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(device) DO UPDATE SET group_name=excluded.group_name, "
                "device_token_hash=excluded.device_token_hash",
                (device, device, group, hash_token(token), stamp),
            )
            n += 1
    return n


def walker_for_token(conn: sqlite3.Connection, token: str) -> Walker | None:
    """Find the walker for a bearer token. Compares against every hash (no early exit)."""
    found: Walker | None = None
    for row in conn.execute("SELECT id, device, group_name, device_token_hash FROM walkers"):
        if verify_token(token, row["device_token_hash"]) and found is None:
            found = Walker(row["id"], row["device"], row["group_name"])
    return found


# -------------------------------------------------------------------- missions


def insert_mission(
    conn: sqlite3.Connection, mission: Mission, created_at: dt.datetime | None = None
) -> None:
    """Store a mission and link referenced facts (ids that exist) into ``mission_refs``."""
    with transaction(conn):
        conn.execute(
            "INSERT INTO missions (id, date, group_name, type, text, references_json, source, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mission.id,
                mission.date.isoformat(),
                mission.group,
                mission.type.value,
                mission.text,
                json.dumps(mission.references),
                mission.source.value,
                _iso(created_at),
            ),
        )
        _record_refs(conn, mission.id, mission.references)


def _record_refs(conn: sqlite3.Connection, mission_id: str, fact_ids: Sequence[str]) -> None:
    for fid in fact_ids:
        conn.execute(
            "INSERT OR IGNORE INTO mission_refs (mission_id, fact_id) "
            "SELECT ?, id FROM facts WHERE id = ?",
            (mission_id, fid),
        )


def record_mission_refs(conn: sqlite3.Connection, mission_id: str, fact_ids: Sequence[str]) -> None:
    """Link a mission to the facts it asks about (powers "Asked about again on...")."""
    with transaction(conn):
        _record_refs(conn, mission_id, fact_ids)


def _mission_from_row(row: sqlite3.Row) -> Mission:
    return Mission(
        id=row["id"],
        date=iso_to_date(row["date"]),
        group=row["group_name"],
        type=MissionType(row["type"]),
        text=row["text"],
        references=json.loads(row["references_json"]),
        source=MissionSource(row["source"]),
    )


def get_cached_mission(conn: sqlite3.Connection, group: str, date: dt.date) -> Mission | None:
    row = conn.execute(
        "SELECT * FROM missions WHERE group_name = ? AND date = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (group, date.isoformat()),
    ).fetchone()
    return _mission_from_row(row) if row else None


def latest_missions(conn: sqlite3.Connection, group: str, n: int = 5) -> list[Mission]:
    """Most recent first. The planner gets at most the last 5 (FR-M3)."""
    rows = conn.execute(
        "SELECT * FROM missions WHERE group_name = ? "
        "ORDER BY date DESC, created_at DESC, rowid DESC LIMIT ?",
        (group, max(0, n)),
    ).fetchall()
    return [_mission_from_row(r) for r in rows]


# ----------------------------------------------------------------------- walks


def insert_walk(
    conn: sqlite3.Connection,
    *,
    walk_id: str,
    group: str,
    recorded_at: dt.datetime,
    duration_s: float | None = None,
    mission_id: str | None = None,
) -> None:
    """Record a received walk (status ``received``). ``recorded_at`` must be timezone-aware."""
    if recorded_at.tzinfo is None:
        raise ValueError("recorded_at must be timezone-aware")
    conn.execute(
        "INSERT INTO walks (id, mission_id, group_name, recorded_at, duration_s, status) "
        "VALUES (?, ?, ?, ?, ?, 'received')",
        (walk_id, mission_id, group, recorded_at.isoformat(), duration_s),
    )


def update_walk_status(
    conn: sqlite3.Connection,
    walk_id: str,
    status: WalkStatus,
    reason: ReasonCode | None = None,
) -> None:
    """Set status; ``reason`` (an enum code, never text) is meaningful for ``lost``."""
    conn.execute(
        "UPDATE walks SET status = ?, reason = ? WHERE id = ?",
        (status.value, reason.value if reason else None, walk_id),
    )


def set_transcript(conn: sqlite3.Connection, walk_id: str, transcript: str) -> None:
    conn.execute("UPDATE walks SET transcript = ? WHERE id = ?", (transcript, walk_id))


@dataclass(frozen=True, slots=True)
class GroundingSummary:
    """Outcome of the grounding check, as the store needs it (codes and flags only).

    ``fact_verified`` aligns by index with ``WriterOutput.facts``; ``mission_result`` and
    ``mission_span`` are the values after check-7 coercion.
    """

    passed: bool
    fact_verified: tuple[bool, ...] = ()
    mission_result: MissionResult = MissionResult.NOT_MENTIONED
    mission_span: str | None = None


def store_journal_result(
    conn: sqlite3.Connection,
    walk_id: str,
    transcript: str,
    writer_output: WriterOutput | None,
    check_result: GroundingSummary,
    entry_text: str,
    writer: str,
    regen_count: int,
    latency: dict[str, float],
    created_at: dt.datetime | None = None,
) -> list[str]:
    """One transaction: walk row + facts (with verified flags) + entry; status becomes ``done``.

    ``writer_output`` is None for a template entry (no facts). Returns the new fact ids.
    """
    facts = writer_output.facts if writer_output is not None else []
    if len(check_result.fact_verified) != len(facts):
        raise ValueError("fact_verified length does not match facts")
    stamp = _iso(created_at)
    ids: list[str] = []
    with transaction(conn):
        cur = conn.execute(
            "UPDATE walks SET transcript = ?, mission_result = ?, mission_span = ?, writer = ?, "
            "check_passed = ?, regen_count = ?, latency_ms_json = ?, status = 'done', "
            "reason = NULL WHERE id = ?",
            (
                transcript,
                check_result.mission_result.value,
                check_result.mission_span,
                writer,
                int(check_result.passed),
                regen_count,
                json.dumps(latency, sort_keys=True),
                walk_id,
            ),
        )
        if cur.rowcount != 1:
            raise LookupError("unknown walk")
        for fact, ok in zip(facts, check_result.fact_verified, strict=True):
            fid = new_id()
            ids.append(fid)
            conn.execute(
                "INSERT INTO facts (id, walk_id, span, kind, subject, place, certainty, "
                "verified, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fid,
                    walk_id,
                    fact.span,
                    fact.kind.value,
                    fact.subject,
                    fact.place,
                    fact.certainty.value,
                    int(ok),
                    stamp,
                ),
            )
        conn.execute(
            "INSERT INTO entries (walk_id, text, created_at) VALUES (?, ?, ?)",
            (walk_id, entry_text, stamp),
        )
    return ids


def get_walk_row(conn: sqlite3.Connection, walk_id: str) -> sqlite3.Row | None:
    """Raw row. Callers must not forward ``transcript`` anywhere but the DB/graph."""
    row: sqlite3.Row | None = conn.execute(
        "SELECT * FROM walks WHERE id = ?", (walk_id,)
    ).fetchone()
    return row


def verified_facts_for_group(
    conn: sqlite3.Connection, group: str, last_n_walks: int = 7
) -> list[VerifiedFact]:
    """Verified facts from the group's last ``last_n_walks`` finished walks, newest walk first."""
    rows = conn.execute(
        "SELECT f.id, f.walk_id, w.recorded_at, w.group_name, f.kind, f.subject, f.place, "
        "f.certainty, f.span FROM facts f JOIN ("
        "  SELECT id, recorded_at, group_name FROM walks "
        "  WHERE group_name = ? AND status = 'done' "
        "  ORDER BY recorded_at DESC, rowid DESC LIMIT ?"
        ") w ON w.id = f.walk_id WHERE f.verified = 1 "
        "ORDER BY w.recorded_at DESC, f.rowid ASC",
        (group, max(0, last_n_walks)),
    ).fetchall()
    return [
        VerifiedFact(
            id=r["id"],
            walk_id=r["walk_id"],
            walk_date=iso_to_date(r["recorded_at"]),
            group=r["group_name"],
            kind=FactKind(r["kind"]),
            subject=r["subject"],
            place=r["place"],
            certainty=Certainty(r["certainty"]),
            span=r["span"],
        )
        for r in rows
    ]


# ----------------------------------------------------------------------- diary


def diary(conn: sqlite3.Connection, group: str | None = None) -> list[DiaryEntry]:
    """Finished walks, newest first, with verified facts and "asked again on" dates (FR-D2)."""
    where = "w.status = 'done'" + (" AND w.group_name = ?" if group is not None else "")
    params: tuple[str, ...] = (group,) if group is not None else ()
    walks = conn.execute(
        "SELECT w.id, w.group_name, w.recorded_at, w.mission_result, e.text AS entry, "
        "m.text AS mission_text FROM walks w JOIN entries e ON e.walk_id = w.id "
        f"LEFT JOIN missions m ON m.id = w.mission_id WHERE {where} "
        "ORDER BY w.recorded_at DESC, w.rowid DESC",
        params,
    ).fetchall()
    out: list[DiaryEntry] = []
    for w in walks:
        fact_rows = conn.execute(
            "SELECT id, kind, subject, place, certainty FROM facts "
            "WHERE walk_id = ? AND verified = 1 ORDER BY rowid",
            (w["id"],),
        ).fetchall()
        facts: list[DiaryFact] = []
        for f in fact_rows:
            again = conn.execute(
                "SELECT DISTINCT m.date FROM mission_refs r JOIN missions m ON m.id = r.mission_id "
                "WHERE r.fact_id = ? ORDER BY m.date",
                (f["id"],),
            ).fetchall()
            facts.append(
                DiaryFact(
                    subject=f["subject"],
                    kind=FactKind(f["kind"]),
                    certainty=Certainty(f["certainty"]),
                    place=f["place"],
                    asked_again_on=[iso_to_date(a["date"]) for a in again],
                )
            )
        out.append(
            DiaryEntry(
                walk_id=w["id"],
                date=iso_to_date(w["recorded_at"]),
                group=w["group_name"],
                mission_text=w["mission_text"],
                entry=w["entry"],
                facts=facts,
                mission_result=MissionResult(w["mission_result"] or "not_mentioned"),
            )
        )
    return out


def diary_summary(
    conn: sqlite3.Connection, today: dt.date, group: str | None = None
) -> DiarySummary:
    """Walks and verified facts in ``today``'s calendar month."""
    month = today.strftime("%Y-%m")
    gsql = " AND group_name = ?" if group is not None else ""
    gparams: tuple[str, ...] = (group,) if group is not None else ()
    walks = conn.execute(
        f"SELECT COUNT(*) FROM walks WHERE status = 'done' AND substr(recorded_at,1,7) = ?{gsql}",
        (month, *gparams),
    ).fetchone()[0]
    facts = conn.execute(
        "SELECT COUNT(*) FROM facts f JOIN walks w ON w.id = f.walk_id "
        "WHERE f.verified = 1 AND w.status = 'done' AND substr(w.recorded_at,1,7) = ?"
        + gsql.replace("group_name", "w.group_name"),
        (month, *gparams),
    ).fetchone()[0]
    return DiarySummary(walks_this_month=int(walks), things_noticed=int(facts))


# -------------------------------------------------------------------- feedback


def insert_feedback(
    conn: sqlite3.Connection,
    walk_id: str,
    correct: bool,
    note: str | None = None,
    created_at: dt.datetime | None = None,
) -> None:
    """One feedback row per walk; a second submission replaces the first."""
    conn.execute(
        "INSERT INTO feedback (walk_id, correct, note, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(walk_id) DO UPDATE SET correct=excluded.correct, note=excluded.note, "
        "created_at=excluded.created_at",
        (walk_id, int(correct), note, _iso(created_at)),
    )


def feedback_for_walk(conn: sqlite3.Connection, walk_id: str) -> bool | None:
    """Whether the owner marked the entry correct (None if no feedback). No note text."""
    row = conn.execute("SELECT correct FROM feedback WHERE walk_id = ?", (walk_id,)).fetchone()
    return None if row is None else bool(row["correct"])


# ------------------------------------------------------------------------ meta


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_meta(
    conn: sqlite3.Connection, key: str, value: str, updated_at: dt.datetime | None = None
) -> None:
    conn.execute(
        "INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, _iso(updated_at)),
    )


# --------------------------------------------------------------------- metrics


def insert_metric(
    conn: sqlite3.Connection,
    node: str,
    latency_ms: float,
    *,
    walk_id: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    created_at: dt.datetime | None = None,
) -> None:
    conn.execute(
        "INSERT INTO metrics (walk_id, node, latency_ms, tokens_in, tokens_out, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (walk_id, node, latency_ms, tokens_in, tokens_out, _iso(created_at)),
    )


def _nearest_rank(sorted_vals: Sequence[float], pct: float) -> float:
    rank = max(1, math.ceil(pct / 100 * len(sorted_vals)))
    return sorted_vals[rank - 1]


def latency_percentiles(conn: sqlite3.Connection) -> dict[str, LatencyStat]:
    """p50 / p95 per node (nearest-rank). Targets are filled in by the API layer."""
    by_node: dict[str, list[float]] = {}
    for row in conn.execute("SELECT node, latency_ms FROM metrics ORDER BY node, latency_ms"):
        by_node.setdefault(row["node"], []).append(float(row["latency_ms"]))
    return {
        node: LatencyStat(
            n=len(vals), p50_ms=_nearest_rank(vals, 50), p95_ms=_nearest_rank(vals, 95)
        )
        for node, vals in by_node.items()
    }


@dataclass(frozen=True, slots=True)
class StatusCounts:
    """Everything /status needs from the DB. Counts, rates and enum codes only (FR-O2)."""

    per_group: dict[str, GroupCounts] = field(default_factory=dict)
    grounding_pass_rate: float | None = None
    grounding_n: int = 0
    reason_counts: dict[ReasonCode, int] = field(default_factory=dict)
    mission_fallbacks: int = 0
    template_entries: int = 0
    feedback_pending: int = 0

    @property
    def fallback_count(self) -> int:
        return self.mission_fallbacks + self.template_entries


def status_counts(conn: sqlite3.Connection) -> StatusCounts:
    per_group: dict[str, GroupCounts] = {}
    for r in conn.execute(
        "SELECT w.group_name AS g, COUNT(DISTINCT w.id) AS walks, "
        "COUNT(f.id) AS facts FROM walks w "
        "LEFT JOIN facts f ON f.walk_id = w.id AND f.verified = 1 "
        "WHERE w.status = 'done' GROUP BY w.group_name"
    ):
        per_group[r["g"]] = GroupCounts(walks=r["walks"], facts=r["facts"])
    n, passed = conn.execute(
        "SELECT COUNT(check_passed), COALESCE(SUM(check_passed), 0) FROM walks"
    ).fetchone()
    reasons: dict[ReasonCode, int] = {}
    for r in conn.execute(
        "SELECT reason, COUNT(*) AS c FROM walks WHERE reason IS NOT NULL GROUP BY reason"
    ):
        try:
            reasons[ReasonCode(r["reason"])] = r["c"]
        except ValueError:
            continue
    return StatusCounts(
        per_group=per_group,
        grounding_pass_rate=(passed / n) if n else None,
        grounding_n=int(n),
        reason_counts=reasons,
        mission_fallbacks=conn.execute(
            "SELECT COUNT(*) FROM missions WHERE source = 'fallback'"
        ).fetchone()[0],
        template_entries=conn.execute(
            "SELECT COUNT(*) FROM walks WHERE writer = 'template'"
        ).fetchone()[0],
        feedback_pending=conn.execute(
            "SELECT COUNT(*) FROM walks w WHERE w.status = 'done' "
            "AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.walk_id = w.id)"
        ).fetchone()[0],
    )


def raw_audio_on_disk(tmp_dir: Path) -> int:
    """Count of files in the raw-audio temp dir (must be 0 after each walk, FR-J6)."""
    if not tmp_dir.is_dir():
        return 0
    return sum(1 for p in tmp_dir.rglob("*") if p.is_file())


# ---------------------------------------------------------------------- forget


def forget(settings: Settings) -> list[str]:
    """FR-O5: delete the DB (+ -wal/-shm), temp audio, audio cache, mission audio, model-server
    logs and any checkpoint dir. Files are only deleted, never opened. Returns removed paths.
    """
    db = settings.db_path
    removed: list[str] = []

    def _unsafe(p: Path) -> bool:
        r = p.resolve()
        return r == Path(r.anchor) or r == Path.home().resolve()

    for f in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if f.is_file() or f.is_symlink():
            f.unlink()
            removed.append(str(f))
    for d in (
        settings.tmp_dir,
        settings.audio_dir,
        settings.audio_cache_dir,
        settings.checkpoint_dir,
    ):
        if d.exists() and not _unsafe(d):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    logs = settings.log_dir
    if logs.is_dir() and not _unsafe(logs):
        for p in sorted(logs.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed.append(str(p))
            elif p.is_dir():
                p.rmdir()
    return removed
