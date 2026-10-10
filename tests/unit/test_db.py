"""PRD §9.4 storage: schema, constraints, repository behaviour."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from hub import db
from hub.schemas import (
    LatencyStat,
    MissionSource,
    ReasonCode,
    WalkStatus,
)
from hub.settings import Settings
from tests.unit._dbfix import GROUP, IST, add_walk, mission, output

PRD_COLUMNS: dict[str, list[str]] = {
    "walkers": ["id", "device", "group_name", "device_token_hash", "created_at"],
    "missions": [
        "id",
        "date",
        "group_name",
        "type",
        "text",
        "references_json",
        "source",
        "created_at",
    ],
    "walks": [
        "id",
        "mission_id",
        "group_name",
        "recorded_at",
        "duration_s",
        "transcript",
        "mission_result",
        "mission_span",
        "writer",
        "check_passed",
        "regen_count",
        "status",
        "latency_ms_json",
    ],
    "facts": [
        "id",
        "walk_id",
        "span",
        "kind",
        "subject",
        "place",
        "certainty",
        "verified",
        "created_at",
    ],
    "entries": ["walk_id", "text", "created_at"],
    "mission_refs": ["mission_id", "fact_id"],
    "feedback": ["walk_id", "correct", "note", "created_at"],
    "meta": ["key", "value", "updated_at"],
    "metrics": ["walk_id", "node", "latency_ms", "tokens_in", "tokens_out", "created_at"],
}
EXTRA_COLUMNS = {"walks": {"reason"}}  # proposed PRD diff: reason code for lost walks


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = db.init_db(tmp_path / "sair.db")
    yield c  # type: ignore[misc]
    c.close()


def cols(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_init_idempotent_and_versioned(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "sair.db"
    c1 = db.init_db(p)
    assert db.schema_version(c1) == len(db.MIGRATIONS)
    db.set_meta(c1, "k", "v")
    c1.close()
    c2 = db.init_db(p)
    assert db.schema_version(c2) == len(db.MIGRATIONS)
    assert db.get_meta(c2, "k") == "v"
    c2.close()


def test_wal_and_pragmas(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0


def test_every_prd_table_and_column_present(conn: sqlite3.Connection) -> None:
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert tables == set(PRD_COLUMNS)
    for table, expected in PRD_COLUMNS.items():
        actual = cols(conn, table)
        assert set(expected) <= set(actual), table
        assert set(actual) - set(expected) == EXTRA_COLUMNS.get(table, set()), table


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO missions VALUES ('m','2026-10-01','g','bogus','t','[]','model','x')",
        "INSERT INTO missions VALUES ('m','2026-10-01','g','sense','t','[]','bogus','x')",
        "INSERT INTO walks (id, group_name, recorded_at, status) VALUES ('w','g','x','bogus')",
        "INSERT INTO walks (id, group_name, recorded_at, status, mission_result) "
        "VALUES ('w','g','x','done','bogus')",
        "INSERT INTO walks (id, group_name, recorded_at, status, writer) "
        "VALUES ('w','g','x','done','bogus')",
    ],
)
def test_enum_checks_reject_bad_values(conn: sqlite3.Connection, sql: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)


def test_fact_enum_checks(conn: sqlite3.Connection) -> None:
    add_walk(conn, "w1", 1, verified=(True,))
    base = "INSERT INTO facts VALUES ('f9','w1','s','{kind}','subj',NULL,'{cert}',{ver},'x')"
    for kind, cert, ver in [("bogus", "sure", 1), ("sight", "bogus", 1), ("sight", "sure", 2)]:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(base.format(kind=kind, cert=cert, ver=ver))


def test_foreign_keys_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO entries VALUES ('nope','t','x')")


def test_store_journal_result_happy_path(conn: sqlite3.Connection) -> None:
    ids = add_walk(conn, "w1", 1, verified=(True, False))
    assert len(ids) == 2
    row = db.get_walk_row(conn, "w1")
    assert row is not None
    assert row["status"] == "done" and row["check_passed"] == 1 and row["writer"] == "llama"
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 1
    flags = [r[0] for r in conn.execute("SELECT verified FROM facts ORDER BY rowid")]
    assert flags == [1, 0]


def test_store_journal_result_rolls_back_on_error(conn: sqlite3.Connection) -> None:
    add_walk(conn, "w1", 1)
    before = (
        conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0],
    )
    # Second store for the same walk: facts insert fine, the entry insert hits the PK -> rollback.
    with pytest.raises(sqlite3.IntegrityError):
        db.store_journal_result(
            conn,
            "w1",
            "CHANGED",
            output(3),
            db.GroundingSummary(passed=False, fact_verified=(True, True, True)),
            "other",
            "gemma",
            1,
            {},
        )
    after = (
        conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0],
    )
    assert after == before
    row = db.get_walk_row(conn, "w1")
    assert row is not None and row["transcript"] != "CHANGED" and row["writer"] == "llama"
    assert not conn.in_transaction


def test_store_journal_result_validates_inputs(conn: sqlite3.Connection) -> None:
    db.insert_walk(
        conn, walk_id="w1", group=GROUP, recorded_at=dt.datetime(2026, 10, 1, tzinfo=IST)
    )
    with pytest.raises(ValueError):
        db.store_journal_result(
            conn,
            "w1",
            "t",
            output(2),
            db.GroundingSummary(passed=True, fact_verified=(True,)),
            "e",
            "gemma",
            0,
            {},
        )
    with pytest.raises(LookupError):
        db.store_journal_result(
            conn, "missing", "t", None, db.GroundingSummary(passed=True), "e", "template", 0, {}
        )
    with pytest.raises(ValueError):
        db.insert_walk(conn, walk_id="w2", group=GROUP, recorded_at=dt.datetime(2026, 10, 1))


def test_template_entry_has_no_facts(conn: sqlite3.Connection) -> None:
    db.insert_walk(
        conn, walk_id="w1", group=GROUP, recorded_at=dt.datetime(2026, 10, 1, tzinfo=IST)
    )
    assert (
        db.store_journal_result(
            conn,
            "w1",
            "t",
            None,
            db.GroundingSummary(passed=True),
            "Mom and Dad went for a walk.",
            "template",
            1,
            {},
        )
        == []
    )
    assert db.status_counts(conn).template_entries == 1


def test_verified_facts_for_group_window(conn: sqlite3.Connection) -> None:
    for day in range(1, 10):  # 9 walks, days 1..9
        add_walk(conn, f"w{day}", day, verified=(True, False))
    add_walk(conn, "other", 9, group="Aunt", verified=(True,))
    db.insert_walk(
        conn, walk_id="pending", group=GROUP, recorded_at=dt.datetime(2026, 10, 20, tzinfo=IST)
    )  # received, not done
    facts = db.verified_facts_for_group(conn, GROUP)
    assert {f.walk_id for f in facts} == {f"w{d}" for d in range(3, 10)}  # oldest two excluded
    assert len(facts) == 7  # one verified fact per walk; unverified excluded
    assert all(f.group == GROUP for f in facts)
    assert facts[0].walk_id == "w9" and facts[0].walk_date == dt.date(2026, 10, 9)
    assert len(db.verified_facts_for_group(conn, GROUP, last_n_walks=2)) == 2


def test_latest_missions_and_cache(conn: sqlite3.Connection) -> None:
    for d in range(1, 9):
        db.insert_mission(conn, mission(f"m{d}", d, text=f"Mission {d}"))
    db.insert_mission(conn, mission("o", 8, group="Aunt"))
    latest = db.latest_missions(conn, GROUP)
    assert [m.id for m in latest] == ["m8", "m7", "m6", "m5", "m4"]
    assert len(db.latest_missions(conn, GROUP, n=3)) == 3
    cached = db.get_cached_mission(conn, GROUP, dt.date(2026, 10, 8))
    assert cached is not None and cached.id == "m8"
    assert db.get_cached_mission(conn, GROUP, dt.date(2026, 11, 1)) is None
    db.insert_mission(conn, mission("m8b", 8, source=MissionSource.FALLBACK))
    newest = db.get_cached_mission(conn, GROUP, dt.date(2026, 10, 8))
    assert newest is not None and newest.id == "m8b" and newest.source is MissionSource.FALLBACK


def test_diary_asked_again_dates(conn: sqlite3.Connection) -> None:
    db.insert_mission(conn, mission("m0", 1))
    ids = add_walk(conn, "w1", 1, verified=(True, False), mission_id="m0")
    db.insert_mission(conn, mission("m5", 5, refs=[ids[0]]))
    db.insert_mission(conn, mission("m7", 7, refs=[ids[0], "unknown-fact"]))
    db.record_mission_refs(conn, "m7", [ids[0]])  # idempotent
    add_walk(conn, "w2", 3, group="Aunt", verified=(True,))
    entries = db.diary(conn, GROUP)
    assert len(entries) == 1
    e = entries[0]
    assert e.walk_id == "w1" and e.date == dt.date(2026, 10, 1) and type(e.date) is dt.date
    assert e.mission_text == "Notice the sounds."
    assert len(e.facts) == 1  # unverified fact hidden
    assert e.facts[0].asked_again_on == [dt.date(2026, 10, 5), dt.date(2026, 10, 7)]
    assert {x.walk_id for x in db.diary(conn, None)} == {"w1", "w2"}
    s = db.diary_summary(conn, dt.date(2026, 10, 20), GROUP)
    assert (s.walks_this_month, s.things_noticed) == (1, 1)
    assert db.diary_summary(conn, dt.date(2026, 9, 20)).walks_this_month == 0


def test_date_helpers_return_date_not_datetime() -> None:
    d = db.iso_to_date("2026-10-01T18:05:00+05:30")
    assert type(d) is dt.date and d == dt.date(2026, 10, 1)
    d2 = db.iso_to_date("2026-10-01")
    assert type(d2) is dt.date


def test_metrics_percentiles(conn: sqlite3.Connection) -> None:
    for i in range(1, 101):
        db.insert_metric(conn, "write_entry", float(i), walk_id="w", tokens_in=1, tokens_out=2)
    db.insert_metric(conn, "transcribe", 10.0)
    stats = db.latency_percentiles(conn)
    assert stats["write_entry"] == LatencyStat(n=100, p50_ms=50.0, p95_ms=95.0)
    assert stats["transcribe"].n == 1 and stats["transcribe"].p95_ms == 10.0


def test_status_counts_have_no_text(conn: sqlite3.Connection) -> None:
    db.insert_mission(conn, mission("mf", 1, source=MissionSource.FALLBACK))
    add_walk(conn, "w1", 1, verified=(True, False), passed=True)
    add_walk(conn, "w2", 2, verified=(True,), passed=False)
    db.insert_walk(
        conn, walk_id="w3", group=GROUP, recorded_at=dt.datetime(2026, 10, 3, tzinfo=IST)
    )
    db.update_walk_status(conn, "w3", WalkStatus.LOST, ReasonCode.AUDIO_SILENT)
    db.insert_feedback(conn, "w1", True, "note text")
    c = db.status_counts(conn)
    assert c.per_group[GROUP].walks == 2 and c.per_group[GROUP].facts == 2
    assert c.grounding_n == 2 and c.grounding_pass_rate == 0.5
    assert c.reason_counts == {ReasonCode.AUDIO_SILENT: 1}
    assert c.mission_fallbacks == 1 and c.fallback_count == 1
    assert c.feedback_pending == 1
    assert db.feedback_for_walk(conn, "w1") is True and db.feedback_for_walk(conn, "w2") is None
    blob = repr(c)
    for secret in ("we saw a dog", "dog", "Notice the sounds", "note text"):
        assert secret not in blob


def test_meta_roundtrip(conn: sqlite3.Connection) -> None:
    assert db.get_meta(conn, "last_canary_result") is None
    db.set_meta(conn, "last_canary_result", "passed")
    db.set_meta(conn, "last_canary_result", "failed")
    assert db.get_meta(conn, "last_canary_result") == "failed"


def test_tokens_hashed_and_verified(conn: sqlite3.Connection) -> None:
    s = Settings(_env_file=None, sair_token_mom="mom-tok", sair_token_owner="own-tok")  # type: ignore[call-arg]
    assert db.upsert_walkers_from_settings(conn, s) == 2  # dad has no token: skipped
    assert db.upsert_walkers_from_settings(conn, s) == 2  # idempotent
    stored = [r[0] for r in conn.execute("SELECT device_token_hash FROM walkers")]
    assert "mom-tok" not in stored and all(len(h) == 64 for h in stored)
    w = db.walker_for_token(conn, "mom-tok")
    assert w is not None and w.group_name == GROUP and not w.is_owner
    o = db.walker_for_token(conn, "own-tok")
    assert o is not None and o.is_owner
    assert db.walker_for_token(conn, "wrong") is None
    assert db.walker_for_token(conn, "") is None
    assert not db.verify_token("", db.hash_token(""))
    assert db.verify_token("a", db.hash_token("a"))


def test_mission_audio_path_derived() -> None:
    assert db.mission_audio_path(Path("/a"), "m1") == Path("/a/m1.mp3")
