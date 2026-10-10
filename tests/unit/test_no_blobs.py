"""§9.4: no BLOBs, ever (typeof scan over a full synthetic walk)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from hub import db
from hub.schemas import ReasonCode, WalkStatus
from tests.unit._dbfix import GROUP, IST, add_walk, mission


def test_no_blob_columns_or_values(tmp_path: Path) -> None:
    conn = db.init_db(tmp_path / "sair.db")
    db.insert_mission(conn, mission("m0", 1))
    ids = add_walk(conn, "w1", 1, mission_id="m0")
    db.insert_mission(conn, mission("m1", 2, refs=[ids[0]]))
    db.insert_walk(
        conn, walk_id="w2", group=GROUP, recorded_at=dt.datetime(2026, 10, 3, tzinfo=IST)
    )
    db.update_walk_status(conn, "w2", WalkStatus.LOST, ReasonCode.AUDIO_SILENT)
    db.insert_feedback(conn, "w1", True, "ok")
    db.set_meta(conn, "lexicon_hash", "abc")
    db.insert_metric(conn, "transcribe", 12.5, walk_id="w1", tokens_in=3, tokens_out=4)

    ddl = " ".join(r[0] or "" for r in conn.execute("SELECT sql FROM sqlite_master"))
    assert "BLOB" not in ddl.upper()

    seen_rows = 0
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for t in tables:
        for col in conn.execute(f"PRAGMA table_info({t})").fetchall():
            assert "BLOB" not in (col["type"] or "").upper(), (t, col["name"])
            types = {r[0] for r in conn.execute(f"SELECT typeof({col['name']}) FROM {t}")}
            assert "blob" not in types, (t, col["name"])
        seen_rows += conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert seen_rows > 10
    conn.close()
