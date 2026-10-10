"""Shared synthetic builders for DB tests (generic places only)."""

from __future__ import annotations

import datetime as dt
import sqlite3

from hub import db
from hub.schemas import (
    Certainty,
    Fact,
    FactKind,
    Mission,
    MissionResult,
    MissionSource,
    MissionType,
    WriterOutput,
)

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
GROUP = "Mom and Dad"


def output(n_facts: int = 2) -> WriterOutput:
    facts = [
        Fact(
            span=f"a dog near the lane {i}",
            kind=FactKind.ANIMAL,
            subject="dog",
            certainty=Certainty.SURE if i % 2 == 0 else Certainty.UNSURE,
            place="the lane",
        )
        for i in range(n_facts)
    ]
    return WriterOutput(
        facts=facts, mission_result=MissionResult.DONE, mission_span="we looked", entry="x"
    )


def add_walk(
    conn: sqlite3.Connection,
    walk_id: str,
    day: int,
    group: str = GROUP,
    verified: tuple[bool, ...] = (True, False),
    mission_id: str | None = None,
    writer: str = "llama",
    passed: bool = True,
) -> list[str]:
    db.insert_walk(
        conn,
        walk_id=walk_id,
        group=group,
        recorded_at=dt.datetime(2026, 10, day, 18, 5, tzinfo=IST),
        duration_s=30.0,
        mission_id=mission_id,
    )
    out = output(len(verified))
    return db.store_journal_result(
        conn,
        walk_id,
        "we saw a dog by the lane",
        out,
        db.GroundingSummary(
            passed=passed,
            fact_verified=verified,
            mission_result=MissionResult.DONE,
            mission_span="we looked",
        ),
        "Mom and Dad saw a dog.",
        writer,
        0,
        {"transcribe": 800.0, "write_entry": 2500.0},
    )


def mission(
    mid: str,
    day: int,
    text: str = "Notice the sounds.",
    refs: list[str] | None = None,
    source: MissionSource = MissionSource.MODEL,
    group: str = GROUP,
) -> Mission:
    return Mission(
        id=mid,
        date=dt.date(2026, 10, day),
        group=group,
        type=MissionType.SENSE,
        text=text,
        references=refs or [],
        source=source,
    )
