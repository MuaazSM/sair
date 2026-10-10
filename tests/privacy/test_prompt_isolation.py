"""FR-M3, FR-J3 (PRD §13.6): a canary phrase in ``entries.text`` never reaches a planner or
writer prompt. Entries are prose; the planner gets only verified facts + the last 5 mission
texts, the writer only group, today's mission and the new transcript.

Controls prove the test can see: the canary placed in a mission text (allowed planner input)
does appear, and a marker placed in a verified fact does appear.
"""

from __future__ import annotations

import datetime as dt
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest

from hub import db
from hub.prompts.builders import (
    ChatMessage,
    MissionText,
    Transcript,
    WalkerGroup,
    build_planner_prompt,
    build_writer_prompt,
)
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

try:  # the graphs are optional here so this file also runs before they land
    from hub.graphs import journal as J
    from hub.graphs import mission as M
    from tests.unit import _graphfix as GF

    HAVE_GRAPHS = True
except ImportError:  # pragma: no cover
    HAVE_GRAPHS = False

CANARY = "zebra-ultramarine-4471"
FACT_MARKER = "heron"  # appears in a verified fact: must reach the planner
GROUP = "Mom and Dad"
TODAY = dt.date(2026, 10, 10)
TRANSCRIPTS = (
    "We walked past the school. A heron was standing in the canal near the bridge.",
    "The neem tree near the gate had new leaves. It was windy near the park.",
    "Maybe some kind of owl near the temple, I think.",
    "We counted four dogs sleeping near the shop.",
)
SPANS = (
    ("A heron was standing in the canal near the bridge", "heron", "near the bridge"),
    ("The neem tree near the gate had new leaves", "neem tree", "near the gate"),
    ("Maybe some kind of owl near the temple", "owl", "near the temple"),
    ("four dogs sleeping near the shop", "dogs", "near the shop"),
)
NEW_TRANSCRIPT = "We heard a koel near the temple and the puddles were still there."
MISSION_TEXTS = (
    "Listen for one bird you can hear from the path.",
    "Count the trees with flowers on your way.",
    "Find the heron near the bridge again and look at what it does.",
)


def _text(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(m.content for m in messages)


def _uid() -> str:
    return uuid.uuid4().hex


def _seed_db(path: Path, *, mission_texts: Sequence[str] = MISSION_TEXTS) -> list[str]:
    """Done walks whose ENTRY (and feedback note) carry the canary; transcripts/facts don't.

    Returns the mission ids, oldest first.
    """
    conn = db.init_db(path)
    mission_ids: list[str] = []
    try:
        for i, text in enumerate(mission_texts):
            mid = _uid()
            mission_ids.append(mid)
            db.insert_mission(conn, Mission(id=mid, date=TODAY - dt.timedelta(days=10 - i),
                                            group=GROUP, type=MissionType.SENSE, text=text,
                                            source=MissionSource.MODEL))  # fmt: skip
        for i, transcript in enumerate(TRANSCRIPTS):
            wid = _uid()
            day = dt.datetime(2026, 10, 2 + i, 13, 0, tzinfo=dt.UTC)
            db.insert_walk(conn, walk_id=wid, group=GROUP, recorded_at=day,
                           mission_id=mission_ids[min(i, len(mission_ids) - 1)])  # fmt: skip
            span, subject, place = SPANS[i]
            assert span.lower() in transcript.lower()
            out = WriterOutput(
                facts=[Fact(span=span, kind=FactKind.SIGHT, subject=subject, place=place,
                            certainty=Certainty.UNSURE if i == 2 else Certainty.SURE)],
                mission_result=MissionResult.NOT_MENTIONED,
                entry=f"{GROUP} saw {subject}. {CANARY} was in the prose.",
            )  # fmt: skip
            db.store_journal_result(
                conn, wid, transcript, out,
                db.GroundingSummary(passed=True, fact_verified=(True,)),
                entry_text=f"{GROUP} went for a walk and noticed {subject}. {CANARY}.",
                writer="gemma", regen_count=0, latency={"write_entry": 1.0},
            )  # fmt: skip
            db.insert_feedback(conn, wid, correct=False, note=f"wrong, {CANARY}")
            conn.commit()
        rows = conn.execute("SELECT count(*) FROM entries WHERE text LIKE ?", (f"%{CANARY}%",))
        assert rows.fetchone()[0] == len(TRANSCRIPTS)  # the canary really is in entries.text
    finally:
        conn.close()
    return mission_ids


def _production_planner_prompt(path: Path) -> list[ChatMessage]:
    """Exactly what the mission graph's recall + plan nodes feed the builder."""
    conn = db.connect(path)
    try:
        facts = db.verified_facts_for_group(conn, GROUP, last_n_walks=7)
        recent = [m.text for m in db.latest_missions(conn, GROUP, n=5)]
    finally:
        conn.close()
    assert facts, "seed produced no verified facts"
    return build_planner_prompt(
        facts, [MissionText(t) for t in recent], group=WalkerGroup(GROUP), today=TODAY
    )


def _production_writer_prompt(path: Path) -> list[ChatMessage]:
    conn = db.connect(path)
    try:
        latest = db.latest_missions(conn, GROUP, n=1)
    finally:
        conn.close()
    return build_writer_prompt(
        WalkerGroup(GROUP), MissionText(latest[0].text), Transcript(NEW_TRANSCRIPT)
    )


# ------------------------------------------------------------------ builders + DB


def test_canary_in_entries_never_reaches_planner(tmp_path: Path) -> None:
    path = tmp_path / "sair.db"
    _seed_db(path)
    text = _text(_production_planner_prompt(path))
    assert CANARY not in text
    assert FACT_MARKER in text  # control: verified facts do reach the planner


def test_canary_in_entries_never_reaches_writer(tmp_path: Path) -> None:
    path = tmp_path / "sair.db"
    _seed_db(path)
    text = _text(_production_writer_prompt(path))
    assert CANARY not in text
    assert NEW_TRANSCRIPT in text
    for transcript in TRANSCRIPTS:  # no previous transcript either (FR-J3)
        assert transcript not in text


def test_control_canary_in_mission_text_does_reach_planner(tmp_path: Path) -> None:
    """FR-M3 allows the last 5 mission texts; proves the assertion above can fail."""
    path = tmp_path / "sair.db"
    _seed_db(path, mission_texts=(*MISSION_TEXTS, f"Look for the {CANARY} on the path."))
    assert CANARY in _text(_production_planner_prompt(path))


# ------------------------------------------------------------------ real graphs


@pytest.mark.skipif(not HAVE_GRAPHS, reason="hub.graphs not available yet")
def test_mission_graph_planner_messages_exclude_entries(tmp_path: Path) -> None:
    stub = GF.StubOllama()
    deps = GF.mission_deps(tmp_path, ollama=stub)
    deps.settings.db_path.unlink()  # replace the empty DB with the seeded one
    _seed_db(deps.settings.db_path)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None
    assert stub.planner_calls, "planner was not called (seed path?)"
    for messages in stub.planner_calls:
        text = _text(messages)
        assert CANARY not in text
        assert FACT_MARKER in text


@pytest.mark.skipif(not HAVE_GRAPHS, reason="hub.graphs not available yet")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_journal_graph_writer_messages_exclude_entries(tmp_path: Path) -> None:
    stub = GF.StubOllama()
    deps = GF.journal_deps(tmp_path, ollama=stub)
    deps.settings.db_path.unlink()
    mission_ids = _seed_db(deps.settings.db_path)
    wid = _uid()
    GF.add_received_walk(deps, wid, mission_id=mission_ids[-1])
    raw = GF.stage_audio(deps, "en_20s.webm", wid)
    J.run_journal(wid, raw, GROUP, mission_ids[-1], deps)
    assert stub.writer_calls, "writer was not called"
    for messages in stub.writer_calls:
        text = _text(messages)
        assert CANARY not in text
        assert MISSION_TEXTS[-1] in text  # control: today's mission does reach the writer
        for transcript in TRANSCRIPTS:
            assert transcript not in text
