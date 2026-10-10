"""§T FR-J8: walk status flow received -> done | lost, incl. regen/template/writer errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from hub.graphs import journal as J
from hub.models.errors import Permanent
from hub.schemas import ReasonCode, WalkStatus
from tests.unit._graphfix import (
    BAD_ENTRY,
    GOOD_ENTRY,
    GROUP,
    FakeClock,
    RecNotifier,
    StubOllama,
    add_received_walk,
    files_in,
    journal_deps,
    stage_audio,
    transient,
    walk_row,
)

pytestmark = pytest.mark.integration


def _entry(deps: J.JournalDeps, walk_id: str = "w1") -> str | None:
    conn = deps.connect()
    try:
        row = conn.execute("SELECT text FROM entries WHERE walk_id = ?", (walk_id,)).fetchone()
    finally:
        conn.close()
    return None if row is None else str(row["text"])


def _verified(deps: J.JournalDeps, walk_id: str = "w1") -> list[int]:
    conn = deps.connect()
    try:
        rows = conn.execute("SELECT verified FROM facts WHERE walk_id = ?", (walk_id,)).fetchall()
    finally:
        conn.close()
    return [int(r["verified"]) for r in rows]


def _go(deps: J.JournalDeps, fixture: str = "en_20s.webm") -> J.JournalOutcome:
    add_received_walk(deps, "w1")
    assert walk_row(deps, "w1")["status"] == "received"  # what POST /walk acknowledges
    raw = stage_audio(deps, fixture, "w1")
    return J.run_journal("w1", raw, GROUP, None, deps)


def test_received_to_done_with_entry(tmp_path: Path) -> None:
    n = RecNotifier()
    deps = journal_deps(tmp_path, notifier=n)
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.writer == "gemma" and out.check_passed
    row = walk_row(deps, "w1")
    assert (row["status"], row["writer"], row["check_passed"], row["regen_count"]) == (
        "done", "gemma", 1, 0,
    )  # fmt: skip
    assert row["reason"] is None
    assert _entry(deps) == "Mom and Dad heard a koel calling near the temple."
    assert _verified(deps) == [1]
    assert len(n.walks) == 1 and n.walks[0][:2] == (GROUP, 1)


def test_regen_path(tmp_path: Path) -> None:
    ollama = StubOllama(writer=[BAD_ENTRY, GOOD_ENTRY])
    deps = journal_deps(tmp_path, ollama=ollama)
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.regen_count == 1 and out.check_passed
    row = walk_row(deps, "w1")
    assert (row["writer"], row["check_passed"], row["regen_count"]) == ("gemma", 1, 1)
    assert len(ollama.writer_calls) == 2


def test_template_path(tmp_path: Path) -> None:
    ollama = StubOllama(writer=[BAD_ENTRY, BAD_ENTRY])
    deps = journal_deps(tmp_path, ollama=ollama)
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.writer == "template"
    assert out.check_passed is False and out.facts == 1
    row = walk_row(deps, "w1")
    assert (row["writer"], row["check_passed"], row["regen_count"]) == ("template", 0, 1)
    assert _entry(deps) == "Mom and Dad went for a walk. They noticed koel."
    assert _verified(deps) == [1]  # verified facts survive into memory


def test_lost_path(tmp_path: Path) -> None:
    n = RecNotifier()
    deps = journal_deps(tmp_path, notifier=n)
    out = _go(deps, "silent_5s.webm")
    assert out.status is WalkStatus.LOST and out.reason is ReasonCode.AUDIO_SILENT
    row = walk_row(deps, "w1")
    assert (row["status"], row["reason"]) == ("lost", "audio_silent")
    assert _entry(deps) is None and n.walks == []


def test_writer_transient_retried_from_stored_transcript(tmp_path: Path) -> None:
    clock = FakeClock()
    ollama = StubOllama(writer=[transient(), GOOD_ENTRY])
    deps = journal_deps(tmp_path, ollama=ollama, clock=clock)
    out = _go(deps)
    assert out.pending and out.reason is ReasonCode.WRITER_TRANSIENT
    assert files_in(deps.settings.tmp_dir) == []  # audio already gone after transcription
    row = walk_row(deps, "w1")
    assert (row["status"], row["reason"]) == ("received", "writer_transient")
    assert row["transcript"]  # FR-J7: the evidence is stored
    res = J.retry_pending(clock.advance(31), deps)
    assert [r.status for r in res] == [WalkStatus.DONE]
    assert ollama.transcribe_calls == 1  # the writer is retried, not transcription
    assert walk_row(deps, "w1")["writer"] == "gemma"


def test_writer_transient_window_expiry_stores_template(tmp_path: Path) -> None:
    clock = FakeClock()
    n = RecNotifier()
    deps = journal_deps(tmp_path, ollama=StubOllama(writer=[transient()]), clock=clock,
                        notifier=n)  # fmt: skip
    _go(deps)
    res = J.retry_pending(clock.advance(601), deps)
    assert [r.status for r in res] == [WalkStatus.DONE]
    row = walk_row(deps, "w1")
    assert (row["status"], row["reason"], row["writer"]) == (
        "done", "writer_transient", "template",
    )  # fmt: skip
    assert _entry(deps) == "Mom and Dad went for their evening walk."
    assert len(n.walks) == 1


def test_writer_permanent_goes_to_template(tmp_path: Path) -> None:
    perm = Permanent(ReasonCode.WRITER_PERMANENT, "ollama")
    deps = journal_deps(tmp_path, ollama=StubOllama(writer=[perm]))
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.writer == "template"
    assert out.reason is ReasonCode.WRITER_PERMANENT
    assert _entry(deps) == "Mom and Dad went for their evening walk."


def test_unexpected_writer_error_goes_to_template_not_lost(tmp_path: Path) -> None:
    """FR-J5/FR-J7: a writer bug after the transcript is stored is a permanent writer failure."""
    deps = journal_deps(tmp_path, ollama=StubOllama(writer=[ValueError("boom")]))
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.writer == "template"
    assert out.reason is ReasonCode.WRITER_PERMANENT
    row = walk_row(deps, "w1")
    assert (row["status"], row["reason"], row["writer"]) == (
        "done", "writer_permanent", "template",
    )  # fmt: skip
    assert row["transcript"]
    assert _entry(deps) == "Mom and Dad went for their evening walk."
    assert files_in(deps.settings.tmp_dir) == []


def test_unexpected_regen_error_keeps_first_checks_verified_facts(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path, ollama=StubOllama(writer=[BAD_ENTRY, ValueError("boom")]))
    out = _go(deps)
    assert out.status is WalkStatus.DONE and out.writer == "template"
    assert out.reason is ReasonCode.WRITER_PERMANENT and out.facts == 1
    assert _entry(deps) == "Mom and Dad went for a walk. They noticed koel."
    assert _verified(deps) == [1]
