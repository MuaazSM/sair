"""§T FR-J6: raw audio lifetime (delete timing, transient retry, window expiry, sweep)."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from hub.graphs import journal as J
from hub.graphs import mission as M
from hub.graphs.common import work_dir_for
from hub.models.errors import Permanent
from hub.prompts.builders import ChatMessage
from hub.schemas import ReasonCode, WalkStatus
from tests.unit._graphfix import (
    GOOD_ENTRY,
    GROUP,
    FakeClock,
    StubOllama,
    add_received_walk,
    files_in,
    journal_deps,
    mission_deps,
    stage_audio,
    transient,
    walk_row,
)

pytestmark = pytest.mark.integration


def _run(deps: J.JournalDeps, fixture: str, walk_id: str = "w1") -> tuple[J.JournalOutcome, Path]:
    add_received_walk(deps, walk_id)
    raw = stage_audio(deps, fixture, walk_id)
    return J.run_journal(walk_id, raw, GROUP, None, deps), raw


# ------------------------------------------------------------------ success


@pytest.mark.parametrize("fixture", ["en_20s.webm", "en_20s.m4a", "hinglish_20s.webm"])
def test_success_deletes_raw_and_wav_before_writer(tmp_path: Path, fixture: str) -> None:
    seen: list[list[Path]] = []
    holder: dict[str, J.JournalDeps] = {}

    def writer(messages: Sequence[ChatMessage]) -> str:
        seen.append(files_in(holder["deps"].settings.tmp_dir))
        return GOOD_ENTRY

    deps = journal_deps(tmp_path, ollama=StubOllama(writer=[writer]))
    holder["deps"] = deps
    out, raw = _run(deps, fixture)
    assert seen == [[]], "temp dir must be empty when the writer runs"
    assert not raw.exists() and files_in(deps.settings.tmp_dir) == []
    assert out.status is WalkStatus.DONE


# ------------------------------------------------------------------ permanent


@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("silent_5s.webm", ReasonCode.AUDIO_SILENT),
        ("silent_5s.m4a", ReasonCode.AUDIO_SILENT),
        ("en_95s.webm", ReasonCode.AUDIO_TOO_LONG),
        ("short_0_5s.webm", ReasonCode.AUDIO_TOO_SHORT),
    ],
)
def test_permanent_audio_deleted_immediately(
    tmp_path: Path, fixture: str, code: ReasonCode
) -> None:
    ollama = StubOllama()
    deps = journal_deps(tmp_path, ollama=ollama)
    out, raw = _run(deps, fixture)
    assert out.status is WalkStatus.LOST and out.reason is code
    assert not raw.exists() and files_in(deps.settings.tmp_dir) == []
    assert ollama.transcribe_calls == 0  # the gate runs before Gemma (D-009)
    row = walk_row(deps, "w1")
    assert row["status"] == "lost" and row["reason"] == code.value


def test_undecodable_bytes_deleted(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    add_received_walk(deps, "w1")
    raw = deps.settings.tmp_dir / "w1.webm"
    raw.write_bytes(os.urandom(4096))
    out = J.run_journal("w1", raw, GROUP, None, deps)
    assert out.status is WalkStatus.LOST and out.reason is ReasonCode.AUDIO_UNDECODABLE
    assert files_in(deps.settings.tmp_dir) == []


def test_permanent_model_error_deleted(tmp_path: Path) -> None:
    perm = Permanent(ReasonCode.TRANSCRIBE_PERMANENT, "ollama")
    deps = journal_deps(tmp_path, ollama=StubOllama(transcripts=[perm]))
    out, raw = _run(deps, "en_20s.webm")
    assert out.status is WalkStatus.LOST and out.reason is ReasonCode.TRANSCRIBE_PERMANENT
    assert files_in(deps.settings.tmp_dir) == []


# ------------------------------------------------------------------ transient


def test_transient_kept_then_retried_then_deleted(tmp_path: Path) -> None:
    clock = FakeClock()
    ollama = StubOllama(transcripts=[transient(), "We walked to the park."])
    ollama.writer = [GOOD_ENTRY]
    deps = journal_deps(tmp_path, ollama=ollama, clock=clock)
    out, raw = _run(deps, "en_20s.webm")
    assert out.status is WalkStatus.RECEIVED and out.reason is ReasonCode.TRANSCRIBE_TRANSIENT
    assert raw.is_file(), "raw audio is kept for the retry worker"
    assert not work_dir_for(deps.settings.tmp_dir, "w1").exists()  # decoded WAV gone
    assert files_in(deps.settings.tmp_dir) == [raw]
    row = walk_row(deps, "w1")
    assert row["status"] == "received" and row["reason"] == "transcribe_transient"

    assert J.retry_pending(clock.advance(10), deps) == []  # not due yet (30 s backoff)
    assert ollama.transcribe_calls == 1
    results = J.retry_pending(clock.advance(25), deps)
    assert [r.status for r in results] == [WalkStatus.DONE]
    assert ollama.transcribe_calls == 2
    assert files_in(deps.settings.tmp_dir) == []
    assert walk_row(deps, "w1")["status"] == "done"
    assert J.retry_pending(clock.advance(60), deps) == []


def test_transient_until_window_expiry(tmp_path: Path) -> None:
    clock = FakeClock()
    ollama = StubOllama(transcripts=[transient()])
    deps = journal_deps(tmp_path, ollama=ollama, clock=clock)
    out, raw = _run(deps, "en_20s.webm")
    assert out.pending and raw.is_file()
    attempts_at: list[float] = []
    t = 0.0
    while t < 700:
        t += 5
        for r in J.retry_pending(clock.advance(5), deps):
            attempts_at.append(t)
            if r.status is WalkStatus.LOST:
                break
        if walk_row(deps, "w1")["status"] == "lost":
            break
    # backoff 30, 60, 120, 240 (+240 capped) -> retries at 30, 90, 210, 450; expiry at 600
    assert attempts_at == [30, 90, 210, 450, 600]
    assert ollama.transcribe_calls == 5
    row = walk_row(deps, "w1")
    assert row["status"] == "lost" and row["reason"] == "retry_window_expired"
    assert not raw.exists() and files_in(deps.settings.tmp_dir) == []


def test_retry_after_restart_without_memory(tmp_path: Path) -> None:
    clock = FakeClock()
    ollama = StubOllama(transcripts=[transient(), "We walked to the park."])
    deps = journal_deps(tmp_path, ollama=ollama, clock=clock)
    _run(deps, "en_20s.webm")
    deps.retry = J.RetryState()  # hub restarted: no in-memory backoff state
    results = J.retry_pending(clock.advance(1), deps)
    assert [r.status for r in results] == [WalkStatus.DONE]
    assert files_in(deps.settings.tmp_dir) == []


# ------------------------------------------------------------------ startup sweep


def test_startup_sweep_deletes_old_audio(tmp_path: Path) -> None:
    clock = FakeClock()
    deps = journal_deps(tmp_path, clock=clock)
    add_received_walk(deps, "old")
    add_received_walk(deps, "new")
    old = stage_audio(deps, "en_20s.webm", "old")
    work = work_dir_for(deps.settings.tmp_dir, "old")
    work.mkdir()
    (work / "old.16k.wav").write_bytes(b"RIFF")
    new = stage_audio(deps, "en_20s.m4a", "new")
    eleven_min_ago = (clock.now - dt.timedelta(minutes=11)).timestamp()
    nine_min_ago = (clock.now - dt.timedelta(minutes=9)).timestamp()
    for p in (old, work / "old.16k.wav"):
        os.utime(p, (eleven_min_ago, eleven_min_ago))
    os.utime(new, (nine_min_ago, nine_min_ago))
    assert J.sweep_stale_audio(clock.now, deps) == 2
    assert files_in(deps.settings.tmp_dir) == [new] and not work.exists()
    assert walk_row(deps, "old")["status"] == "lost"
    assert walk_row(deps, "old")["reason"] == "stale_audio_swept"
    assert walk_row(deps, "new")["status"] == "received"


# ------------------------------------------------------------------ state and checkpointer


def _assert_no_bytes(value: Any, path: str = "state") -> None:
    assert not isinstance(value, bytes | bytearray | memoryview), path
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_no_bytes(v, f"{path}.{k}")
    elif isinstance(value, list | tuple | set):
        for i, v in enumerate(value):
            _assert_no_bytes(v, f"{path}[{i}]")


@pytest.mark.parametrize("ok", [True, False])
def test_graph_state_never_holds_bytes(tmp_path: Path, ok: bool) -> None:
    ollama = StubOllama() if ok else StubOllama(transcripts=[transient()])
    deps = journal_deps(tmp_path, ollama=ollama)
    add_received_walk(deps, "w1")
    raw = stage_audio(deps, "en_20s.webm", "w1")
    graph = J.build_journal_graph(deps)
    states = list(
        graph.stream(
            {"walk_id": "w1", "group": GROUP, "mission_id": None, "raw_path": str(raw),
             "regen_count": 0, "latency_ms": {}},
            stream_mode="values",
        )
    )  # fmt: skip
    assert len(states) >= 3
    for st in states:
        _assert_no_bytes(st)
        for key in ("raw_path", "wav_path"):
            assert st.get(key) is None or isinstance(st[key], str)


def test_no_checkpointer(tmp_path: Path) -> None:
    assert J.build_journal_graph(journal_deps(tmp_path)).checkpointer is None
    assert M.build_mission_graph(mission_deps(tmp_path / "m")).checkpointer is None
