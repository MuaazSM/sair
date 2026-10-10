"""POST /walk and GET /walk/{id} through the API (FR-J1, FR-J6, FR-J8)."""

from __future__ import annotations

import time

import pytest

from tests.integration.conftest import Env, bearer
from tests.unit._graphfix import FIXTURES, files_in

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", ["en_20s.webm", "en_20s.m4a"])
def test_upload_to_done_entry(env: Env, name: str) -> None:
    r = env.post_walk(name)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "received" and len(body["walk_id"]) == 32
    done = env.wait_walk(body["walk_id"])
    assert done["status"] == "done"
    assert done["entry"] == "Mom and Dad heard a koel calling near the temple."
    assert done["facts"] == [
        {"subject": "koel", "kind": "sound", "certainty": "sure", "place": "near the temple"}
    ]
    assert done["mission_result"] == "not_mentioned"
    assert "reason" not in done
    assert files_in(env.tmp_dir) == []  # FR-J6 via the API: no raw audio left
    for _ in range(100):  # notify runs just after the store that flips the status
        if env.notifier.walks:
            break
        time.sleep(0.02)
    assert env.notifier.walks and env.notifier.walks[0][1] == 1


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("short_0_5s.webm", "audio_too_short"),
        ("silent_5s.webm", "audio_silent"),
        ("en_95s.webm", "audio_too_long"),
    ],
)
def test_permanent_audio_failures_are_lost_with_reason(env: Env, name: str, reason: str) -> None:
    walk_id = env.post_walk(name).json()["walk_id"]
    body = env.wait_walk(walk_id)
    assert body["status"] == "lost" and body["reason"] == reason
    assert "entry" not in body and "facts" not in body
    assert files_in(env.tmp_dir) == []


def test_unsupported_type_is_415_and_leaves_nothing(env: Env) -> None:
    r = env.client.post(
        "/walk",
        headers=bearer("mom"),
        files={"audio": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415 and r.json() == {"error": "unsupported_media_type"}
    assert files_in(env.tmp_dir) == []


def test_generic_content_type_falls_back_to_filename_extension(env: Env) -> None:
    src = FIXTURES / "en_20s.m4a"
    r = env.client.post(
        "/walk",
        headers=bearer("mom"),
        files={"audio": ("note.m4a", src.read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 202
    assert env.wait_walk(r.json()["walk_id"])["status"] == "done"


def test_missing_audio_part_is_422(env: Env) -> None:
    r = env.client.post("/walk", headers=bearer("mom"), data={"group": "Dad"})
    assert r.status_code in (400, 422)
    r = env.client.post(
        "/walk", headers=bearer("mom"), files={"other": ("x.webm", b"abc", "audio/webm")}
    )
    assert r.status_code == 422 and r.json() == {"error": "missing_audio"}
    assert files_in(env.tmp_dir) == []


def test_size_cap_is_413_and_partial_file_removed(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(options={"max_upload_bytes": 1000})
    r = env.client.post(
        "/walk",
        headers=bearer("mom"),
        files={"audio": ("big.webm", b"x" * 5000, "audio/webm")},
    )
    assert r.status_code == 413 and r.json() == {"error": "payload_too_large"}
    assert files_in(env.tmp_dir) == []


def test_unknown_walk_is_404(env: Env) -> None:
    r = env.client.get("/walk/" + "b" * 32, headers=bearer("mom"))
    assert r.status_code == 404 and r.json() == {"error": "not_found"}


def test_upload_is_saved_under_walk_id_in_tmp_dir(make_env) -> None:  # type: ignore[no-untyped-def]
    """The journal graph finds the upload as <tmp>/<walk_id><ext>; hold it with a slow stub."""
    import threading

    gate = threading.Event()

    def slow(*_a: object) -> str:
        gate.wait(10)
        return "A koel was calling from somewhere near the temple, the same as last time."

    from tests.unit._graphfix import StubOllama

    env = make_env(ollama=StubOllama(transcripts=[slow]))
    try:
        walk_id = env.post_walk("en_20s.webm").json()["walk_id"]
        names = [p.name for p in files_in(env.tmp_dir)]
        assert any(n.startswith(walk_id) for n in names)
        assert env.client.get(f"/walk/{walk_id}", headers=bearer("mom")).json() == {
            "status": "received",
            "date": "2026-10-10",
        }
    finally:
        gate.set()
    assert env.wait_walk(walk_id)["status"] == "done"
    assert files_in(env.tmp_dir) == []


def test_unexpected_writer_error_is_done_with_template(make_env) -> None:  # type: ignore[no-untyped-def]
    """FR-J5/FR-J7 via the API: never requeued as a transcription failure, never lost."""
    from hub import db
    from tests.unit._graphfix import StubOllama

    env = make_env(ollama=StubOllama(writer=[ValueError("boom")]))
    wid = env.post_walk().json()["walk_id"]
    done = env.wait_walk(wid)
    assert done["status"] == "done"
    assert done["entry"] == "Mom and Dad went for their evening walk."
    assert "reason" not in done  # reason is exposed only for lost walks
    assert files_in(env.tmp_dir) == []
    conn = db.connect(env.deps.settings.db_path)
    try:
        row = db.get_walk_row(conn, wid)
    finally:
        conn.close()
    assert row["transcript"] and (row["status"], row["reason"]) == ("done", "writer_permanent")
    assert env.ollama.transcribe_calls == 1


def test_graph_crash_after_transcript_requeues_as_writer_transient(  # type: ignore[no-untyped-def]
    make_env, monkeypatch
) -> None:
    """A crash with the transcript stored must resume at the writer, not re-transcribe."""
    from hub import db
    from hub.api import routes

    def crash(walk_id, path, group, mission_id, deps):  # type: ignore[no-untyped-def]
        conn = deps.connect()
        try:
            db.set_transcript(conn, walk_id, "A koel was calling near the temple.")
        finally:
            conn.close()
        path.unlink()
        raise RuntimeError("graph bug")

    env = make_env()
    monkeypatch.setattr(routes, "run_journal", crash)
    wid = env.post_walk().json()["walk_id"]
    for _ in range(200):
        conn = db.connect(env.deps.settings.db_path)
        try:
            row = db.get_walk_row(conn, wid)
        finally:
            conn.close()
        if row["reason"] is not None:
            break
        time.sleep(0.02)
    assert (row["status"], row["reason"]) == ("received", "writer_transient")
