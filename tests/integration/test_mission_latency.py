"""§T NFR-L1 / FR-M7: /mission/today + /audio/{id} start fast and never wait on an LLM."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from hub import db
from hub.schemas import Mission, MissionSource, MissionType, WalkStatus
from tests.integration.conftest import Env, bearer
from tests.unit._graphfix import GOOD_PLAN_SENSE, StubOllama

pytestmark = pytest.mark.integration

GROUP = "Mom and Dad"


def test_cached_mission_and_audio_in_milliseconds(env: Env) -> None:
    mission = Mission(id="d" * 32, date=env.deps.today(), group=GROUP, type=MissionType.SENSE,
                      text="Listen for one bird you can hear from the path.",
                      source=MissionSource.MODEL)  # fmt: skip
    conn = db.connect(env.deps.settings.db_path)
    try:
        db.insert_mission(conn, mission)
    finally:
        conn.close()
    audio = db.mission_audio_path(env.deps.settings.audio_dir, mission.id)
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"ID3cached")
    t0 = time.perf_counter()
    r = env.client.get("/mission/today", headers=bearer("mom"))
    body = r.json()
    a = env.client.get(body["audio_path"], headers=bearer("mom"))
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200 and body["mission_id"] == mission.id
    assert body["audio_path"] == f"/audio/{mission.id}" and body["type"] == "sense"
    assert a.status_code == 200 and a.content == b"ID3cached"
    assert a.headers["cache-control"] == "no-store" and a.headers["content-type"] == "audio/mpeg"
    assert elapsed < 2.0  # NFR-L1 (in practice a few ms)


def test_cache_miss_serves_a_fallback_without_waiting_for_the_planner(
    make_env: Callable[..., Env],
) -> None:
    release = threading.Event()
    entered = threading.Event()

    def slow_planner(*_a: object) -> str:
        entered.set()
        release.wait(10)
        return GOOD_PLAN_SENSE

    env = make_env(ollama=StubOllama(planner=[slow_planner]))
    conn = db.connect(env.deps.settings.db_path)
    try:  # past the seed phase, so a miss means fallback now + planner later
        for i in range(3):
            db.insert_walk(conn, walk_id=f"w{i}", group=GROUP, recorded_at=env.clock())
            db.update_walk_status(conn, f"w{i}", WalkStatus.DONE)
    finally:
        conn.close()
    try:
        t0 = time.perf_counter()
        r = env.client.get("/mission/today", headers=bearer("mom"))
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200 and r.json()["text"]
        assert r.json()["type"] in {"return", "change", "sense", "count"}
        assert elapsed < 2.0
        assert entered.wait(5), "planning was scheduled in the background"
        again = time.perf_counter()
        r2 = env.client.get("/mission/today", headers=bearer("mom"))
        assert r2.json()["mission_id"] == r.json()["mission_id"]  # stable for the day
        assert time.perf_counter() - again < 2.0
        assert r.json()["audio_path"] is None  # TTS render is queued behind, not awaited
    finally:
        release.set()
