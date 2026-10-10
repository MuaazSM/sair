"""Lifespan: sweep + resume + planning at startup; loopback-only settings (NFR-P2)."""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

import hub.app
from hub import db
from hub.api.deps import AppDeps
from hub.schemas import ReasonCode, WalkStatus
from hub.settings import Settings
from tests.integration.conftest import Env
from tests.unit._graphfix import EN_20S

pytestmark = pytest.mark.integration

GROUP = "Mom and Dad"


def _wait(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _row(env: Env, walk_id: str) -> Any:
    conn = db.connect(env.deps.settings.db_path)
    try:
        return db.get_walk_row(conn, walk_id)
    finally:
        conn.close()


def test_startup_sweeps_stale_audio_and_marks_walk_lost(make_env: Callable[..., Env]) -> None:
    def seed(deps: AppDeps) -> None:
        now = deps.clock()
        conn = deps.connect()
        try:
            for wid in ("stale", "fresh"):
                db.insert_walk(conn, walk_id=wid, group=GROUP, recorded_at=now)
        finally:
            conn.close()
        deps.settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        for wid, age in (("stale", 3600), ("fresh", 5)):
            p = deps.settings.tmp_dir / f"{wid}.webm"
            p.write_bytes(b"x")
            os.utime(p, (now.timestamp() - age,) * 2)

    env = make_env(before_start=seed, options={"retry_interval_s": 3600.0})
    names = sorted(p.name for p in env.tmp_dir.iterdir())
    assert "stale.webm" not in names  # FR-J6 sweep
    row = _row(env, "stale")
    assert row["status"] == "lost" and row["reason"] == ReasonCode.STALE_AUDIO_SWEPT.value


def test_startup_resumes_walk_with_transcript_and_no_entry(make_env: Callable[..., Env]) -> None:
    def seed(deps: AppDeps) -> None:
        conn = deps.connect()
        try:
            db.insert_walk(conn, walk_id="stuck", group=GROUP, recorded_at=deps.clock())
            db.set_transcript(conn, "stuck", EN_20S)  # crash after transcribe, before store
        finally:
            conn.close()

    env = make_env(before_start=seed, options={"run_workers": True, "retry_interval_s": 0.05})
    assert _wait(lambda: _row(env, "stuck")["status"] == WalkStatus.DONE.value)
    conn = db.connect(env.deps.settings.db_path)
    try:
        entry = conn.execute("SELECT text FROM entries WHERE walk_id = 'stuck'").fetchone()
    finally:
        conn.close()
    assert entry is not None and entry["text"].startswith("Mom and Dad heard a koel")
    assert env.ollama.transcribe_calls == 0  # no audio was needed: resumed at the writer


def _cached(env: Env) -> bool:
    conn = db.connect(env.deps.settings.db_path)
    try:
        return db.get_cached_mission(conn, GROUP, env.deps.today()) is not None
    finally:
        conn.close()


def test_startup_plans_todays_mission(make_env: Callable[..., Env]) -> None:
    env = make_env(options={"plan_on_startup": True})  # 18:30 IST: outside the heat block
    assert _wait(lambda: _cached(env))


def test_startup_does_not_plan_inside_the_heat_block(make_env: Callable[..., Env]) -> None:
    def noon(deps: AppDeps) -> None:
        deps.journal.clock = lambda: dt.datetime(2026, 10, 10, 7, 0, tzinfo=dt.UTC)  # 12:30 IST

    env = make_env(options={"plan_on_startup": True}, before_start=noon)
    time.sleep(0.5)
    assert not _cached(env)


def test_non_loopback_host_is_refused() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, sair_env="test", sair_host="0.0.0.0")  # type: ignore[call-arg]
    assert "0.0.0.0" not in str(exc.value)
    assert "loopback" in str(exc.value)


def test_importing_hub_app_builds_nothing() -> None:
    """The lazy ``app`` attribute is the only thing that builds the app (and its clients)."""
    assert hub.app._default is None  # noqa: SLF001
