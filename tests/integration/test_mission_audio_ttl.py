"""§T FR-M8: mission audio rendered by TTS, cached 48 h, then deleted (frozen clock)."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from hub import db
from hub.graphs import mission as M
from hub.models.errors import Transient
from hub.schemas import MissionSource
from tests.unit._graphfix import GROUP, StubTts, make_settings, mission_deps

pytestmark = pytest.mark.integration

NOW = dt.datetime(2026, 10, 10, 13, 0, tzinfo=dt.UTC)


def _file(d: Path, name: str, age_h: float) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"ID3")
    ts = (NOW - dt.timedelta(hours=age_h)).timestamp()
    os.utime(p, (ts, ts))
    return p


def test_expire_deletes_only_older_than_48h(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    old = _file(s.audio_dir, "old.mp3", 48.5)
    edge = _file(s.audio_dir, "edge.mp3", 47.9)
    fresh = _file(s.audio_dir, "fresh.mp3", 1)
    assert M.expire_mission_audio(NOW, s) == 1
    assert not old.exists() and edge.exists() and fresh.exists()
    assert M.expire_mission_audio(NOW + dt.timedelta(hours=1), s) == 1
    assert not edge.exists() and fresh.exists()


def test_planned_mission_rendered_to_audio_dir(tmp_path: Path) -> None:
    tts = StubTts()
    deps = mission_deps(tmp_path, tts=tts)
    out = M.plan_mission(GROUP, NOW.date(), deps)
    assert out is not None and out.audio and tts.calls == 1
    assert db.mission_audio_path(deps.settings.audio_dir, out.mission_id).is_file()


def test_tts_failure_still_caches_without_audio(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path, tts=StubTts(fail=Transient(None, "tts")))
    out = M.plan_mission(GROUP, NOW.date(), deps)
    assert out is not None and out.audio is False and out.source is MissionSource.SEED
    conn = deps.connect()
    try:
        cached = db.get_cached_mission(conn, GROUP, NOW.date())
    finally:
        conn.close()
    assert cached is not None and cached.id == out.mission_id
    assert not db.mission_audio_path(deps.settings.audio_dir, out.mission_id).exists()
    # the API can retry rendering later
    deps.tts = StubTts()
    assert M.render_mission_audio(cached, deps) is not None
