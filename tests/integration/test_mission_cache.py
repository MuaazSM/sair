"""§T FR-M1, FR-M7: /mission/today never waits on a model; plan-after-walk."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from hub import db
from hub.graphs import journal as J
from hub.graphs import mission as M
from hub.prompts.builders import ChatMessage
from hub.schemas import MissionSource, MissionType
from tests.unit import _dbfix
from tests.unit._graphfix import (
    GOOD_ENTRY,
    GROUP,
    StubOllama,
    add_received_walk,
    journal_deps,
    mission_deps,
    stage_audio,
)

pytestmark = pytest.mark.integration

TODAY = dt.date(2026, 10, 10)


def _sleepy(messages: Sequence[ChatMessage]) -> str:
    time.sleep(5)
    raise AssertionError("planner must not be called on /mission/today")


def _never(messages: Sequence[ChatMessage]) -> str:
    raise AssertionError("no LLM call expected")


def _done(deps: M.MissionDeps, n: int) -> None:
    conn = deps.connect()
    try:
        for i in range(n):
            _dbfix.add_walk(conn, f"d{i}", i + 1, verified=(True,))
    finally:
        conn.close()


def test_miss_returns_fallback_instantly_and_schedules(tmp_path: Path) -> None:
    scheduled: list[tuple[str, dt.date]] = []
    ollama = StubOllama(planner=[_sleepy], writer=[_sleepy])
    deps = mission_deps(tmp_path, ollama=ollama, scheduler=lambda g, d: scheduled.append((g, d)))
    _done(deps, 3)
    t0 = time.perf_counter()
    m1 = M.get_or_fallback_mission(GROUP, TODAY, deps)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"{elapsed_ms:.1f} ms"
    assert m1.source is MissionSource.FALLBACK and m1.date == TODAY
    assert scheduled == [(GROUP, TODAY + dt.timedelta(days=1))]
    assert ollama.planner_calls == [] and ollama.warm_calls == 0

    ollama.planner = [_never]
    m2 = M.get_or_fallback_mission(GROUP, TODAY, deps)
    assert m2 == m1  # same cached mission
    assert ollama.planner_calls == [] and len(scheduled) == 1


def test_miss_serves_seed_for_first_walks(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path, ollama=StubOllama(planner=[_never]))
    m = M.get_or_fallback_mission(GROUP, TODAY, deps)
    assert m.source is MissionSource.SEED and m.type is MissionType.SEED


def test_cached_model_mission_is_served(tmp_path: Path) -> None:
    ollama = StubOllama()
    deps = mission_deps(tmp_path, ollama=ollama)
    _done(deps, 3)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None and out.source is MissionSource.MODEL
    ollama.planner = [_never]
    m = M.get_or_fallback_mission(GROUP, TODAY, deps)
    assert m.id == out.mission_id and m.source is MissionSource.MODEL


def test_scheduled_plan_skips_when_cached(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path, ollama=StubOllama(planner=[_never]))
    M.get_or_fallback_mission(GROUP, TODAY, deps)
    assert M.plan_mission(GROUP, TODAY, deps, replace=False) is None


def test_plan_after_walk_triggered_by_journal(tmp_path: Path) -> None:
    text = "You heard a koel near the temple. Listen for it again."
    plan = json.dumps({"type": "return", "references": ["__FID__"], "text": text})
    holder: dict[str, str] = {}

    def planner(messages: Sequence[ChatMessage]) -> str:
        user = messages[1].content
        fid = next(
            json.loads(line)["id"] for line in user.splitlines()
            if line.startswith("{") and json.loads(line)["subject"] == "koel"
        )  # fmt: skip
        holder["fid"] = fid
        return plan.replace("__FID__", fid)

    ollama = StubOllama(writer=[GOOD_ENTRY], planner=[planner])
    jdeps = journal_deps(tmp_path, ollama=ollama)
    mdeps = mission_deps(tmp_path, ollama=ollama, settings=jdeps.settings, clock=jdeps.clock)  # type: ignore[arg-type]
    conn = jdeps.connect()
    try:
        for i in range(2):
            _dbfix.add_walk(conn, f"d{i}", i + 1, verified=(True,))
    finally:
        conn.close()
    submitted: list[Callable[[], object]] = []
    jdeps.mission_trigger = M.post_walk_trigger(mdeps, submitted.append)

    add_received_walk(jdeps, "w1")
    raw = stage_audio(jdeps, "en_20s.webm", "w1")
    out = J.run_journal("w1", raw, GROUP, None, jdeps)
    assert out.facts == 1
    assert len(submitted) == 1 and ollama.planner_calls == []  # scheduled, not inline
    submitted[0]()  # the background task runs
    tomorrow = M.next_mission_date(jdeps.clock(), jdeps.settings)
    conn = jdeps.connect()
    try:
        cached = db.get_cached_mission(conn, GROUP, tomorrow)
        refs = conn.execute("SELECT fact_id FROM mission_refs").fetchall()
    finally:
        conn.close()
    assert cached is not None and cached.source is MissionSource.MODEL
    assert cached.type is MissionType.RETURN and cached.references == [holder["fid"]]
    assert [r["fact_id"] for r in refs] == [holder["fid"]]
