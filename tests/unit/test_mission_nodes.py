"""Mission-graph nodes: recall limits, regen then fallback, seeds, pregen block, repeats."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from hub import db
from hub.graphs import mission as M
from hub.schemas import MissionSource, MissionType, ReasonCode
from tests.unit import _dbfix
from tests.unit._graphfix import (
    GOOD_PLAN_SENSE,
    GROUP,
    UNSAFE_PLAN,
    StubOllama,
    StubTts,
    make_settings,
    mission_deps,
    transient,
)

TODAY = dt.date(2026, 10, 10)


def _done_walks(deps: M.MissionDeps, n: int, group: str = GROUP, start_day: int = 1) -> None:
    conn = deps.connect()
    try:
        for i in range(n):
            _dbfix.add_walk(conn, f"{group.replace(' ', '')}-w{i}", start_day + i, group=group,
                            verified=(True,))  # fmt: skip
    finally:
        conn.close()


def _state(**kw: object) -> M.MissionState:
    base: dict[str, object] = {"group": GROUP, "date": TODAY, "regen_count": 0, "reasons": [],
                               "latency_ms": {}}  # fmt: skip
    base.update(kw)
    return base  # type: ignore[return-value]


def test_recall_limits_and_group_scope(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path)
    _done_walks(deps, 9)  # 9 done walks, one verified fact each
    _done_walks(deps, 2, group="Mom", start_day=20)
    conn = deps.connect()
    try:
        for i in range(7):
            db.insert_mission(conn, _dbfix.mission(f"m{i}", i + 1, text=f"Notice sound {i}."))
        db.insert_mission(conn, _dbfix.mission("other", 9, text="Other group.", group="Mom"))
    finally:
        conn.close()
    upd = M.recall(_state(), deps)
    facts = upd["facts"]
    assert len(facts) == 7 and {f.group for f in facts} == {GROUP}
    assert {f.walk_id for f in facts} == {f"MomandDad-w{i}" for i in range(2, 9)}  # last 7 walks
    assert upd["recent"] == [f"Notice sound {i}." for i in range(6, 1, -1)]  # last 5
    assert upd["done_walks"] == 9


def test_regen_then_fallback(tmp_path: Path) -> None:
    ollama = StubOllama(planner=[UNSAFE_PLAN, UNSAFE_PLAN])
    deps = mission_deps(tmp_path, ollama=ollama)
    _done_walks(deps, 3)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None
    assert out.source is MissionSource.FALLBACK and out.regen_count == 1
    assert len(ollama.planner_calls) == 2
    assert ReasonCode.REGENERATED in out.reasons and ReasonCode.MISSION_FALLBACK in out.reasons
    assert "Do not mention any time of day." in ollama.planner_calls[1][1].content


def test_regen_then_pass(tmp_path: Path) -> None:
    ollama = StubOllama(planner=[UNSAFE_PLAN, GOOD_PLAN_SENSE])
    deps = mission_deps(tmp_path, ollama=ollama)
    _done_walks(deps, 3)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None and out.source is MissionSource.MODEL and out.regen_count == 1


def test_planner_node_and_check_node(tmp_path: Path) -> None:
    ollama = StubOllama(planner=[GOOD_PLAN_SENSE])
    deps = mission_deps(tmp_path, ollama=ollama)
    upd = M.plan(_state(facts=[], recent=[]), deps)
    assert upd["raw_plan"] == GOOD_PLAN_SENSE and upd["planner_error"] is None
    chk = M.mission_check(_state(facts=[], recent=[], raw_plan=upd["raw_plan"]), deps)
    assert chk["check_passed"] is True and chk["mission"].source is MissionSource.MODEL
    chk = M.mission_check(_state(facts=[], recent=[], raw_plan=UNSAFE_PLAN), deps)
    assert chk["check_passed"] is False and ReasonCode.TIME_OF_DAY in chk["check_reasons"]


def test_planner_transient_goes_to_fallback(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path, ollama=StubOllama(planner=[transient()]))
    _done_walks(deps, 3)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None and out.source is MissionSource.FALLBACK
    assert ReasonCode.PLANNER_TRANSIENT in out.reasons


@pytest.mark.parametrize("done", [0, 1, 2])
def test_seed_missions_for_first_three_walks(tmp_path: Path, done: int) -> None:
    ollama = StubOllama()
    deps = mission_deps(tmp_path, ollama=ollama)
    _done_walks(deps, done)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None and out.source is MissionSource.SEED and out.type is MissionType.SEED
    assert ollama.planner_calls == []
    conn = deps.connect()
    try:
        cached = db.get_cached_mission(conn, GROUP, TODAY)
    finally:
        conn.close()
    assert cached is not None
    assert cached.text == M._seeds()[done].text


def test_fourth_walk_uses_planner(tmp_path: Path) -> None:
    ollama = StubOllama()
    deps = mission_deps(tmp_path, ollama=ollama)
    _done_walks(deps, 3)
    out = M.plan_mission(GROUP, TODAY, deps)
    assert out is not None and out.source is MissionSource.MODEL
    assert len(ollama.planner_calls) == 1


def test_seed_node(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path)
    upd = M.seed(_state(done_walks=1), deps)
    assert upd["mission"].text == M._seeds()[1].text


def test_fallback_node_and_avoids_recent_repeats(tmp_path: Path) -> None:
    deps = mission_deps(tmp_path)
    first = M.pick_fallback(TODAY, [])[1]
    again = M.pick_fallback(TODAY, [])[1]
    assert first == again  # deterministic
    other = M.pick_fallback(TODAY, [first])[1]
    assert other != first
    pool = [m.text for m in M._fallbacks()]
    recent = pool[:5]
    for day in range(30):
        _t, text = M.pick_fallback(TODAY + dt.timedelta(days=day), recent)
        assert text not in recent
    upd = M.fallback(_state(recent=recent), deps)
    assert upd["mission"].source is MissionSource.FALLBACK
    assert upd["mission"].text not in recent


def test_speak_and_cache_nodes(tmp_path: Path) -> None:
    tts = StubTts()
    deps = mission_deps(tmp_path, tts=tts)
    mission = M.seed(_state(done_walks=0), deps)["mission"]
    upd = M.speak(_state(mission=mission), deps)
    assert upd["audio_path"] == str(db.mission_audio_path(deps.settings.audio_dir, mission.id))
    M.cache(_state(mission=mission, audio_path=upd["audio_path"]), deps)
    conn = deps.connect()
    try:
        assert db.get_cached_mission(conn, GROUP, TODAY) == mission
    finally:
        conn.close()


def test_regen_node(tmp_path: Path) -> None:
    ollama = StubOllama(planner=[GOOD_PLAN_SENSE])
    deps = mission_deps(tmp_path, ollama=ollama)
    upd = M.regen(_state(facts=[], recent=[], check_reasons=(ReasonCode.TIME_OF_DAY,)), deps)
    assert upd["regen_count"] == 1 and ReasonCode.REGENERATED in upd["reasons"]


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    ("hh", "mm", "allowed"),
    [(11, 29, True), (11, 30, False), (13, 0, False), (15, 59, False), (16, 0, True),
     (16, 30, True), (6, 0, True)],
)  # fmt: skip
def test_pregen_block(tmp_path: Path, hh: int, mm: int, allowed: bool) -> None:
    s = make_settings(tmp_path)
    now = dt.datetime(2026, 10, 10, hh, mm, tzinfo=IST)
    assert M.should_pregenerate(now, s) is allowed
    assert M.should_pregenerate(now.astimezone(dt.UTC), s) is allowed


def test_planner_prompt_has_only_facts_and_missions(tmp_path: Path) -> None:
    ollama = StubOllama()
    deps = mission_deps(tmp_path, ollama=ollama)
    _done_walks(deps, 3)
    M.plan_mission(GROUP, TODAY, deps)
    user = ollama.planner_calls[0][1].content
    assert "Mom and Dad saw a dog." not in user  # entry prose never reaches the planner
    assert json.loads(user.split("\n")[3])["subject"] == "dog"
