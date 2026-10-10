"""§T FR-D1-D4: diary order, chips, asked-again dates, summary, group filter, no times."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable

import pytest

from hub import db
from hub.schemas import Mission, MissionSource, MissionType
from tests.integration.conftest import Env, bearer
from tests.unit._graphfix import EN_20S, GOOD_ENTRY, StubOllama

pytestmark = pytest.mark.integration

T_MAYBE = "Maybe a kingfisher was sitting near the temple pond. The lane was empty."
T_DAD = "A koel was calling from somewhere near the temple, the same as last time. It was quiet."
MAYBE_OUT = json.dumps(
    {
        "facts": [
            {"span": "Maybe a kingfisher was sitting near the temple pond", "kind": "animal",
             "subject": "kingfisher", "certainty": "unsure", "place": "near the temple pond"},
        ],
        "mission_result": "not_mentioned",
        "entry": "Mom and Dad saw maybe a kingfisher near the temple pond.",
    }
)  # fmt: skip
DAD_OUT = json.dumps(
    {
        "facts": [
            {"span": "A koel was calling from somewhere near the temple", "kind": "sound",
             "subject": "koel", "certainty": "sure", "place": "near the temple"},
        ],
        "mission_result": "not_mentioned",
        "entry": "Dad heard a koel calling near the temple.",
    }
)  # fmt: skip


def _writer(messages: object) -> str:
    text = " ".join(m.content for m in messages)  # type: ignore[attr-defined]
    if "kingfisher" in text:
        return MAYBE_OUT
    if "It was quiet." in text:
        return DAD_OUT
    return GOOD_ENTRY


def _walk(env: Env, device: str) -> str:
    wid: str = env.post_walk(device=device).json()["walk_id"]
    assert env.wait_walk(wid, device)["status"] == "done"
    return wid


@pytest.fixture
def diary_env(make_env: Callable[..., Env]) -> Env:
    stub = StubOllama(transcripts=[EN_20S, T_MAYBE, T_DAD], writer=[_writer])
    env = make_env(ollama=stub, sair_device_groups={"mom": "Mom and Dad", "dad": "Dad"})
    w1 = _walk(env, "mom")
    env.clock.advance(86400)
    _walk(env, "mom")
    env.clock.advance(86400)
    _walk(env, "dad")
    conn = db.connect(env.deps.settings.db_path)
    try:  # a later mission asks again about walk 1's koel (FR-D2)
        fid = conn.execute("SELECT id FROM facts WHERE walk_id = ?", (w1,)).fetchone()["id"]
        db.insert_mission(
            conn,
            Mission(id="c" * 32, date=env.clock.now.date() + dt.timedelta(days=3),
                    group="Mom and Dad", type=MissionType.RETURN, references=[fid],
                    text="Listen again near the temple.", source=MissionSource.MODEL),
        )  # fmt: skip
    finally:
        conn.close()
    return env


def test_order_chips_asked_again_and_summary(diary_env: Env) -> None:
    r = diary_env.client.get("/diary", headers=bearer("mom"))
    assert r.status_code == 200
    body = r.json()
    entries = body["entries"]
    # newest first
    assert [e["date"] for e in entries] == ["2026-10-12", "2026-10-11", "2026-10-10"]
    assert [e["group"] for e in entries] == ["Dad", "Mom and Dad", "Mom and Dad"]
    assert body["summary"] == {"walks_this_month": 3, "things_noticed": 3}
    maybe = entries[1]["facts"][0]
    assert maybe["certainty"] == "unsure" and maybe["subject"] == "kingfisher"
    sure = entries[2]["facts"][0]
    assert sure["certainty"] == "sure" and sure["asked_again_on"] == ["2026-10-15"]
    assert entries[1]["facts"][0]["asked_again_on"] == []
    assert entries[2]["entry"].startswith("Mom and Dad heard a koel")
    assert entries[2]["mission_result"] == "not_mentioned"


def test_group_filter_and_summary_follow_it(diary_env: Env) -> None:
    dad = diary_env.client.get("/diary", params={"group": "Dad"}, headers=bearer("mom")).json()
    assert [e["group"] for e in dad["entries"]] == ["Dad"]
    assert dad["summary"] == {"walks_this_month": 1, "things_noticed": 1}
    both = diary_env.client.get("/diary", params={"group": "Mom and Dad"}, headers=bearer("dad"))
    assert len(both.json()["entries"]) == 2


def test_no_times_no_streaks_no_likes(diary_env: Env) -> None:
    text = diary_env.client.get("/diary", headers=bearer("mom")).text
    assert not re.search(r"\d{2}:\d{2}", text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)
    for word in ("streak", "like", "share", "recorded_at", "created_at"):
        assert word not in text.lower()
