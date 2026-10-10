"""§T NFR-P5: every API-facing date is the local calendar date in SAIR_TZ (one helper)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pytest

from tests.integration.conftest import Env, bearer

pytestmark = pytest.mark.integration


def _walk_at(env: Env, when: dt.datetime) -> str:
    env.clock.now = when
    wid: str = env.post_walk().json()["walk_id"]
    assert env.wait_walk(wid)["status"] == "done"
    return wid


def test_late_night_utc_walk_is_the_next_local_day_everywhere(
    make_env: Callable[..., Env],
) -> None:
    env = make_env()  # SAIR_TZ defaults to Asia/Kolkata
    wid = _walk_at(env, dt.datetime(2026, 10, 10, 20, 45, tzinfo=dt.UTC))  # 02:15 IST, 11 Oct
    assert env.client.get(f"/walk/{wid}", headers=bearer("mom")).json()["date"] == "2026-10-11"
    body = env.client.get("/diary", headers=bearer("mom")).json()
    assert [e["date"] for e in body["entries"]] == ["2026-10-11"]
    assert body["summary"]["walks_this_month"] == 1  # counts toward October
    assert body["summary"]["things_noticed"] == 1


def test_month_boundary_follows_the_local_month(make_env: Callable[..., Env]) -> None:
    env = make_env()
    wid = _walk_at(env, dt.datetime(2026, 9, 30, 20, 45, tzinfo=dt.UTC))  # 02:15 IST, 1 Oct
    assert env.client.get(f"/walk/{wid}", headers=bearer("mom")).json()["date"] == "2026-10-01"
    body = env.client.get("/diary", headers=bearer("mom")).json()  # today: 1 Oct local
    assert [e["date"] for e in body["entries"]] == ["2026-10-01"]
    assert body["summary"] == {"walks_this_month": 1, "things_noticed": 1}
