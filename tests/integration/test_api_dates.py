"""§T NFR-P5: API responses expose dates, never times."""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.integration.conftest import Env, bearer

pytestmark = pytest.mark.integration

TIME = re.compile(r"\d{2}:\d{2}")
ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_KEYS = {"date", "last_canary_date"}


def _scan(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in DATE_KEYS and v is not None:
                assert DATE.match(v), f"{path}.{k} is not YYYY-MM-DD"
            _scan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if path.endswith("asked_again_on"):
                assert DATE.match(v)
            _scan(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        assert not TIME.search(obj), f"time-like string at {path}"
        assert not ISO_DATETIME.search(obj), f"datetime at {path}"


def test_no_response_exposes_a_time(env: Env) -> None:
    mom, owner = bearer("mom"), bearer("owner")
    c = env.client
    responses = [c.get("/healthz")]
    mission = c.get("/mission/today", headers=mom)
    responses.append(mission)
    walk_id = env.post_walk().json()["walk_id"]
    responses.append(env.post_walk())
    env.wait_walk(walk_id)
    responses += [
        c.get(f"/walk/{walk_id}", headers=mom),
        c.get("/diary", headers=mom),
        c.get("/status", headers=owner),
        c.post(f"/feedback/{walk_id}", headers=owner, json={"correct": True, "note": "ok"}),
        c.get("/walk/" + "0" * 32, headers=mom),  # error body
        c.get("/status", headers=mom),
        c.get("/nope"),
    ]
    for r in responses:
        _scan(r.json())
        for name in r.headers:
            if name.lower() in ("last-modified", "etag", "expires"):
                raise AssertionError(f"time-bearing header {name}")
    assert DATE.match(c.get(f"/walk/{walk_id}", headers=mom).json()["date"])
