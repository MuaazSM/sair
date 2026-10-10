"""§T FR-O2: /status is metadata only. A canary marker must never appear in it."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from hub import db
from hub.schemas import CanaryResult, Mission, MissionSource, MissionType
from tests.integration.conftest import Env, bearer
from tests.unit._graphfix import StubOllama

pytestmark = pytest.mark.integration

MARK = "zqxcanary7"
TRANSCRIPT = f"We heard a {MARK} calling near the temple."
WRITER = json.dumps(
    {
        "facts": [
            {"span": f"We heard a {MARK} calling near the temple", "kind": "sound",
             "subject": MARK, "certainty": "sure", "place": "near the temple"},
        ],
        "mission_result": "not_mentioned",
        "entry": f"Mom and Dad heard a {MARK} calling near the temple.",
    }
)  # fmt: skip
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _strings(obj: Any) -> list[str]:
    """String values."""
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return [obj] if isinstance(obj, str) else []


def _keys(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [s for v in obj.values() for s in _keys(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _keys(v)]
    return []


def test_status_has_no_free_text(make_env: Any) -> None:
    env: Env = make_env(ollama=StubOllama(transcripts=[TRANSCRIPT], writer=[WRITER]))
    env.deps.health["ollama"] = lambda: True
    conn = db.connect(env.deps.settings.db_path)
    try:
        db.insert_mission(
            conn,
            Mission(id="m" * 32, date=env.deps.today(), group="Mom and Dad", type=MissionType.SENSE,
                    text=f"Listen for the {MARK} near the path.", source=MissionSource.MODEL),
        )  # fmt: skip
        db.set_meta(conn, "last_canary_result", CanaryResult.PASSED.value)
    finally:
        conn.close()
    walk_id = env.post_walk().json()["walk_id"]
    assert env.wait_walk(walk_id)["status"] == "done"

    conn = db.connect(env.deps.settings.db_path)
    try:  # the marker really is in the DB (transcript at least) so the test can fail
        assert MARK in db.get_walk_row(conn, walk_id)["transcript"]  # type: ignore[index]
        assert conn.execute("SELECT COUNT(*) FROM missions WHERE text LIKE ?", (f"%{MARK}%",))
    finally:
        conn.close()

    r = env.client.get("/status", headers=bearer("owner"))
    assert r.status_code == 200
    assert MARK not in r.text
    body = r.json()
    assert body["raw_audio_on_disk"] == 0
    assert body["model_health"] == {"ollama": True}
    assert body["last_canary_result"] == "passed"
    assert body["per_group"] == {"Mom and Dad": {"walks": 1, "facts": 1}}
    assert body["latency"]["journal_total"]["target_p50_ms"] == 15000.0
    # Every string value is an enum code or a date; keys are node/group/code names.
    allowed_values = {c.value for c in CanaryResult}
    for value in _strings(body):
        assert value in allowed_values or DATE.match(value), "free text in /status"
    # Keys are node names, enum codes, field names or configured group names: no free text.
    for key in _keys(body):
        assert key in ("Mom and Dad",) or re.fullmatch(r"[a-z0-9_]+", key), "free text key"
        assert MARK not in key
