"""§T §9.3 auth: 401 without/with a bad token, 403 for family on owner-only, group from token."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from hub import db
from tests.integration.conftest import Env, bearer

pytestmark = pytest.mark.integration

WALK_ID = "a" * 32
ENDPOINTS: list[tuple[str, str]] = [
    ("GET", "/mission/today"),
    ("GET", f"/audio/{WALK_ID}"),
    ("POST", "/walk"),
    ("GET", f"/walk/{WALK_ID}"),
    ("GET", "/diary"),
    ("POST", f"/feedback/{WALK_ID}"),
    ("GET", "/status"),
]


def _call(env: Env, method: str, path: str, headers: dict[str, str] | None = None) -> Any:
    return env.client.request(method, path, headers=headers or {})


@pytest.mark.parametrize(("method", "path"), ENDPOINTS)
def test_missing_token_is_401(env: Env, method: str, path: str) -> None:
    r = _call(env, method, path)
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}
    assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path"), ENDPOINTS)
@pytest.mark.parametrize(
    "header",
    ["Bearer nope", "Bearer ", "Basic abc", "tok-mom-0123456789", "Bearer tok-mom-0123456780"],
)
def test_bad_token_is_401(env: Env, method: str, path: str, header: str) -> None:
    r = _call(env, method, path, {"Authorization": header})
    assert r.status_code == 401 and r.json() == {"error": "unauthorized"}


def test_healthz_needs_no_token(env: Env) -> None:
    r = env.client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


@pytest.mark.parametrize(("method", "path"), [("POST", f"/feedback/{WALK_ID}"), ("GET", "/status")])
@pytest.mark.parametrize("device", ["mom", "dad"])
def test_owner_only_is_403_for_family(env: Env, method: str, path: str, device: str) -> None:
    r = env.client.request(method, path, headers=bearer(device), json={"correct": True})
    assert r.status_code == 403 and r.json() == {"error": "forbidden"}


def test_owner_token_is_refused_on_family_only_endpoints(env: Env) -> None:
    assert env.client.get("/mission/today", headers=bearer("owner")).status_code == 403
    assert env.post_walk(device="owner").status_code == 403
    assert env.client.get("/status", headers=bearer("owner")).status_code == 200
    assert env.client.get("/diary", headers=bearer("owner")).status_code == 200


def test_group_comes_from_the_token_not_the_client(make_env: Callable[..., Env]) -> None:
    env = make_env(sair_device_groups={"mom": "Mom and Dad", "dad": "Dad"})
    r = env.post_walk(device="mom", data={"group": "Dad"})
    assert r.status_code == 202
    walk_id = r.json()["walk_id"]
    env.wait_walk(walk_id, "mom")
    r2 = env.post_walk(device="dad", data={"group": "Mom and Dad"})
    walk2 = r2.json()["walk_id"]
    env.wait_walk(walk2, "dad")
    conn = db.connect(env.deps.settings.db_path)
    try:
        g1 = db.get_walk_row(conn, walk_id)["group_name"]  # type: ignore[index]
        g2 = db.get_walk_row(conn, walk2)["group_name"]  # type: ignore[index]
    finally:
        conn.close()
    assert (g1, g2) == ("Mom and Dad", "Dad")
    # another group's walk is invisible to the caller (404), visible to the owner
    assert env.client.get(f"/walk/{walk_id}", headers=bearer("dad")).status_code == 404
    assert env.client.get(f"/walk/{walk_id}", headers=bearer("owner")).status_code == 200
    # mission audio and /mission/today are scoped to the token's group too
    m_mom = env.client.get("/mission/today", headers=bearer("mom")).json()["mission_id"]
    assert env.client.get(f"/audio/{m_mom}", headers=bearer("dad")).status_code == 404


def test_diary_group_filter_is_restricted_to_configured_groups(
    make_env: Callable[..., Env],
) -> None:
    env = make_env(sair_device_groups={"mom": "Mom and Dad", "dad": "Dad"})
    ok = env.client.get("/diary", params={"group": "Dad"}, headers=bearer("mom"))
    assert ok.status_code == 200
    bad = env.client.get("/diary", params={"group": "Stranger"}, headers=bearer("mom"))
    assert bad.status_code == 400 and bad.json() == {"error": "unknown_group"}
