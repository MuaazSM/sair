"""Error paths leak nothing: no marker in responses, logs or stdout (NFR-P4, §11.1)."""

from __future__ import annotations

import json
import logging

import pytest
import structlog
from structlog.testing import capture_logs

from tests.integration.conftest import Env, bearer

pytestmark = pytest.mark.integration

MARK = "ZQXLEAK-canary-9931"


def _bad_feedback(env: Env, walk_id: str) -> object:
    return env.client.post(
        f"/feedback/{walk_id}",
        headers=bearer("owner"),
        json={"correct": MARK, "note": 12, MARK: MARK},
    )


def test_validation_error_does_not_echo_input(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    r = _bad_feedback(env, "e" * 32)
    assert r.status_code == 422  # type: ignore[attr-defined]
    assert MARK not in r.text  # type: ignore[attr-defined]
    body = r.json()  # type: ignore[attr-defined]
    assert body["error"] == "invalid_request"
    assert set(body) == {"error", "fields"}
    # Only field locations, never values or the unknown key name.
    assert all(isinstance(p, str | int) for loc in body["fields"] for p in loc)
    assert MARK not in capsys.readouterr().out


def test_bad_json_and_query_never_reach_logs(
    env: Env, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    r1 = env.client.post(
        "/feedback/" + "f" * 32, headers=bearer("owner"), content=f"{{{MARK}".encode()
    )
    r2 = env.client.get("/diary", params={"group": MARK}, headers=bearer("mom"))
    r3 = env.client.get(f"/{MARK}")
    assert (r1.status_code, r2.status_code, r3.status_code) == (422, 400, 404)
    for r in (r1, r2, r3):
        assert MARK not in r.text
    out = capsys.readouterr()
    server_logs = "\n".join(
        r.getMessage() for r in caplog.records if not r.name.startswith("httpx")
    )  # (httpx is the test client's own logger, not the hub)
    assert MARK not in out.out and MARK not in out.err and MARK not in server_logs


def test_request_log_has_only_method_route_status_latency_walk_id(env: Env) -> None:
    with capture_logs() as events:
        env.client.get("/diary", params={"group": MARK}, headers=bearer("mom"))
        env.client.get("/walk/" + "a" * 32, headers=bearer("mom"))
    assert MARK not in json.dumps(events, default=str)
    reqs = [e for e in events if e["event"] == "request"]
    assert [set(e) - {"event", "log_level"} for e in reqs] == [
        {"method", "route", "status", "ms", "walk_id"}
    ] * 2
    assert reqs[0]["route"] == "/diary" and reqs[1]["route"] == "/walk/{walk_id}"
    assert reqs[1]["walk_id"] == "a" * 32


def test_internal_error_is_fixed_body_and_logs_type_only(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError(f"transcript was {MARK}")

    monkeypatch.setattr("hub.api.routes.queries.status_response", boom)
    caplog.set_level(logging.DEBUG)
    with capture_logs() as events:
        r = env.client.get("/status", headers=bearer("owner"))
    assert r.status_code == 500 and r.json() == {"error": "internal_error"}
    assert MARK not in r.text
    assert MARK not in json.dumps(events, default=str)
    assert any(e["event"] == "unhandled_exception" and e["exc_type"] == "RuntimeError"
               for e in events)  # fmt: skip
    # same again under the real renderer: stdout/stderr and stdlib logging stay clean
    env.client.get("/status", headers=bearer("owner"))
    out = capsys.readouterr()
    assert MARK not in out.out and MARK not in out.err
    assert MARK not in "\n".join(
        r.getMessage() for r in caplog.records if not r.name.startswith("httpx")
    )
    assert structlog.is_configured()


def test_security_headers_on_every_response(env: Env) -> None:
    for r in (env.client.get("/healthz"), env.client.get("/status"), env.client.get("/x")):
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert "http" not in r.headers["content-security-policy"]
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["permissions-policy"] == "geolocation=()"
