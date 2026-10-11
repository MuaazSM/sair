"""NFR-P4 / PRD §15: Sentry scrubbers drop every text field; only allowlisted attrs survive.

Two layers are tested: the pure scrubbers on hand-built events with the canary in every field,
and the real SDK (fake transport) fed the canary through every capture path.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from hub import db, tracing
from hub.schemas import ReasonCode, WalkStatus, WriterBackend
from hub.settings import Settings
from tests.unit.test_tracing_prod_off import FAKE_DSN, CapturingTransport

# NFR-P7: Sentry settings refuse the prod data dir (~/.sair); never created here.
SYNTHETIC = "/nonexistent/sair-synthetic"

MARKER = "zebra-ultramarine-4471"
WALK_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    tracing.shutdown_tracing()
    yield
    tracing.shutdown_tracing()


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _poisoned_event(kind: str | None = None) -> dict[str, Any]:
    m = MARKER
    frame = {
        "function": "transcribe", "module": "hub.graphs.journal",
        "filename": "hub/graphs/journal.py",
        "abs_path": f"/Users/{m}/hub/graphs/journal.py", "lineno": 12, "in_app": True,
        "vars": {"transcript": f"we saw {m}", "self": {"x": m}},
        "context_line": f"x = '{m}'", "pre_context": [m], "post_context": [m],
    }  # fmt: skip
    ev: dict[str, Any] = {
        "event_id": "a" * 32,
        "timestamp": "2026-10-10T13:00:00Z",
        "level": "error",
        "platform": "python",
        "environment": "eval",
        "release": m,
        "dist": m,
        "server_name": f"{m}.local",
        "logger": m,
        "culprit": m,
        "fingerprint": [m],
        "message": f"failed on {m}",
        "logentry": {"message": f"walk %s {m}", "params": [m], "formatted": m},
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "module": "builtins",
                    "value": f"bad entry: {m}",
                    "mechanism": {
                        "type": "generic",
                        "handled": True,
                        "data": {"x": m},
                        "description": m,
                    },
                    "stacktrace": {"frames": [frame]},
                },
                {"type": m, "value": m},
            ]
        },  # fmt: skip
        "threads": {"values": [{"name": m, "stacktrace": {"frames": [frame]}}]},
        "request": {
            "url": f"http://127.0.0.1/{m}",
            "data": {"t": m},
            "headers": {"X": m},
            "query_string": m,
            "cookies": m,
        },  # fmt: skip
        "user": {"id": m, "username": m, "ip_address": m},
        "breadcrumbs": {"values": [{"message": m, "data": {"a": m}, "category": m}]},
        "extra": {"transcript": m, "unsupported_details": [m]},
        "contexts": {
            "trace": {
                "trace_id": "b" * 32,
                "span_id": "c" * 16,
                "op": m,
                "description": m,
                "data": {"walk_id": WALK_ID, "entry": m, "node": m},
            },  # fmt: skip
            "os": {"name": m},
            "runtime": {"name": m},
            "custom": {"text": m},
        },
        "tags": {"walk_id": WALK_ID, "mission": m, "status": m, m: "1"},
        "modules": {m: "1.0"},
        "measurements": {m: {"value": 1}},
        "sdk": {
            "name": "sentry.python",
            "version": "2.71.0",
            "integrations": [m],
            "packages": [{"name": m}],
        },  # fmt: skip
    }
    if kind == "transaction":
        ev["type"] = "transaction"
        ev["transaction"] = f"journal {m}"
        ev["transaction_info"] = {"source": m}
        ev["spans"] = [{
            "trace_id": "b" * 32, "span_id": "d" * 16, "parent_span_id": "c" * 16,
            "op": m, "description": f"llm call: {m}", "status": "ok",
            "start_timestamp": 1.0, "timestamp": 2.0, "origin": m,
            "data": {"model": m, "tokens_in": 5, "latency_ms": 10.5, "prompt": m,
                     "unsupported_details": [m], "reason_codes": [m, "new_number"],
                     "walk_id": WALK_ID, "thread.name": m},
            "tags": {m: m, "status": "ok"},
        }]  # fmt: skip
    return ev


@pytest.mark.parametrize("kind", [None, "transaction"])
def test_marker_appears_nowhere_after_scrub(kind: str | None) -> None:
    event = _poisoned_event(kind)
    assert MARKER in _dump(event)
    fn = tracing.scrub_transaction if kind else tracing.scrub_event
    out = fn(event, {"exc_info": None})
    assert MARKER not in _dump(out)
    assert "unsupported_details" not in _dump(out)
    for dropped in ("message", "logentry", "request", "user", "server_name", "extra",
                    "breadcrumbs", "threads", "modules", "release", "measurements"):  # fmt: skip
        assert dropped not in out


def test_exception_keeps_type_and_code_location_only() -> None:
    out = tracing.scrub_event(_poisoned_event(), None)
    values = out["exception"]["values"]
    assert values[0]["type"] == "ValueError"
    assert values[1]["type"] == "Exception"  # a non-identifier type is replaced
    assert all("value" not in v for v in values)
    (frame,) = values[0]["stacktrace"]["frames"]
    assert frame == {
        "function": "transcribe", "module": "hub.graphs.journal",
        "filename": "hub/graphs/journal.py", "lineno": 12, "in_app": True,
    }  # fmt: skip


def test_allowlisted_attributes_survive() -> None:
    out = tracing.scrub_transaction(_poisoned_event("transaction"), None)
    assert out["transaction"] == "node"  # name was not in the fixed set
    assert out["contexts"]["trace"]["data"] == {"walk_id": WALK_ID}
    assert out["contexts"]["trace"]["op"] == "span"
    assert out["tags"] == {"walk_id": WALK_ID}
    (span,) = out["spans"]
    assert span["description"] == "node"
    assert span["data"] == {"tokens_in": 5, "latency_ms": 10.5,
                            "reason_codes": ["new_number"], "walk_id": WALK_ID}  # fmt: skip
    assert span["tags"] == {"status": "ok"}
    assert "origin" not in span


def test_known_names_survive() -> None:
    ev = _poisoned_event("transaction")
    ev["transaction"] = "journal"
    ev["spans"][0]["description"] = "ground_check"
    ev["spans"][0]["op"] = "graph.node"
    out = tracing.scrub_transaction(ev, None)
    assert out["transaction"] == "journal"
    assert out["spans"][0]["description"] == "ground_check"
    assert out["spans"][0]["op"] == "graph.node"


def test_clean_attrs_drops_text_values() -> None:
    m = MARKER
    got = tracing.clean_attrs({
        "walk_id": m, "mission_id": f"{m}-id", "node": m, "model": m, "backend": m,
        "status": m, "origin": m, "reason_codes": [m], "tokens_in": m, "latency_ms": m,
        "check_passed": m, "unsupported_details": [m], "transcript": m, "entry": m,
        "facts_count": -1, "audio_duration_s": float("nan"), "regen_count": True,
    })  # fmt: skip
    assert got == {}
    ok = tracing.clean_attrs({
        "walk_id": WALK_ID, "node": "transcribe", "model": "gemma", "backend": WriterBackend.LLAMA,
        "status": WalkStatus.DONE, "reason_codes": (ReasonCode.NEW_NUMBER, "unsupported_detail"),
        "tokens_in": 10, "tokens_out": 3, "latency_ms": 12.3456, "check_passed": False,
        "facts_count": 2, "verified_count": 1, "chunk_count": 1, "retry_count": 0,
        "audio_duration_s": 20, "regen_count": 1, "origin": "template",
    })  # fmt: skip
    assert ok == {
        "walk_id": WALK_ID, "node": "transcribe", "model": "gemma", "backend": "llama",
        "status": "done", "reason_codes": ["new_number", "unsupported_detail"],
        "tokens_in": 10, "tokens_out": 3, "latency_ms": 12.346, "check_passed": False,
        "facts_count": 2, "verified_count": 1, "chunk_count": 1, "retry_count": 0,
        "audio_duration_s": 20.0, "regen_count": 1, "origin": "template",
    }  # fmt: skip
    assert "unsupported_details" not in tracing.ALLOWED_ATTRS


# ------------------------------------------------------------------ through the real SDK


def _enable(**kw: Any) -> Any:
    cap = CapturingTransport.make()
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        sair_env="eval",
        sentry_dsn=SecretStr(FAKE_DSN),
        sair_data_dir=SYNTHETIC,
        **kw,
    )
    assert tracing.init_tracing(s, transport=cap) is True
    return cap


def _payloads(cap: Any) -> list[dict[str, Any]]:
    tracing.shutdown_tracing()  # flush
    return [item.payload.json for env in cap.envelopes for item in env.items]


def test_sdk_capture_paths_never_carry_the_marker() -> None:
    import sentry_sdk

    cap = _enable()
    sentry_sdk.set_user({"id": MARKER})
    sentry_sdk.set_tag("note", MARKER)
    sentry_sdk.set_extra("transcript", MARKER)
    sentry_sdk.set_context("walk", {"entry": MARKER})
    sentry_sdk.add_breadcrumb(message=MARKER, data={"x": MARKER})
    sentry_sdk.capture_message(f"message {MARKER}")

    def failing(transcript: str) -> None:
        local_entry = f"entry {transcript}"  # noqa: F841 (frame var must not be sent)
        raise RuntimeError(f"cannot parse {transcript}")

    with (
        pytest.raises(RuntimeError),
        tracing.graph_transaction("journal", walk_id=WALK_ID),
        tracing.node_span("write_entry"),
    ):
        tracing.set_attrs(regen_count=1, unsupported_details=[MARKER], node=MARKER,
                          reason_codes=[ReasonCode.UNSUPPORTED_DETAIL, MARKER])  # fmt: skip
        tracing.llm_span(MARKER, tokens_in=7, tokens_out=9, latency_ms=12.0)
        span = sentry_sdk.get_current_span()
        assert span is not None
        span.set_data("prompt", MARKER)  # bypassing set_attrs: scrubber still drops it
        span.set_tag("t", MARKER)
        with sentry_sdk.start_span(op=MARKER, name=MARKER):
            pass
        failing(MARKER)

    payloads = _payloads(cap)
    kinds = sorted(p.get("type", "event") for p in payloads)
    assert kinds == ["event", "event", "transaction"]
    blob = _dump(payloads)
    assert MARKER not in blob
    assert "unsupported_details" not in blob
    (tx,) = [p for p in payloads if p.get("type") == "transaction"]
    assert tx["transaction"] == "journal"
    assert tx["contexts"]["trace"]["data"] == {"walk_id": WALK_ID}
    by_op = {s["op"]: s for s in tx["spans"]}
    assert by_op["graph.node"]["description"] == "write_entry"
    assert by_op["graph.node"]["data"] == {
        "node": "write_entry", "regen_count": 1, "reason_codes": ["unsupported_detail"]
    }  # fmt: skip
    assert by_op["llm"]["data"] == {"tokens_in": 7, "tokens_out": 9, "latency_ms": 12.0}
    assert by_op["span"]["description"] == "node"
    (err,) = [p for p in payloads if "exception" in p]
    assert [v["type"] for v in err["exception"]["values"]] == ["RuntimeError"]


def test_mission_graph_emits_one_transaction_with_a_span_per_node(tmp_path: Path) -> None:
    from hub.graphs import mission as M
    from tests.unit._graphfix import GROUP, mission_deps

    cap = _enable()
    deps = mission_deps(tmp_path)
    deps.span = tracing.node_span
    conn = deps.connect()
    try:  # 3 done walks so the planner (not a seed) runs
        for i in range(3):
            wid = f"{i:032x}"
            db.insert_walk(conn, walk_id=wid, group=GROUP,
                           recorded_at=dt.datetime(2026, 10, 1 + i, 13, tzinfo=dt.UTC))  # fmt: skip
            db.update_walk_status(conn, wid, WalkStatus.DONE)
    finally:
        conn.close()
    with tracing.graph_transaction("mission"):
        out = M.plan_mission(GROUP, dt.date(2026, 10, 11), deps)
    assert out is not None
    (tx,) = [p for p in _payloads(cap) if p.get("type") == "transaction"]
    names = [s["description"] for s in tx["spans"] if s["op"] == "graph.node"]
    assert names[0] == "mission_recall" and "mission_cache" in names
    assert all(n in tracing.NODE_NAMES for n in names)
    assert all(s["parent_span_id"] == tx["contexts"]["trace"]["span_id"] for s in tx["spans"])


def test_init_uses_hardened_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defense in depth: the SDK must not collect what the scrubber would have to remove."""
    import sentry_sdk

    seen: dict[str, Any] = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: seen.update(kw))
    s = Settings(  # type: ignore[call-arg]
        _env_file=None, sair_env="eval", sentry_dsn=SecretStr(FAKE_DSN), sair_data_dir=SYNTHETIC
    )
    assert tracing.init_tracing(s) is True
    expected = {
        "default_integrations": False, "auto_enabling_integrations": False, "integrations": [],
        "include_local_variables": False, "max_request_body_size": "never",
        "send_default_pii": False, "include_source_context": False, "attach_stacktrace": False,
        "max_breadcrumbs": 0, "server_name": "", "auto_session_tracking": False,
        "send_client_reports": False, "enable_logs": False, "environment": "eval",
    }  # fmt: skip
    assert {k: seen.get(k) for k in expected} == expected
    assert seen["before_send"] is tracing.scrub_event
    assert seen["before_send_transaction"] is tracing.scrub_transaction
    assert seen["before_breadcrumb"]({"message": MARKER}, {}) is None
    assert "transport" not in seen
    tracing._STATE.sdk = None  # init was faked; nothing to close
