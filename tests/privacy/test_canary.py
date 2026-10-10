"""Privacy canary (PRD §11.1, S5; NFR-P2, NFR-P3, NFR-P4, NFR-P7). Stub models, synthetic data.

A unique marker goes through every input path:

(a) PWA upload: multipart WebM whose filename and form fields carry the marker;
(b) Shortcut upload: multipart M4A (``audio/mp4``) with the marker in a custom header, the
    User-Agent and the query string;
(c) text fixture: ``run_journal`` driven directly with a stub transcript;
(d) a Pydantic ValidationError: invalid ``POST /feedback`` bodies (owner token) and a writer
    returning invalid JSON / invalid enum values (``parse_writer_output`` -> template);
(e) exception paths: a writer raising ``RuntimeError(marker)``, a transcriber raising one, and
    a route whose query helper raises one (500 path).

Stub transcripts, writer facts/entries contain the marker, so content really flows. Sinks that
must stay marker-free: every Sentry envelope (fake transport), every ntfy request (URL, headers,
body), every log record at DEBUG (all loggers, formatted with exc_info), stdout/stderr (where
structlog writes), every HTTP response body except the product content endpoints
(``ALLOWED_CONTENT_ROUTES``) and every response header, the DB outside walks.transcript, facts,
entries (PRD §11.1), file names, the hub log dir. Outbound connections must be loopback only.

missions.text and feedback.note are content columns in PRD §9.4 but are not in the §11.1
allow-list. The main canary uses a marker-free planner output; a marker mission is tested
separately (sinks only). A feedback note with the marker is tested against the strict §11.1
list (see ``test_feedback_note_db_scope``).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import sentry_sdk
from pydantic import ValidationError

from hub import db, tracing
from hub.api import queries
from hub.graphs.journal import run_journal
from hub.graphs.mission import plan_mission
from hub.schemas import parse_writer_output
from hub.settings import Settings
from tests.privacy.conftest import (
    GROUP,
    MARKER,
    CanaryEnv,
    bearer,
    db_marker_columns,
)
from tests.unit._graphfix import FIXTURES

REPO = Path(__file__).resolve().parents[2]

# --- synthetic content carrying the marker -------------------------------------------------

SPAN = f"We saw a {MARKER} bird sitting on the wall near the gate"
TRANSCRIPT = f"We walked to the park. {SPAN}. It was cool and windy."
TRANSCRIPT_HEALTH = f"{TRANSCRIPT} Dad felt a bit dizzy on the way back."  # FR-O4 ping
GOOD_ENTRY = json.dumps(
    {
        "facts": [
            {"span": SPAN, "kind": "animal", "subject": f"{MARKER} bird", "certainty": "sure",
             "place": "near the gate"},
        ],
        "mission_result": "not_mentioned",
        "entry": f"Mom and Dad saw a {MARKER} bird sitting on the wall near the gate.",
    }
)  # fmt: skip
INVALID_JSON = '{"facts": [{"span": "' + MARKER + '", "kind": "zeb'  # truncated
INVALID_ENUMS = json.dumps(
    {
        "facts": [{"span": SPAN, "kind": MARKER, "subject": MARKER, "certainty": MARKER}],
        "mission_result": MARKER,
        "entry": MARKER,
    }
)
CLEAN_PLAN = json.dumps(
    {"type": "sense", "references": [], "text": "Listen for one bird you can hear from the path."}
)


def _stage(env: CanaryEnv, walk_id: str, fixture: str = "en_20s.webm") -> Path:
    src = FIXTURES / fixture
    dst = env.deps.settings.tmp_dir / f"{walk_id}{src.suffix}"
    shutil.copyfile(src, dst)
    return dst


def _repo_log_names() -> set[str]:
    """Names only (never content) of files in ./logs, to prove the hub creates none there."""
    d = REPO / "logs"
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def _write_canary_meta(env: CanaryEnv, passed: bool) -> None:
    conn = env.deps.connect()
    try:
        db.set_meta(conn, "last_canary_result", "passed" if passed else "failed")
        db.set_meta(conn, "last_canary_date", dt.date.today().isoformat())
    finally:
        conn.close()


def _sink_violations(env: CanaryEnv, out: str, err: str) -> list[str]:
    v = env.sentry_hits() + env.ntfy_hits() + env.log_hits() + env.response_hits()
    v += env.connection_hits() + env.file_hits()
    if MARKER in out:
        v.append("stdout")
    if MARKER in err:
        v.append("stderr")
    if env.deps.settings.log_dir.exists():
        v.append("hub log dir was created")
    return v


# --- the canary ---------------------------------------------------------------------------


def test_canary_all_paths_all_sinks(
    canary_env: CanaryEnv, capfd: pytest.CaptureFixture[str]
) -> None:
    env = canary_env
    m = env.models
    m.planner = CLEAN_PLAN
    logs_before = _repo_log_names()

    # (a) PWA upload: WebM, marker in the filename and in form fields
    m.transcript = TRANSCRIPT
    m.set_writer(GOOD_ENTRY)
    walk_a = env.post_walk(
        "en_20s.webm", f"{MARKER}.webm", "audio/webm",
        data={"group": MARKER, "note": MARKER, "mission_id": MARKER},
    )  # fmt: skip
    env.wait_done(walk_a)

    # (b) Shortcut upload: M4A as audio/mp4, marker in a custom header, UA and query string
    m.transcript = TRANSCRIPT_HEALTH
    m.set_writer(GOOD_ENTRY)
    walk_b = env.post_walk(
        "en_20s.m4a", "recording.m4a", "audio/mp4",
        headers={"X-Sair-Note": MARKER, "User-Agent": f"Shortcuts {MARKER}"},
        query=f"?note={MARKER}&group={MARKER}",
    )  # fmt: skip
    env.wait_done(walk_b)

    # (c) text fixture: run_journal directly (traced like the app run)
    m.transcript = TRANSCRIPT
    m.set_writer(GOOD_ENTRY)
    walk_c = db.new_id()
    with tracing.graph_transaction("journal", walk_id=walk_c):
        out_c = run_journal(walk_c, _stage(env, walk_c), GROUP, None, env.deps.journal)
    assert out_c.status.value == "done"
    env.wait_idle()

    # (d1) ValidationError: invalid feedback bodies with the marker (owner token)
    owner = bearer("owner")
    for body in (
        {"correct": MARKER, "note": MARKER},
        {"correct": True, "note": "x" * 600 + MARKER},
        {"correct": True, "note": MARKER, MARKER: MARKER},
    ):
        r = env.request("POST", f"/feedback/{walk_a}", "/feedback/{walk_id}", headers=owner,
                        json=body)  # fmt: skip
        assert r.status_code == 422
    r = env.request("POST", f"/feedback/{walk_a}", "/feedback/{walk_id}",
                    headers={**owner, "Content-Type": "application/json"},
                    content=MARKER.encode())  # fmt: skip
    assert r.status_code == 422
    r = env.request("POST", f"/feedback/{MARKER}", "/feedback/{walk_id}", headers=owner,
                    json={"correct": True})  # fmt: skip
    assert r.status_code == 404

    # (d2) ValidationError inside the graph: invalid writer JSON, then invalid enums -> template
    m.transcript = TRANSCRIPT
    m.set_writer(INVALID_JSON, INVALID_ENUMS)
    walk_d = env.post_walk("en_20s.webm", "note.webm", "audio/webm")
    env.wait_done(walk_d)
    assert m.writer_calls == 2
    for raw in (INVALID_JSON, INVALID_ENUMS):
        parsed = parse_writer_output(raw)
        assert parsed.value is None
        assert MARKER not in repr(parsed.reasons)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, sair_host=MARKER)  # type: ignore[call-arg]
    # str/repr are what reaches a log or a message (hide_input_in_errors); .errors() is data
    assert MARKER not in str(exc_info.value) and MARKER not in repr(exc_info.value)

    # (e1) exception: the writer raises RuntimeError(marker), via the app
    m.transcript = TRANSCRIPT
    m.set_writer(RuntimeError(MARKER))
    walk_e1 = env.post_walk("en_20s.webm", "note.webm", "audio/webm")
    env.wait_idle()
    assert env.walk_status(walk_e1) == "done"  # FR-J5/J7: a writer bug falls back to the template

    # (e2) exception: the writer raises, via run_journal in a traced graph run
    walk_e2 = db.new_id()
    traced = tracing.graph_transaction("journal", walk_id=walk_e2)
    with traced:
        out_e2 = run_journal(walk_e2, _stage(env, walk_e2), GROUP, None, env.deps.journal)
    assert out_e2.writer == "template"  # FR-J5/J7: the exception becomes a template entry

    # (e3) exception: the transcriber raises RuntimeError(marker)
    m.transcript = RuntimeError(MARKER)
    walk_e3 = env.post_walk("en_20s.m4a", "recording.m4a", "audio/mp4")
    env.wait_idle()
    assert env.walk_status(walk_e3) == "received"

    # (e4) exception inside a route: 500 without exception text
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError(MARKER)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(queries, "status_response", boom)
        r = env.request("GET", "/status", "/status", headers=owner)
    assert r.status_code == 500

    # (e5) Sentry API used directly with the marker everywhere (scrubbers must drop it)
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("note", MARKER)
        scope.set_extra("transcript", MARKER)
        scope.set_user({"id": MARKER, "username": MARKER})
        scope.set_context("walk", {"entry": MARKER})
        sentry_sdk.add_breadcrumb(message=MARKER)
        sentry_sdk.capture_message(MARKER)
        try:
            raise ValueError(MARKER)
        except ValueError as exc:
            sentry_sdk.capture_exception(exc)
    with tracing.graph_transaction("journal", walk_id=walk_a), tracing.node_span(MARKER):
        tracing.set_attrs(node=MARKER, model=MARKER, walk_id=MARKER, transcript=MARKER,
                          unsupported_details=[MARKER], reason_codes=[MARKER])  # fmt: skip
        tracing.llm_span(MARKER, tokens_in=1, tokens_out=1, latency_ms=1.0)

    # Product reads (allowed content) + every other read (must be clean)
    fam = bearer("mom")
    for wid in (walk_a, walk_b, walk_c, walk_d, walk_e1, walk_e3):
        env.request("GET", f"/walk/{wid}", "/walk/{walk_id}", headers=fam)
    env.request("GET", "/diary", "/diary", headers=fam)
    env.request("GET", "/mission/today", "/mission/today", headers=fam)
    env.request("GET", f"/walk/{MARKER}", "/walk/{walk_id}", headers=fam)  # 404
    env.request("GET", f"/diary?group={MARKER}", "/diary", headers=fam)  # 400
    env.request("GET", f"/healthz?q={MARKER}", "/healthz")
    env.request("GET", "/status", "/status", headers={"Authorization": f"Bearer {MARKER}"})
    env.request("GET", f"/{MARKER}", "unknown")
    env.request("POST", "/walk", "/walk", headers=fam, content=MARKER.encode())  # 400
    env.request("GET", "/status", "/status", headers=owner)
    env.wait_idle()
    tracing.shutdown_tracing()  # flush the client before reading the fake transport

    out, err = capfd.readouterr()

    # Positive controls: content flowed, and every sink was live.
    walk_view = next(r for r in env.responses if r.route == "GET /walk/{walk_id}")
    assert MARKER.encode() in walk_view.body
    hits = set(db_marker_columns(env.deps.settings.db_path))
    assert ("walks", "transcript") in hits and ("entries", "text") in hits
    assert ("facts", "span") in hits and ("facts", "subject") in hits
    assert env.sentry.envelopes, "fake Sentry transport received nothing"
    assert any(b'"type":"transaction"' in e or b'"transaction"' in e for e in env.sentry.envelopes)
    assert len(env.ntfy) >= 4, "walk_done for a, b, c, d"
    assert any(b"Check in with" in body for _, _, body in env.ntfy), "FR-O4 ping (walk b)"
    assert '"walk_done"' in out, "structlog output not captured"
    assert env.logs.records, "stdlib log capture saw nothing"

    violations = _sink_violations(env, out, err) + env.db_hits()
    if _repo_log_names() - logs_before:
        violations.append("new file in ./logs")

    _write_canary_meta(env, not violations)
    conn = env.deps.connect()
    try:
        assert db.get_meta(conn, "last_canary_result") == ("failed" if violations else "passed")
    finally:
        conn.close()
    assert not violations, violations


def test_marker_mission_never_reaches_sinks(
    canary_env: CanaryEnv, capfd: pytest.CaptureFixture[str]
) -> None:
    """A planner output with the marker (references a verified marker fact) is served by
    ``/mission/today`` (product) and may be stored in missions.text (PRD §9.4 content column,
    an open PRD question) but reaches no other sink."""
    env = canary_env
    m = env.models
    m.transcript = TRANSCRIPT
    m.set_writer(GOOD_ENTRY)
    m.planner = CLEAN_PLAN
    for _ in range(3):  # SEED_WALKS: the planner runs from the 4th walk on
        wid = db.new_id()
        run_journal(wid, _stage(env, wid), GROUP, None, env.deps.journal)
    env.wait_idle()
    conn = env.deps.connect()
    try:
        fact_id = str(conn.execute("SELECT id FROM facts WHERE verified = 1 LIMIT 1").fetchone()[0])
    finally:
        conn.close()
    m.planner = json.dumps(
        {"type": "return", "references": [fact_id],
         "text": f"Look again for the {MARKER} bird near the gate and see what it does."}
    )  # fmt: skip
    today = env.deps.today()
    with tracing.graph_transaction("mission"):
        outcome = plan_mission(GROUP, today, env.deps.mission, replace=True)
    assert outcome is not None
    r = env.request("GET", "/mission/today", "/mission/today", headers=bearer("mom"))
    assert r.status_code == 200
    env.request("GET", "/status", "/status", headers=bearer("owner"))
    env.wait_idle()
    tracing.shutdown_tracing()
    out, err = capfd.readouterr()

    violations = _sink_violations(env, out, err)
    violations += env.db_hits(extra_allowed=frozenset({("missions", "text")}))
    assert env.sentry.envelopes
    assert not violations, violations


def test_feedback_note_db_scope(canary_env: CanaryEnv, capfd: pytest.CaptureFixture[str]) -> None:
    """FR-O3 + PRD §11.1: the owner's feedback note with the marker. The note must not reach any
    sink, and per §11.1 the DB may hold the marker only in walks.transcript/facts/entries.
    ``hub/api/routes.py`` stores ``body.note`` in ``feedback.note``; if that column holds the
    marker this test fails (PRD §9.4 lists the column, §11.1 does not allow it: PRD question)."""
    env = canary_env
    m = env.models
    m.transcript = TRANSCRIPT
    m.set_writer(GOOD_ENTRY)
    m.planner = CLEAN_PLAN
    walk = env.post_walk("en_20s.webm", "note.webm", "audio/webm")
    env.wait_done(walk)
    note = {"correct": False, "note": f"the {MARKER} bird was not there"}
    r = env.request("POST", f"/feedback/{walk}", "/feedback/{walk_id}", headers=bearer("owner"),
                    json=note)  # fmt: skip
    assert r.status_code == 200
    env.request("GET", "/status", "/status", headers=bearer("owner"))
    env.wait_idle()
    tracing.shutdown_tracing()
    out, err = capfd.readouterr()

    sinks = _sink_violations(env, out, err)
    assert not sinks, sinks
    assert not env.db_hits(), env.db_hits()


def test_funnel_off() -> None:
    """NFR-P2: Tailscale Funnel stays off (the owner runs this where tailscale exists)."""
    exe = shutil.which("tailscale")
    if exe is None:
        pytest.skip("tailscale not installed")
    res = subprocess.run([exe, "serve", "status", "--json"], capture_output=True, text=True,
                         timeout=10, check=False)  # fmt: skip
    assert res.returncode == 0, "tailscale serve status failed"
    status = json.loads(res.stdout or "{}")
    assert not status.get("AllowFunnel"), "Funnel is on"
    res = subprocess.run([exe, "funnel", "status"], capture_output=True, text=True, timeout=10,
                         check=False)  # fmt: skip
    assert "funnel on" not in (res.stdout + res.stderr).lower()
