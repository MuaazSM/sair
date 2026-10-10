"""Journal graph (PRD J3, FR-J1..J9, FR-O1/O4, §8): voice note -> grounded diary entry.

Nodes (one function each; one metric row and, in dev/eval, one span each)::

    START -> ingest -> transcribe -> write_entry -> ground_check -> store
                                                     |   ^   \\
                                                     v   |    -> template -> store
                                                    regen
    store -> plan_next_mission -> notify -> END

    START routes to write_entry when a transcript is already stored (writer retry) and to
    template when the writer retry window has expired.

FR-J6 audio lifetime (the rule this module exists to keep):

* ``ingest`` decodes into ``<tmp>/work-<walk_id>/``. A permanent failure (undecodable, too
  short/long, silent) deletes the raw upload and the work dir before the node returns; the
  walk becomes ``lost`` with the code.
* ``transcribe`` deletes the work dir (WAV + chunks) in ``finally`` always, and the raw upload
  in the same ``finally`` on success or on a permanent failure. On a transient failure the raw
  upload is kept for :func:`retry_pending`; the walk stays ``received`` / TRANSCRIBE_TRANSIENT.
* :func:`retry_pending` retries with backoff (30, 60, 120, 240, 240 ... s) until
  TRANSIENT_RETRY_WINDOW_S after the upload, then deletes the audio and marks the walk
  ``lost`` / RETRY_WINDOW_EXPIRED. :func:`sweep_stale_audio` (startup) deletes temp audio
  older than 10 minutes and marks such walks ``lost`` / STALE_AUDIO_SWEPT.
* Graph state carries ``str`` paths only, never bytes. No checkpointer is configured.

Writer transient (decision): the transcript is stored before the writer runs and the audio is
already gone, so a transient writer failure leaves the walk ``received`` / WRITER_TRANSIENT and
:func:`retry_pending` re-runs the graph from ``write_entry`` (transcript from the DB) with the
same backoff and window. When the window expires the walk is stored with the no-fact template,
status ``done``, reason WRITER_TRANSIENT (the walk is not lost: the transcript is the evidence,
FR-J7). A permanent writer error, or any writer error during the one regeneration, goes to the
template (from the best check result so far) and records WRITER_PERMANENT / WRITER_TRANSIENT.

The transcript and entry live in the in-process state only for this run; they are never
logged, and :class:`JournalOutcome` carries codes and counts only (NFR-P4).
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

import structlog
import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from hub import db
from hub.graphs.common import (
    REPO_ROOT,
    Clock,
    ConnectFn,
    SpanFn,
    TranscriberLike,
    WriterLike,
    delete_path,
    find_raw_audio,
    mission_text,
    null_span,
    open_conn,
    pending_walks,
    repo_path,
    set_duration,
    timed_node,
    utc_now,
    work_dir_for,
)
from hub.grounding import CheckResult, build_template_entry, check_entry, template_output
from hub.models import audio
from hub.models.errors import Permanent, Transient
from hub.prompts.builders import (
    ChatMessage,
    MissionText,
    Transcript,
    WalkerGroup,
    build_writer_prompt,
)
from hub.schemas import (
    MissionResult,
    ReasonCode,
    WalkStatus,
    WriterBackend,
    WriterOutput,
    parse_writer_output,
)
from hub.settings import Settings

__all__ = [
    "BACKOFF_S",
    "STALE_AUDIO_S",
    "JournalDeps",
    "JournalOutcome",
    "JournalState",
    "MissionTrigger",
    "Notifier",
    "RetryState",
    "build_journal_graph",
    "ground_check",
    "grounding_summary",
    "has_health_keyword",
    "ingest",
    "notify",
    "plan_next_mission",
    "regen",
    "retry_pending",
    "run_journal",
    "store",
    "sweep_stale_audio",
    "template",
    "transcribe",
    "write_entry",
]

log = structlog.get_logger("hub.graphs.journal")

BACKOFF_S: tuple[float, ...] = (30.0, 60.0, 120.0, 240.0)  # then 240 s until the window ends
STALE_AUDIO_S = 600.0  # FR-J6 startup sweep: older than 10 minutes
HEALTH_KEYWORDS_PATH = REPO_ROOT / "data" / "health_keywords.yaml"

WriterLabel = Literal["gemma", "llama", "template"]


# ------------------------------------------------------------------ protocols / deps


class Notifier(Protocol):
    """Metadata-only notifications (FR-O1, FR-O4). Implemented by ``hub.notify``."""

    def walk_done(self, group: str, facts: int, mission_result: MissionResult) -> None: ...

    def health_ping(self, group: str) -> None: ...


MissionTrigger = Callable[[str], None]
"""Schedules mission planning for a group (FR-M7). Must return quickly (never plan inline)."""


@dataclass
class RetryState:
    """In-memory retry bookkeeping for :func:`retry_pending` (attempts and next due time).

    Lost on restart by design: a restarted hub retries pending walks immediately and the
    window is measured from ``walks.recorded_at`` (the upload time), which is in the DB.
    """

    attempts: dict[str, int] = field(default_factory=dict)
    next_at: dict[str, dt.datetime] = field(default_factory=dict)
    in_flight: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def schedule(self, walk_id: str, now: dt.datetime) -> None:
        n = self.attempts.get(walk_id, 0)
        delay = BACKOFF_S[min(n, len(BACKOFF_S) - 1)]
        self.attempts[walk_id] = n + 1
        self.next_at[walk_id] = now + dt.timedelta(seconds=delay)

    def forget(self, walk_id: str) -> None:
        self.attempts.pop(walk_id, None)
        self.next_at.pop(walk_id, None)


@dataclass
class JournalDeps:
    settings: Settings
    connect: ConnectFn  # returns a NEW connection each call (closed by the graph)
    ollama: TranscriberLike
    llama: WriterLike | None
    notifier: Notifier
    mission_trigger: MissionTrigger
    clock: Clock = utc_now
    span: SpanFn = null_span
    retry: RetryState = field(default_factory=RetryState)


# ------------------------------------------------------------------ state / outcome


class JournalState(TypedDict, total=False):
    # identity (set by the caller)
    walk_id: str
    group: str
    mission_id: str | None
    raw_path: str | None  # upload path (str, never bytes)
    resume: Literal["write", "template"] | None
    forced_reason: ReasonCode | None
    # ingest
    wav_path: str | None
    duration_s: float | None
    # transcribe / write (text lives here only for this in-process run; never logged)
    transcript: str | None
    mission_text: str | None
    raw_output: str | None
    writer_error: Literal["transient", "permanent"] | None
    # check
    output: WriterOutput | None
    check: CheckResult | None
    best_check: CheckResult | None
    check_passed: bool | None
    check_reasons: tuple[ReasonCode, ...]
    regen_count: int
    # final
    final_output: WriterOutput | None
    final_passed: bool | None
    writer: WriterLabel | None
    final_reason: ReasonCode | None
    status: WalkStatus
    reason: ReasonCode | None
    facts: int
    mission_result: MissionResult | None
    latency_ms: dict[str, float]


@dataclass(frozen=True)
class JournalOutcome:
    """What a run did: IDs, codes, counts and timings only (safe to log)."""

    walk_id: str
    status: WalkStatus
    reason: ReasonCode | None = None
    writer: str | None = None
    check_passed: bool | None = None
    regen_count: int = 0
    facts: int = 0
    mission_result: MissionResult | None = None
    latency_ms: Mapping[str, float] = field(default_factory=dict)

    @property
    def pending(self) -> bool:
        return self.status is WalkStatus.RECEIVED


# ------------------------------------------------------------------ health keywords


@functools.cache
def _health_patterns() -> tuple[re.Pattern[str] | None, tuple[str, ...]]:
    data = yaml.safe_load(HEALTH_KEYWORDS_PATH.read_text(encoding="utf-8"))
    kw = data.get("keywords", {}) if isinstance(data, dict) else {}
    latin: list[str] = []
    for key in ("en", "hinglish"):
        latin += [unicodedata.normalize("NFC", str(w)).casefold() for w in kw.get(key) or []]
    deva = tuple(unicodedata.normalize("NFC", str(w)) for w in kw.get("devanagari") or [])
    latin.sort(key=len, reverse=True)
    pattern = (
        re.compile(r"(?<![0-9a-z])(?:" + "|".join(re.escape(w) for w in latin) + r")(?![0-9a-z])")
        if latin
        else None
    )
    return pattern, deva


def has_health_keyword(transcript: str) -> bool:
    """FR-O4: any English, Hinglish or Devanagari health keyword in the note.

    Latin keywords match case-insensitively on word boundaries; Devanagari as substrings.
    Pure; returns a bool only (the caller sends a text-free ping).
    """
    text = unicodedata.normalize("NFC", transcript)
    low = text.casefold().replace("’", "'")
    pattern, deva = _health_patterns()
    if pattern is not None and pattern.search(low):
        return True
    return any(w in text for w in deva)


# ------------------------------------------------------------------ helpers


def _mark(deps: JournalDeps, walk_id: str, status: WalkStatus, reason: ReasonCode | None) -> None:
    with open_conn(deps.connect) as conn:
        db.update_walk_status(conn, walk_id, status, reason)
    if status is not WalkStatus.RECEIVED:
        with deps.retry.lock:
            deps.retry.forget(walk_id)


@functools.cache
def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _call_writer(
    state: JournalState, deps: JournalDeps, regen_codes: tuple[ReasonCode, ...]
) -> dict[str, Any]:
    """One writer call. Returns raw output (or ``writer_error``) plus token counts."""
    s = deps.settings
    transcript = state.get("transcript") or ""
    mtext = state.get("mission_text")
    messages: list[ChatMessage] = build_writer_prompt(
        WalkerGroup(state["group"]),
        MissionText(mtext) if mtext else None,
        Transcript(transcript),
        regen=regen_codes,
    )
    try:
        if s.writer_backend is WriterBackend.LLAMA:
            if deps.llama is None:
                raise Permanent(ReasonCode.WRITER_PERMANENT, "llama_server")
            grammar = _read_text(str(repo_path(s.writer_grammar)))
            raw, stats = deps.llama.complete_chat(
                messages, grammar=grammar, max_tokens=s.writer_max_tokens
            )
        else:
            schema = json.loads(_read_text(str(repo_path(s.writer_json_schema))))
            raw, stats = deps.ollama.chat_json(messages, schema, role="writer")
    except Transient:
        return {"raw_output": None, "writer_error": "transient"}
    except Permanent:
        return {"raw_output": None, "writer_error": "permanent"}
    return {
        "raw_output": raw,
        "writer_error": None,
        "_tokens_in": stats.prompt_tokens,
        "_tokens_out": stats.completion_tokens,
    }


def grounding_summary(
    output: WriterOutput, check: CheckResult, passed: bool
) -> db.GroundingSummary:
    """Adapter CheckResult -> db.GroundingSummary (verified flags aligned by fact index)."""
    verified_ids = {id(f) for f in check.verified_facts}
    return db.GroundingSummary(
        passed=passed,
        fact_verified=tuple(id(f) in verified_ids for f in output.facts),
        mission_result=check.mission_result,
        mission_span=check.mission_span
        if check.mission_result is not MissionResult.NOT_MENTIONED
        else None,
    )


# ------------------------------------------------------------------ nodes


def ingest(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-J1/J2: decode to 16 kHz mono WAV, bounds 1-90 s, VAD gate. Permanent -> delete."""
    walk_id = state["walk_id"]
    raw = state.get("raw_path")
    work = work_dir_for(deps.settings.tmp_dir, walk_id)
    try:
        if not raw:
            raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio")
        wav = audio.decode_to_wav16k(Path(raw), work)
        dur = audio.duration_s(wav)
        audio.bounds_check(dur, deps.settings)
        audio.vad_gate(wav, deps.settings)
    except Permanent as exc:
        delete_path(work)
        delete_path(raw)
        code = exc.code or ReasonCode.AUDIO_UNDECODABLE
        _mark(deps, walk_id, WalkStatus.LOST, code)
        log.info("walk_lost", walk_id=walk_id, reason=code.value, node="ingest")
        return {"status": WalkStatus.LOST, "reason": code, "wav_path": None}
    except Exception as exc:  # unexpected: keep the upload for the bounded retry window
        delete_path(work)
        _mark(deps, walk_id, WalkStatus.RECEIVED, ReasonCode.TRANSCRIBE_TRANSIENT)
        with deps.retry.lock:
            deps.retry.schedule(walk_id, deps.clock())
        log.warning("ingest_error", walk_id=walk_id, exc_type=type(exc).__name__)
        return {"status": WalkStatus.RECEIVED, "reason": ReasonCode.TRANSCRIBE_TRANSIENT}
    with open_conn(deps.connect) as conn:
        set_duration(conn, walk_id, round(dur, 2))
    return {"wav_path": str(wav), "duration_s": dur, "status": WalkStatus.RECEIVED, "reason": None}


def transcribe(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-J2 + FR-J6: transcribe (temperature 0); ``finally`` deletes audio unless transient."""
    walk_id = state["walk_id"]
    wav = state.get("wav_path")
    work = work_dir_for(deps.settings.tmp_dir, walk_id)
    outcome: Literal["ok", "transient", "permanent"] = "transient"  # BaseException keeps audio
    code: ReasonCode | None = None
    text = ""
    tokens: tuple[int | None, int | None] = (None, None)
    try:
        if not wav:
            raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio")
        text, stats = deps.ollama.transcribe_with_stats(Path(wav))
        tokens = (stats.prompt_tokens, stats.completion_tokens)
        outcome = "ok"
    except Permanent as exc:
        outcome, code = "permanent", exc.code or ReasonCode.TRANSCRIBE_PERMANENT
    except Transient:
        outcome, code = "transient", ReasonCode.TRANSCRIBE_TRANSIENT
    except Exception as exc:
        outcome, code = "transient", ReasonCode.TRANSCRIBE_TRANSIENT
        log.warning("transcribe_error", walk_id=walk_id, exc_type=type(exc).__name__)
    finally:
        delete_path(work)  # decoded WAV and chunks: always (re-decoded on a retry)
        if outcome != "transient":
            delete_path(state.get("raw_path"))  # FR-J6: success or permanent
    base = {"wav_path": None, "_tokens_in": tokens[0], "_tokens_out": tokens[1]}
    if outcome == "transient":
        _mark(deps, walk_id, WalkStatus.RECEIVED, ReasonCode.TRANSCRIBE_TRANSIENT)
        now = deps.clock()
        with deps.retry.lock:
            deps.retry.schedule(walk_id, now)
        log.info("walk_retry_pending", walk_id=walk_id, reason="transcribe_transient")
        return {**base, "status": WalkStatus.RECEIVED, "reason": code}
    if outcome == "permanent" or not text.strip():
        lost = code if outcome == "permanent" and code else ReasonCode.TRANSCRIPT_EMPTY
        _mark(deps, walk_id, WalkStatus.LOST, lost)
        log.info("walk_lost", walk_id=walk_id, reason=lost.value, node="transcribe")
        return {**base, "status": WalkStatus.LOST, "reason": lost}
    with open_conn(deps.connect) as conn:
        db.set_transcript(conn, walk_id, text.strip())  # FR-J7: the evidence
        mtext = mission_text(conn, state.get("mission_id"))
    return {**base, "transcript": text.strip(), "mission_text": mtext, "reason": None}


def write_entry(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-J3/J4: writer gets only group, today's mission and the transcript."""
    upd = _call_writer(state, deps, ())
    if upd.get("writer_error") == "transient":
        walk_id = state["walk_id"]
        _mark(deps, walk_id, WalkStatus.RECEIVED, ReasonCode.WRITER_TRANSIENT)
        with deps.retry.lock:
            deps.retry.schedule(walk_id, deps.clock())
        log.info("walk_retry_pending", walk_id=walk_id, reason="writer_transient")
        upd.update(status=WalkStatus.RECEIVED, reason=ReasonCode.WRITER_TRANSIENT)
    elif upd.get("writer_error") == "permanent":
        upd["final_reason"] = ReasonCode.WRITER_PERMANENT
    upd.setdefault("regen_count", 0)
    return upd


def ground_check(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """PRD §8 in code: parse + checks 1-7. Codes only."""
    parsed = parse_writer_output(state.get("raw_output") or "")
    if parsed.value is None:
        return {"output": None, "check": None, "check_passed": False,
                "check_reasons": parsed.reasons}  # fmt: skip
    res = check_entry(parsed.value, state.get("transcript") or "", state["group"])
    return {
        "output": parsed.value,
        "check": res,
        "best_check": res,
        "check_passed": res.passed,
        "check_reasons": res.reasons,
    }


def regen(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-J5: one regeneration with the failed check's codes as fixed instructions."""
    upd = _call_writer(state, deps, tuple(state.get("check_reasons") or ()))
    upd["regen_count"] = int(state.get("regen_count") or 0) + 1
    err = upd.get("writer_error")
    if err is not None:
        upd["final_reason"] = (
            ReasonCode.WRITER_TRANSIENT if err == "transient" else ReasonCode.WRITER_PERMANENT
        )
    return upd


def template(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-J5: template entry built by code from the verified facts of the best check."""
    best = state.get("best_check")
    group = state["group"]
    if best is not None:
        out = template_output(group, best)
    else:
        out = WriterOutput(
            facts=[],
            mission_result=MissionResult.NOT_MENTIONED,
            entry=build_template_entry(group, [], False),
        )
    final_reason = state.get("final_reason") or state.get("forced_reason")
    return {"final_output": out, "final_passed": False, "writer": "template",
            "final_reason": final_reason}  # fmt: skip


def store(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """One transaction: transcript, facts (verified flags), entry; status ``done``."""
    walk_id = state["walk_id"]
    if state.get("writer") == "template":
        out = state.get("final_output")
        assert out is not None
        # Template facts are exactly the verified facts of the best check.
        summary = db.GroundingSummary(
            passed=False,
            fact_verified=tuple(True for _ in out.facts),
            mission_result=out.mission_result,
            mission_span=out.mission_span,
        )
        writer: str = "template"
        n_facts = len(out.facts)
    else:
        out = state.get("output")
        check = state.get("check")
        assert out is not None and check is not None
        summary = grounding_summary(out, check, True)
        writer = deps.settings.writer_backend.value
        n_facts = len(check.verified_facts)
    final_reason = state.get("final_reason")
    with open_conn(deps.connect) as conn:
        transcript = state.get("transcript")
        if transcript is None:
            row = db.get_walk_row(conn, walk_id)
            transcript = str(row["transcript"] or "") if row is not None else ""
        db.store_journal_result(
            conn,
            walk_id,
            transcript,
            out,
            summary,
            out.entry,
            writer,
            int(state.get("regen_count") or 0),
            dict(state.get("latency_ms") or {}),
        )
        if final_reason is not None:
            db.update_walk_status(conn, walk_id, WalkStatus.DONE, final_reason)
    with deps.retry.lock:
        deps.retry.forget(walk_id)
    log.info(
        "walk_done",
        walk_id=walk_id,
        facts=n_facts,
        writer=writer,
        regen=int(state.get("regen_count") or 0),
        reason=final_reason.value if final_reason else None,
    )
    return {
        "status": WalkStatus.DONE,
        "reason": final_reason,
        "writer": writer,
        "final_passed": summary.passed,
        "facts": n_facts,
        "mission_result": summary.mission_result,
    }


def plan_next_mission(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-M7: plan the next mission now that memory is fresh. Scheduled, never inline."""
    try:
        deps.mission_trigger(state["group"])
    except Exception as exc:
        log.warning("mission_trigger_failed", walk_id=state["walk_id"],
                    exc_type=type(exc).__name__)  # fmt: skip
    return {}


def notify(state: JournalState, deps: JournalDeps) -> dict[str, Any]:
    """FR-O1/FR-O4: metadata-only push; a text-free health ping on a keyword hit."""
    group = state["group"]
    transcript = state.get("transcript")
    if transcript is None:
        with open_conn(deps.connect) as conn:
            row = db.get_walk_row(conn, state["walk_id"])
            transcript = str(row["transcript"] or "") if row is not None else ""
    try:
        if has_health_keyword(transcript):
            deps.notifier.health_ping(group)
        deps.notifier.walk_done(
            group, int(state.get("facts") or 0),
            state.get("mission_result") or MissionResult.NOT_MENTIONED,
        )  # fmt: skip
    except Exception as exc:
        log.warning("notify_failed", walk_id=state["walk_id"], exc_type=type(exc).__name__)
    return {}


# ------------------------------------------------------------------ graph


def _route_start(state: JournalState) -> str:
    resume = state.get("resume")
    if resume == "write":
        return "write_entry"
    if resume == "template":
        return "template"
    return "ingest"


def _route_after_ingest(state: JournalState) -> str:
    return "transcribe" if state.get("wav_path") else END


def _route_after_transcribe(state: JournalState) -> str:
    return "write_entry" if state.get("transcript") else END


def _route_after_write(state: JournalState) -> str:
    err = state.get("writer_error")
    if err == "transient":
        return END  # pending: retry_pending resumes from write_entry
    if err == "permanent":
        return "template"
    return "ground_check"


def _route_after_check(state: JournalState) -> str:
    if state.get("check_passed"):
        return "store"
    if int(state.get("regen_count") or 0) == 0:
        return "regen"
    return "template"


def _route_after_regen(state: JournalState) -> str:
    return "template" if state.get("writer_error") else "ground_check"


NODES: tuple[str, ...] = (
    "ingest", "transcribe", "write_entry", "ground_check", "regen", "template", "store",
    "plan_next_mission", "notify",
)  # fmt: skip

_FUNCS: dict[str, Callable[[JournalState, JournalDeps], dict[str, Any]]] = {
    "ingest": ingest,
    "transcribe": transcribe,
    "write_entry": write_entry,
    "ground_check": ground_check,
    "regen": regen,
    "template": template,
    "store": store,
    "plan_next_mission": plan_next_mission,
    "notify": notify,
}


def _bind(name: str, deps: JournalDeps) -> Callable[[JournalState], dict[str, Any]]:
    func = _FUNCS[name]

    def call(state: JournalState) -> dict[str, Any]:
        return func(state, deps)

    return timed_node(name, call, connect=deps.connect, span=deps.span)


def build_journal_graph(deps: JournalDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the journal graph. No checkpointer (FR-J6, §9.4)."""
    g: StateGraph[Any, Any, Any, Any] = StateGraph(JournalState)
    for name in NODES:
        g.add_node(name, cast(Any, _bind(name, deps)))
    g.add_conditional_edges(START, _route_start, ["ingest", "write_entry", "template"])
    g.add_conditional_edges("ingest", _route_after_ingest, ["transcribe", END])
    g.add_conditional_edges("transcribe", _route_after_transcribe, ["write_entry", END])
    g.add_conditional_edges("write_entry", _route_after_write, ["ground_check", "template", END])
    g.add_conditional_edges("ground_check", _route_after_check, ["store", "regen", "template"])
    g.add_conditional_edges("regen", _route_after_regen, ["ground_check", "template"])
    g.add_edge("template", "store")
    g.add_edge("store", "plan_next_mission")
    g.add_edge("plan_next_mission", "notify")
    g.add_edge("notify", END)
    return g.compile(checkpointer=None)


def _outcome(walk_id: str, final: Mapping[str, Any]) -> JournalOutcome:
    return JournalOutcome(
        walk_id=walk_id,
        status=final.get("status") or WalkStatus.RECEIVED,
        reason=final.get("reason"),
        writer=final.get("writer"),
        check_passed=final.get("final_passed") if final.get("status") is WalkStatus.DONE else None,
        regen_count=int(final.get("regen_count") or 0),
        facts=int(final.get("facts") or 0),
        mission_result=final.get("mission_result"),
        latency_ms=dict(final.get("latency_ms") or {}),
    )


def _invoke(deps: JournalDeps, state: JournalState) -> JournalOutcome:
    walk_id = state["walk_id"]
    graph = build_journal_graph(deps)
    final = graph.invoke(state)
    return _outcome(walk_id, final)


def _ensure_walk(deps: JournalDeps, walk_id: str, group: str, mission_id: str | None) -> None:
    with open_conn(deps.connect) as conn:
        if db.get_walk_row(conn, walk_id) is None:
            db.insert_walk(conn, walk_id=walk_id, group=group, recorded_at=deps.clock(),
                           mission_id=mission_id)  # fmt: skip


def run_journal(
    walk_id: str,
    raw_audio_path: str | Path,
    group: str,
    mission_id: str | None,
    deps: JournalDeps,
) -> JournalOutcome:
    """Run the journal graph for one uploaded note (sync; the app runs it in a thread).

    The app saves the upload as ``<tmp_dir>/<walk_id><ext>`` and inserts the walk row (status
    ``received``) before calling; a missing row is created here with ``recorded_at = now``.
    """
    _ensure_walk(deps, walk_id, group, mission_id)
    with deps.retry.lock:
        if walk_id in deps.retry.in_flight:
            return JournalOutcome(walk_id=walk_id, status=WalkStatus.RECEIVED)
        deps.retry.in_flight.add(walk_id)
    try:
        return _invoke(
            deps,
            {"walk_id": walk_id, "group": group, "mission_id": mission_id,
             "raw_path": str(raw_audio_path), "regen_count": 0, "latency_ms": {}},
        )  # fmt: skip
    finally:
        with deps.retry.lock:
            deps.retry.in_flight.discard(walk_id)


# ------------------------------------------------------------------ retry worker / sweep


def _expire(deps: JournalDeps, row: sqlite3.Row) -> JournalOutcome:
    walk_id = str(row["id"])
    if row["reason"] == ReasonCode.WRITER_TRANSIENT.value:
        # Transcript already stored, audio already gone: store the no-fact template (done).
        return _invoke(
            deps,
            {"walk_id": walk_id, "group": str(row["group_name"]),
             "mission_id": row["mission_id"], "raw_path": None, "resume": "template",
             "forced_reason": ReasonCode.WRITER_TRANSIENT, "regen_count": 0,
             "latency_ms": {}},
        )  # fmt: skip
    delete_path(find_raw_audio(deps.settings.tmp_dir, walk_id))
    delete_path(work_dir_for(deps.settings.tmp_dir, walk_id))
    _mark(deps, walk_id, WalkStatus.LOST, ReasonCode.RETRY_WINDOW_EXPIRED)
    log.info("walk_lost", walk_id=walk_id, reason=ReasonCode.RETRY_WINDOW_EXPIRED.value)
    return JournalOutcome(walk_id, WalkStatus.LOST, ReasonCode.RETRY_WINDOW_EXPIRED)


def _retry_one(deps: JournalDeps, row: sqlite3.Row) -> JournalOutcome:
    walk_id = str(row["id"])
    base: JournalState = {"walk_id": walk_id, "group": str(row["group_name"]),
                          "mission_id": row["mission_id"], "regen_count": 0,
                          "latency_ms": {}}  # fmt: skip
    if row["reason"] == ReasonCode.WRITER_TRANSIENT.value:
        with open_conn(deps.connect) as conn:
            w = db.get_walk_row(conn, walk_id)
            transcript = str(w["transcript"]) if w is not None and w["transcript"] else None
            mtext = mission_text(conn, row["mission_id"])
        if transcript is None:  # cannot happen in normal flow; nothing to write from
            _mark(deps, walk_id, WalkStatus.LOST, ReasonCode.TRANSCRIPT_EMPTY)
            return JournalOutcome(walk_id, WalkStatus.LOST, ReasonCode.TRANSCRIPT_EMPTY)
        return _invoke(deps, {**base, "resume": "write", "transcript": transcript,
                              "mission_text": mtext, "raw_path": None})  # fmt: skip
    raw = find_raw_audio(deps.settings.tmp_dir, walk_id)
    if raw is None:  # audio already gone (e.g. swept): nothing left to transcribe
        _mark(deps, walk_id, WalkStatus.LOST, ReasonCode.RETRY_WINDOW_EXPIRED)
        return JournalOutcome(walk_id, WalkStatus.LOST, ReasonCode.RETRY_WINDOW_EXPIRED)
    return _invoke(deps, {**base, "raw_path": str(raw)})


def retry_pending(now: dt.datetime, deps: JournalDeps) -> list[JournalOutcome]:
    """FR-J6 retry worker step: call periodically (e.g. every 10 s) from the app.

    For every walk still ``received`` with TRANSCRIBE_TRANSIENT or WRITER_TRANSIENT: past the
    window (TRANSIENT_RETRY_WINDOW_S since upload) -> expire; due by backoff -> re-run.
    """
    window = dt.timedelta(seconds=deps.settings.transient_retry_window_s)
    with open_conn(deps.connect) as conn:
        rows = pending_walks(
            conn, (ReasonCode.TRANSCRIBE_TRANSIENT.value, ReasonCode.WRITER_TRANSIENT.value)
        )
    results: list[JournalOutcome] = []
    for row in rows:
        walk_id = str(row["id"])
        uploaded = dt.datetime.fromisoformat(str(row["recorded_at"]))
        with deps.retry.lock:
            if walk_id in deps.retry.in_flight:
                continue
            expired = now - uploaded >= window
            due = deps.retry.next_at.get(walk_id)
            if not expired and due is not None and now < due:
                continue
            deps.retry.in_flight.add(walk_id)
        try:
            if expired:
                results.append(_expire(deps, row))
                with deps.retry.lock:
                    deps.retry.forget(walk_id)
            else:
                results.append(_retry_one(deps, row))
        finally:
            with deps.retry.lock:
                deps.retry.in_flight.discard(walk_id)
    return results


def _walk_id_from(p: Path, tmp_dir: Path) -> str | None:
    rel = p.relative_to(tmp_dir)
    first = rel.parts[0]
    if first.startswith("work-"):
        return first[len("work-") :]
    if len(rel.parts) == 1:
        return first.split(".", 1)[0]
    return None


def sweep_stale_audio(now: dt.datetime, deps: JournalDeps) -> int:
    """FR-J6 startup sweep: delete temp audio older than 10 minutes; mark such walks lost.

    Age is file mtime vs ``now``. Returns the number of files deleted.
    """
    tmp = deps.settings.tmp_dir
    if not tmp.is_dir():
        return 0
    cutoff = now.timestamp() - STALE_AUDIO_S
    removed = 0
    walks: set[str] = set()
    for p in sorted(tmp.rglob("*"), reverse=True):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                wid = _walk_id_from(p, tmp)
                p.unlink(missing_ok=True)
                removed += 1
                if wid:
                    walks.add(wid)
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        except OSError:
            log.warning("sweep_delete_failed", kind="oserror")
    with open_conn(deps.connect) as conn:
        for wid in sorted(walks):
            row = db.get_walk_row(conn, wid)
            if row is not None and row["status"] == WalkStatus.RECEIVED.value:
                db.update_walk_status(conn, wid, WalkStatus.LOST, ReasonCode.STALE_AUDIO_SWEPT)
                with deps.retry.lock:
                    deps.retry.forget(wid)
    log.info("audio_swept", files=removed, walks=len(walks))
    return removed
