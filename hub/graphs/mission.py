"""Mission graph (PRD J1/J4, FR-M1..M9): memory -> one safe mission, spoken and cached.

Nodes (one function each; one metric row and, in dev/eval, one span each)::

    START -> recall -> seed ------------------------------> speak -> cache -> END
                   \\-> plan -> mission_check -> speak
                         |        |    ^   \\
                         |        v    |    -> fallback -> speak
                         |       regen-/
                         \\-> fallback   (planner unreachable / permanent error)

* recall: the group's verified facts from its last 7 done walks + its last 5 mission texts
  (FR-M2/M3) and the number of done walks.
* seed (FR-M2): while the group has fewer than 3 done walks, the hand-written seed missions
  from data/seed_missions.yaml are used in order (walk 1 -> seed 1 ...). No model.
* plan: Gemma planner via ``chat_json`` with the mission JSON schema; mission_check runs
  ``hub.safety.check_mission`` (provenance on, group-scoped); one regeneration with the reason
  codes; then a deterministic fallback from data/fallback_missions.yaml that avoids the last
  5 mission texts (FR-M6).
* speak (FR-M8): Kokoro renders ``<audio_dir>/<mission_id>.mp3``; a TTS failure leaves
  ``audio_path`` None and the mission is still cached (the API can re-render).
* cache: ``insert_mission`` (source model|seed|fallback) which also writes ``mission_refs``.

``GET /mission/today`` uses :func:`get_or_fallback_mission`, which never calls a model.
Mission text is never logged (NFR-P4); outcomes carry IDs, codes and counts.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

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
    TtsLike,
    count_done_walks,
    null_span,
    open_conn,
    timed_node,
    utc_now,
)
from hub.models.errors import Permanent, Transient
from hub.prompts.builders import MissionText, WalkerGroup, build_planner_prompt
from hub.safety import check_mission
from hub.schemas import (
    Mission,
    MissionPlan,
    MissionSource,
    MissionType,
    ReasonCode,
    VerifiedFact,
    parse_mission_plan,
)
from hub.settings import Settings

__all__ = [
    "MEMORY_WALKS",
    "MISSION_AUDIO_TTL_S",
    "RECENT_MISSIONS",
    "SEED_WALKS",
    "MissionDeps",
    "MissionOutcome",
    "MissionScheduler",
    "MissionState",
    "build_mission_graph",
    "cache",
    "expire_mission_audio",
    "fallback",
    "get_or_fallback_mission",
    "mission_check",
    "next_mission_date",
    "pick_fallback",
    "plan",
    "plan_mission",
    "post_walk_trigger",
    "recall",
    "regen",
    "render_mission_audio",
    "seed",
    "should_pregenerate",
    "speak",
    "warm_transcriber",
]

log = structlog.get_logger("hub.graphs.mission")

MEMORY_WALKS = 7  # FR-M2: facts from the group's last 7 walks
RECENT_MISSIONS = 5  # FR-M3: the last 5 mission texts
SEED_WALKS = 3  # FR-M2: hand-written seeds for the first 3 walks
MISSION_AUDIO_TTL_S = 48 * 3600  # FR-M8
FALLBACK_PATH = REPO_ROOT / "data" / "fallback_missions.yaml"
SEED_PATH = REPO_ROOT / "data" / "seed_missions.yaml"
MISSION_SCHEMA_PATH = REPO_ROOT / "schemas" / "mission.schema.json"

MissionScheduler = Callable[[str, dt.date], None]
"""Schedules ``plan_mission(group, date)`` in the background. Must return immediately."""


@dataclass
class MissionDeps:
    settings: Settings
    connect: ConnectFn  # returns a NEW connection each call (closed by the graph)
    ollama: TranscriberLike
    tts: TtsLike | None
    scheduler: MissionScheduler
    clock: Clock = utc_now
    span: SpanFn = null_span


class MissionState(TypedDict, total=False):
    group: str
    date: dt.date
    # recall
    facts: list[VerifiedFact]
    recent: list[str]
    done_walks: int
    # plan / check
    raw_plan: str | None
    planner_error: Literal["transient", "permanent"] | None
    plan: MissionPlan | None
    check_passed: bool | None
    check_reasons: tuple[ReasonCode, ...]
    regen_count: int
    # result
    mission: Mission | None
    reasons: list[ReasonCode]
    audio_path: str | None
    latency_ms: dict[str, float]


@dataclass(frozen=True)
class MissionOutcome:
    """IDs, codes and counts only (safe to log)."""

    mission_id: str
    source: MissionSource
    type: MissionType
    references: int
    regen_count: int
    audio: bool
    reasons: tuple[ReasonCode, ...] = ()
    latency_ms: Mapping[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ data files


@dataclass(frozen=True)
class _Canned:
    type: MissionType
    text: str


def _load_canned(path: Path) -> tuple[_Canned, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data.get("missions", []) if isinstance(data, dict) else []
    return tuple(_Canned(MissionType(str(m["type"])), str(m["text"])) for m in items)


@functools.cache
def _fallbacks() -> tuple[_Canned, ...]:
    return _load_canned(FALLBACK_PATH)


@functools.cache
def _seeds() -> tuple[_Canned, ...]:
    return _load_canned(SEED_PATH)


@functools.cache
def _mission_schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(MISSION_SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split()).strip(" .!?")


def pick_fallback(date: dt.date, recent: list[str]) -> tuple[MissionType, str]:
    """Deterministic fallback for ``date`` that avoids the last 5 mission texts (FR-M6)."""
    pool = _fallbacks()
    seen = {_norm(t) for t in recent[:RECENT_MISSIONS]}
    candidates = [m for m in pool if _norm(m.text) not in seen] or list(pool)
    chosen = candidates[date.toordinal() % len(candidates)]
    return chosen.type, chosen.text


def _new_mission(
    group: str, date: dt.date, mtype: MissionType, text: str, refs: list[str], src: MissionSource
) -> Mission:
    return Mission(id=uuid.uuid4().hex, date=date, group=group, type=mtype, text=text,
                   references=refs, source=src)  # fmt: skip


# ------------------------------------------------------------------ nodes


def recall(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M2/M3: verified facts (last 7 walks) + last 5 mission texts, group-scoped."""
    group = state["group"]
    with open_conn(deps.connect) as conn:
        facts = db.verified_facts_for_group(conn, group, last_n_walks=MEMORY_WALKS)
        recent = [m.text for m in db.latest_missions(conn, group, n=RECENT_MISSIONS)]
        done = count_done_walks(conn, group)
    return {"facts": facts, "recent": recent, "done_walks": done, "regen_count": 0,
            "reasons": []}  # fmt: skip


def _call_planner(
    state: MissionState, deps: MissionDeps, regen_codes: tuple[ReasonCode, ...]
) -> dict[str, Any]:
    messages = build_planner_prompt(
        list(state.get("facts") or []),
        [MissionText(t) for t in (state.get("recent") or [])[:RECENT_MISSIONS]],
        group=WalkerGroup(state["group"]),
        today=state["date"],
        regen=regen_codes,
    )
    try:
        raw, stats = deps.ollama.chat_json(
            messages, _mission_schema(), role="planner",
            temperature=deps.settings.planner_temperature,
        )  # fmt: skip
    except Transient:
        return {"raw_plan": None, "planner_error": "transient",
                "reasons": [*state.get("reasons", []), ReasonCode.PLANNER_TRANSIENT]}  # fmt: skip
    except Permanent:
        return {"raw_plan": None, "planner_error": "permanent",
                "reasons": [*state.get("reasons", []), ReasonCode.PLANNER_PERMANENT]}  # fmt: skip
    return {"raw_plan": raw, "planner_error": None,
            "_tokens_in": stats.prompt_tokens, "_tokens_out": stats.completion_tokens}  # fmt: skip


def plan(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """Gemma planner; the prompt carries only VerifiedFact records + 5 mission texts."""
    return _call_planner(state, deps, ())


def mission_check(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M1/M4/M5/M6/M6b in code (hub.safety), provenance on, group-scoped."""
    parsed = parse_mission_plan(state.get("raw_plan") or "")
    reasons = list(state.get("reasons") or [])
    if parsed.value is None:
        return {"plan": None, "check_passed": False, "check_reasons": parsed.reasons,
                "reasons": reasons + list(parsed.reasons)}  # fmt: skip
    p = parsed.value
    facts = state.get("facts") or []
    res = check_mission(
        p,
        {f.id: f for f in facts},
        recent_missions=list(state.get("recent") or []),
        provenance=True,
        group=state["group"],
    )
    if not res.passed:
        return {"plan": p, "check_passed": False, "check_reasons": res.reasons,
                "reasons": reasons + list(res.reasons)}  # fmt: skip
    mission = _new_mission(state["group"], state["date"], p.type, p.text, list(p.references),
                           MissionSource.MODEL)  # fmt: skip
    return {"plan": p, "check_passed": True, "check_reasons": (), "mission": mission}


def regen(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M6: one regeneration with the failed check's codes as fixed instructions."""
    upd = _call_planner(state, deps, tuple(state.get("check_reasons") or ()))
    upd["regen_count"] = int(state.get("regen_count") or 0) + 1
    if upd.get("planner_error") is None:
        upd["reasons"] = [*state.get("reasons", []), ReasonCode.REGENERATED]
    return upd


def fallback(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M6: a fixed, owner-written fallback mission; deterministic; avoids recent repeats."""
    mtype, text = pick_fallback(state["date"], list(state.get("recent") or []))
    mission = _new_mission(state["group"], state["date"], mtype, text, [], MissionSource.FALLBACK)
    return {"mission": mission,
            "reasons": [*state.get("reasons", []), ReasonCode.MISSION_FALLBACK]}  # fmt: skip


def seed(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M2 seed: hand-written mission N for the group's walk N (N <= 3). No model."""
    seeds = _seeds()
    idx = min(int(state.get("done_walks") or 0), len(seeds) - 1)
    chosen = seeds[idx]
    mission = _new_mission(state["group"], state["date"], MissionType.SEED, chosen.text, [],
                           MissionSource.SEED)  # fmt: skip
    return {"mission": mission}


def speak(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """FR-M8: render mission audio; failure -> audio_path None (mission still cached)."""
    mission = state.get("mission")
    assert mission is not None
    return {"audio_path": _render(mission, deps)}


def cache(state: MissionState, deps: MissionDeps) -> dict[str, Any]:
    """Store the mission (+ mission_refs) so /mission/today serves it without a model."""
    mission = state.get("mission")
    assert mission is not None
    with open_conn(deps.connect) as conn:
        db.insert_mission(conn, mission)  # also links mission_refs for referenced facts
    log.info(
        "mission_cached",
        mission_id=mission.id,
        source=mission.source.value,
        type=mission.type.value,
        refs=len(mission.references),
        regen=int(state.get("regen_count") or 0),
        audio=state.get("audio_path") is not None,
    )
    return {}


# ------------------------------------------------------------------ graph


def _route_after_recall(state: MissionState) -> str:
    return "seed" if int(state.get("done_walks") or 0) < SEED_WALKS else "plan"


def _route_after_plan(state: MissionState) -> str:
    return "fallback" if state.get("planner_error") else "mission_check"


def _route_after_check(state: MissionState) -> str:
    if state.get("check_passed"):
        return "speak"
    return "regen" if int(state.get("regen_count") or 0) == 0 else "fallback"


NODES: tuple[str, ...] = (
    "recall", "plan", "mission_check", "regen", "fallback", "seed", "speak", "cache",
)  # fmt: skip

_FUNCS: dict[str, Callable[[MissionState, MissionDeps], dict[str, Any]]] = {
    "recall": recall,
    "plan": plan,
    "mission_check": mission_check,
    "regen": regen,
    "fallback": fallback,
    "seed": seed,
    "speak": speak,
    "cache": cache,
}


def _bind(name: str, deps: MissionDeps) -> Callable[[MissionState], dict[str, Any]]:
    func = _FUNCS[name]

    def call(state: MissionState) -> dict[str, Any]:
        return func(state, deps)

    return timed_node(f"mission_{name}", call, connect=deps.connect, span=deps.span,
                      walk_id_key=None)  # fmt: skip


def build_mission_graph(deps: MissionDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the mission graph. No checkpointer."""
    g: StateGraph[Any, Any, Any, Any] = StateGraph(MissionState)
    for name in NODES:
        g.add_node(name, cast(Any, _bind(name, deps)))
    g.add_edge(START, "recall")
    g.add_conditional_edges("recall", _route_after_recall, ["seed", "plan"])
    g.add_conditional_edges("plan", _route_after_plan, ["mission_check", "fallback"])
    g.add_conditional_edges("mission_check", _route_after_check, ["speak", "regen", "fallback"])
    g.add_conditional_edges("regen", _route_after_plan, ["mission_check", "fallback"])
    g.add_edge("seed", "speak")
    g.add_edge("fallback", "speak")
    g.add_edge("speak", "cache")
    g.add_edge("cache", END)
    return g.compile(checkpointer=None)


# ------------------------------------------------------------------ public API


def plan_mission(
    group: str, date: dt.date, deps: MissionDeps, *, replace: bool = True
) -> MissionOutcome | None:
    """Run the mission graph for (group, date) and cache the result.

    ``replace=False`` returns None without planning if a mission is already cached for the
    date (startup/pregen/miss-scheduled planning); post-walk planning uses ``replace=True``.
    """
    if not replace:
        with open_conn(deps.connect) as conn:
            if db.get_cached_mission(conn, group, date) is not None:
                return None
    final = build_mission_graph(deps).invoke(
        {"group": group, "date": date, "regen_count": 0, "reasons": [], "latency_ms": {}}
    )
    mission: Mission = final["mission"]
    return MissionOutcome(
        mission_id=mission.id,
        source=mission.source,
        type=mission.type,
        references=len(mission.references),
        regen_count=int(final.get("regen_count") or 0),
        audio=final.get("audio_path") is not None,
        reasons=tuple(final.get("reasons") or ()),
        latency_ms=dict(final.get("latency_ms") or {}),
    )


def next_mission_date(now: dt.datetime, settings: Settings) -> dt.date:
    """Post-walk planning targets the next local day (today's served mission stays stable)."""
    return now.astimezone(settings.tz).date() + dt.timedelta(days=1)


def post_walk_trigger(
    deps: MissionDeps, submit: Callable[[Callable[[], object]], object]
) -> Callable[[str], None]:
    """FR-M7: the journal graph's ``mission_trigger``: plans tomorrow's mission via ``submit``
    (an executor/background-task hook), so the walk never waits on the planner."""

    def trigger(group: str) -> None:
        date = next_mission_date(deps.clock(), deps.settings)
        submit(lambda: plan_mission(group, date, deps, replace=True))

    return trigger


def get_or_fallback_mission(group: str, today: dt.date, deps: MissionDeps) -> Mission:
    """FR-M1/M7: ``GET /mission/today``. Never calls a model.

    Cache hit -> the cached mission. Miss -> a seed (group has < 3 done walks, FR-M2) or a
    fallback mission, inserted for ``today`` and returned at once; planning for the next day
    is handed to ``deps.scheduler`` so today's mission stays the one already heard.
    """
    with open_conn(deps.connect) as conn:
        cached = db.get_cached_mission(conn, group, today)
        if cached is not None:
            return cached
        done = count_done_walks(conn, group)
        if done < SEED_WALKS:
            seeds = _seeds()
            s = seeds[min(done, len(seeds) - 1)]
            mission = _new_mission(group, today, MissionType.SEED, s.text, [], MissionSource.SEED)
        else:
            recent = [m.text for m in db.latest_missions(conn, group, n=RECENT_MISSIONS)]
            mtype, text = pick_fallback(today, recent)
            mission = _new_mission(group, today, mtype, text, [], MissionSource.FALLBACK)
        db.insert_mission(conn, mission)
    log.info("mission_cache_miss", mission_id=mission.id, source=mission.source.value)
    try:
        deps.scheduler(group, today + dt.timedelta(days=1))
    except Exception as exc:
        log.warning("mission_schedule_failed", exc_type=type(exc).__name__)
    return mission


def should_pregenerate(now: dt.datetime, settings: Settings) -> bool:
    """FR-M6b: pre-generation never runs inside MISSION_PREGEN_BLOCK (local time)."""
    start, end = settings.mission_pregen_block
    local = now.astimezone(settings.tz).time() if now.tzinfo else now.time()
    if start <= end:
        return not (start <= local < end)
    return not (local >= start or local < end)  # block wraps midnight


def _render(mission: Mission, deps: MissionDeps) -> str | None:
    if deps.tts is None:
        return None
    out = db.mission_audio_path(deps.settings.audio_dir, mission.id)
    try:
        return str(deps.tts.render(mission.text, out))
    except (Transient, Permanent) as exc:
        log.warning("mission_tts_failed", mission_id=mission.id, kind=exc.kind)
    except Exception as exc:
        log.warning("mission_tts_failed", mission_id=mission.id, kind=type(exc).__name__)
    return None


def render_mission_audio(mission: Mission, deps: MissionDeps) -> str | None:
    """(Re-)render a cached mission's audio (API retry, fallback missions). None on failure."""
    return _render(mission, deps)


def expire_mission_audio(now: dt.datetime, settings: Settings) -> int:
    """FR-M8: delete mission audio files older than 48 h (file mtime vs ``now``)."""
    d = settings.audio_dir
    if not d.is_dir():
        return 0
    cutoff = now.timestamp() - MISSION_AUDIO_TTL_S
    removed = 0
    for p in d.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except OSError:
            log.warning("mission_audio_delete_failed", kind="oserror")
    if removed:
        log.info("mission_audio_expired", files=removed)
    return removed


def warm_transcriber(deps: MissionDeps) -> threading.Thread:
    """FR-M9: warm Gemma for the walk that follows; runs in a daemon thread, never blocks."""

    def run() -> None:
        try:
            deps.ollama.warm()
        except Exception as exc:
            log.warning("warm_failed", exc_type=type(exc).__name__)

    t = threading.Thread(target=run, name="sair-warm", daemon=True)
    t.start()
    return t
