"""Routes (PRD §9.3). Every endpoint but ``/healthz`` needs a per-device bearer token.

The walker group always comes from the token; a ``group`` the client sends is ignored,
except ``GET /diary?group=`` which is a *filter* limited to configured group names.
Responses carry dates, never times (NFR-P5). Handlers log nothing but IDs and codes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, Request, Response

from hub import db
from hub.api import queries
from hub.api.deps import AppDeps, submit_guarded
from hub.api.errors import ApiError
from hub.api.upload import save_audio_upload
from hub.graphs.journal import run_journal
from hub.graphs.mission import (
    get_or_fallback_mission,
    render_mission_audio,
    warm_transcriber,
)
from hub.schemas import (
    DiaryResponse,
    FeedbackRequest,
    Healthz,
    MissionTodayResponse,
    ReasonCode,
    StatusResponse,
    WalkAccepted,
    WalkStatus,
    WalkStatusResponse,
)

__all__ = ["router"]

log = structlog.get_logger("hub.api")
router = APIRouter()

_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def get_deps(request: Request) -> AppDeps:
    deps: AppDeps = request.app.state.deps
    return deps


Deps = Annotated[AppDeps, Depends(get_deps)]


def _bearer(authorization: str | None) -> str:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    raise ApiError(401, "unauthorized", _UNAUTHORIZED)


def any_walker(deps: Deps, authorization: Annotated[str | None, Header()] = None) -> db.Walker:
    """Any valid device token (family or owner); 401 otherwise."""
    token = _bearer(authorization)
    conn = deps.connect()
    try:
        walker = db.walker_for_token(conn, token)
    finally:
        conn.close()
    if walker is None:
        raise ApiError(401, "unauthorized", _UNAUTHORIZED)
    return walker


Walker = Annotated[db.Walker, Depends(any_walker)]


def family_walker(walker: Walker) -> db.Walker:
    """A family device. The owner token has no walker group, so it is refused (403)."""
    if walker.is_owner:
        raise ApiError(403, "forbidden")
    return walker


def owner_walker(walker: Walker) -> db.Walker:
    if not walker.is_owner:
        raise ApiError(403, "forbidden")
    return walker


Family = Annotated[db.Walker, Depends(family_walker)]
Owner = Annotated[db.Walker, Depends(owner_walker)]


# ------------------------------------------------------------------------ health


@router.get("/healthz", response_model=Healthz)
def healthz() -> Healthz:
    return Healthz(ok=True)


# ----------------------------------------------------------------------- mission


def _render_in_background(deps: AppDeps, mission_id: str) -> None:
    """Render a mission's audio off the request path, on the TTS worker (not the planner's)."""
    with deps.render_lock:
        if mission_id in deps.rendering:
            return
        deps.rendering.add(mission_id)

    def job() -> None:
        try:
            conn = deps.connect()
            try:
                mission = queries.mission_by_id(conn, mission_id)
            finally:
                conn.close()
            if mission is not None:
                render_mission_audio(mission, deps.mission)
        finally:
            with deps.render_lock:
                deps.rendering.discard(mission_id)

    try:
        submit_guarded(deps.renderer, job, "render_audio")
    except RuntimeError:  # executor shut down
        with deps.render_lock:
            deps.rendering.discard(mission_id)


@router.get("/mission/today", response_model=MissionTodayResponse)
def mission_today(deps: Deps, walker: Family) -> MissionTodayResponse:
    """FR-M7: never waits on an LLM. FR-M9: warms the transcriber in the background."""
    mission = get_or_fallback_mission(walker.group_name, deps.today(), deps.mission)
    now = time.monotonic()
    if now - deps.last_warm >= deps.warm_throttle_s:
        deps.last_warm = now
        warm_transcriber(deps.mission)
    audio = db.mission_audio_path(deps.settings.audio_dir, mission.id)
    has_audio = audio.is_file()
    if not has_audio:
        _render_in_background(deps, mission.id)
    return MissionTodayResponse(
        mission_id=mission.id,
        text=mission.text,
        type=mission.type,
        audio_path=f"/audio/{mission.id}" if has_audio else None,
    )


@router.get("/audio/{mission_id}")
def mission_audio(mission_id: str, deps: Deps, walker: Family) -> Response:
    conn = deps.connect()
    try:
        group = queries.mission_group(conn, mission_id)
    finally:
        conn.close()
    if group != walker.group_name:
        raise ApiError(404, "not_found")
    path = db.mission_audio_path(deps.settings.audio_dir, mission_id)
    try:
        data = path.read_bytes()  # a few tens of KB; no Last-Modified header (NFR-P5)
    except OSError:
        raise ApiError(404, "not_found") from None
    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


# -------------------------------------------------------------------------- walk


async def _run_walk(
    deps: AppDeps, walk_id: str, path: Path, group: str, mission_id: str | None
) -> None:
    try:
        await asyncio.to_thread(run_journal, walk_id, path, group, mission_id, deps.journal)
    except Exception as exc:
        # A graph bug must not strand the upload: hand it to the bounded retry worker. Once the
        # transcript is stored the audio is gone, so the retry resumes at the writer (FR-J7).
        log.error("journal_failed", walk_id=walk_id, exc_type=type(exc).__name__)
        try:
            conn = deps.connect()
            try:
                row = db.get_walk_row(conn, walk_id)
                if row is not None and row["status"] == WalkStatus.RECEIVED.value:
                    reason = (
                        ReasonCode.WRITER_TRANSIENT
                        if row["transcript"] is not None
                        else ReasonCode.TRANSCRIBE_TRANSIENT
                    )
                    db.update_walk_status(conn, walk_id, WalkStatus.RECEIVED, reason)
            finally:
                conn.close()
        except Exception as exc2:
            log.error("journal_requeue_failed", walk_id=walk_id, exc_type=type(exc2).__name__)


@router.post("/walk", response_model=WalkAccepted, status_code=202)
async def post_walk(request: Request, deps: Deps, walker: Family) -> WalkAccepted:
    walk_id = db.new_id()
    request.state.walk_id = walk_id
    path = await save_audio_upload(request, deps.settings.tmp_dir, walk_id, deps.max_upload_bytes)
    group = walker.group_name  # from the token only; any client-sent group is ignored
    try:
        conn = deps.connect()
        try:
            mission = db.get_cached_mission(conn, group, deps.today())
            mission_id = mission.id if mission else None
            db.insert_walk(
                conn,
                walk_id=walk_id,
                group=group,
                recorded_at=deps.clock(),
                mission_id=mission_id,
            )
        finally:
            conn.close()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    task = asyncio.create_task(_run_walk(deps, walk_id, path, group, mission_id))
    tasks: set[asyncio.Task[None]] = request.app.state.walk_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    log.info("walk_received", walk_id=walk_id)
    return WalkAccepted(walk_id=walk_id, status=WalkStatus.RECEIVED)


@router.get("/walk/{walk_id}", response_model=WalkStatusResponse, response_model_exclude_none=True)
def get_walk(walk_id: str, request: Request, deps: Deps, walker: Walker) -> WalkStatusResponse:
    request.state.walk_id = walk_id
    conn = deps.connect()
    try:
        found = queries.walk_view(conn, walk_id, deps.settings.tz)
    finally:
        conn.close()
    if found is None or (not walker.is_owner and found[0] != walker.group_name):
        raise ApiError(404, "not_found")
    return found[1]


# ------------------------------------------------------------------------- diary


@router.get("/diary", response_model=DiaryResponse)
def get_diary(deps: Deps, walker: Walker, group: str | None = None) -> DiaryResponse:
    """FR-D1-D4. ``group`` is a filter over configured group names; unknown -> 400."""
    if group is not None and group not in deps.groups():
        raise ApiError(400, "unknown_group")
    conn = deps.connect()
    try:
        return DiaryResponse(
            summary=db.diary_summary(conn, deps.today(), group, deps.settings.tz),
            entries=db.diary(conn, group, deps.settings.tz),
        )
    finally:
        conn.close()


# ------------------------------------------------------------------------- owner


@router.post("/feedback/{walk_id}", response_model=Healthz)
def post_feedback(
    walk_id: str, body: FeedbackRequest, request: Request, deps: Deps, walker: Owner
) -> Healthz:
    request.state.walk_id = walk_id
    conn = deps.connect()
    try:
        if not queries.walk_exists(conn, walk_id):
            raise ApiError(404, "not_found")
        # PRD §11.1: content may live only in walks.transcript/facts/entries, so the optional
        # note is accepted (API shape, §9.3) but never stored or logged; only the verdict is.
        db.insert_feedback(conn, walk_id, body.correct, None, deps.clock())
    finally:
        conn.close()
    log.info("feedback_stored", walk_id=walk_id, correct=body.correct)
    return Healthz(ok=True)


def _health(deps: AppDeps) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name, check in deps.health.items():
        try:
            out[name] = bool(check())
        except Exception:
            out[name] = False
    return out


@router.get("/status", response_model=StatusResponse)
def get_status(deps: Deps, walker: Owner) -> StatusResponse:
    """FR-O2: metadata only (counts, rates, enum codes). Never any text."""
    conn = deps.connect()
    try:
        return queries.status_response(conn, deps.settings.tmp_dir, deps.settings.tz, _health(deps))
    finally:
        conn.close()
