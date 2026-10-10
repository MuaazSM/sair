"""Startup work and background loops (asyncio tasks; the sync graphs run via ``to_thread``).

* startup: offline env, DB + migrations, walkers, stale-audio sweep (FR-J6), requeue of walks a
  crash left half-stored;
* retry worker: every 30 s ``retry_pending`` (transient transcription/writer failures, FR-J6);
* mission-audio expiry: hourly (FR-M8);
* pre-generation: daily at ``MISSION_PREGEN_TIME`` local, never inside the heat block (FR-M6b).

Nothing here logs content; log fields are IDs, counts and exception type names.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
from dataclasses import dataclass

import structlog

from hub import db
from hub.api import queries
from hub.api.deps import AppDeps, submit_guarded
from hub.graphs.journal import retry_pending, sweep_stale_audio
from hub.graphs.mission import expire_mission_audio, plan_mission, should_pregenerate
from hub.settings import apply_offline_env

__all__ = ["StartupReport", "Workers", "plan_missing", "seconds_until", "startup"]

log = structlog.get_logger("hub.workers")


@dataclass(frozen=True)
class StartupReport:
    walkers: int
    swept_files: int
    requeued: int


def startup(deps: AppDeps) -> StartupReport:
    """Blocking startup steps, in order. Safe to call once per process."""
    s = deps.settings
    apply_offline_env(s)
    s.tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    s.audio_dir.mkdir(parents=True, exist_ok=True)
    now = deps.clock()
    conn = db.init_db(s.db_path)
    try:
        walkers = db.upsert_walkers_from_settings(conn, s, now)
    finally:
        conn.close()
    swept = sweep_stale_audio(now, deps.journal)
    conn = deps.connect()
    try:
        requeued = queries.requeue_interrupted(conn, s.tmp_dir)
    finally:
        conn.close()
    log.info("startup_done", walkers=walkers, swept_files=swept, requeued=requeued)
    return StartupReport(walkers=walkers, swept_files=swept, requeued=requeued)


def seconds_until(now: dt.datetime, at: dt.time, tz: dt.tzinfo) -> float:
    """Seconds from ``now`` to the next local ``at`` (strictly in the future)."""
    local = now.astimezone(tz)
    target = local.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    if target <= local:
        target += dt.timedelta(days=1)
    return (target - local).total_seconds()


def plan_missing(deps: AppDeps) -> int:
    """FR-M7: queue planning of today's mission for every group that has none cached.

    Skipped inside the heat block (FR-M6b). Returns the number of jobs queued.
    """
    now = deps.clock()
    if not should_pregenerate(now, deps.settings):
        return 0
    today = deps.today()
    for group in deps.groups():
        submit_guarded(
            deps.planner,
            lambda g=group: plan_mission(g, today, deps.mission, replace=False),  # type: ignore[misc]
            "pregen",
        )
    return len(deps.groups())


class Workers:
    def __init__(self, deps: AppDeps) -> None:
        self.deps = deps
        self.tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        if self.deps.plan_on_startup:
            try:
                plan_missing(self.deps)
            except Exception as exc:
                log.warning("startup_plan_failed", exc_type=type(exc).__name__)
        if not self.deps.run_workers:
            return
        self.tasks = [
            asyncio.create_task(self._retry_loop(), name="sair-retry"),
            asyncio.create_task(self._expiry_loop(), name="sair-expiry"),
            asyncio.create_task(self._pregen_loop(), name="sair-pregen"),
        ]

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        for t in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self.tasks = []

    async def _retry_loop(self) -> None:
        d = self.deps
        while True:  # first pass is immediate: resumes walks requeued at startup
            try:
                outcomes = await asyncio.to_thread(retry_pending, d.clock(), d.journal)
                if outcomes:
                    log.info("retry_pass", walks=len(outcomes))
            except Exception as exc:
                log.warning("retry_failed", exc_type=type(exc).__name__)
            await asyncio.sleep(d.retry_interval_s)

    async def _expiry_loop(self) -> None:
        d = self.deps
        while True:
            try:
                await asyncio.to_thread(expire_mission_audio, d.clock(), d.settings)
            except Exception as exc:
                log.warning("expiry_failed", exc_type=type(exc).__name__)
            await asyncio.sleep(d.expiry_interval_s)

    async def _pregen_loop(self) -> None:
        d = self.deps
        while True:
            await asyncio.sleep(
                seconds_until(d.clock(), d.settings.mission_pregen_time, d.settings.tz)
            )
            try:
                plan_missing(d)
            except Exception as exc:
                log.warning("pregen_failed", exc_type=type(exc).__name__)
            await asyncio.sleep(1.0)  # never re-fire inside the same minute
