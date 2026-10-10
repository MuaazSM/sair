"""Application dependencies: settings, graph deps, executors and lifecycle hooks.

``build_deps`` wires the real model clients (none of them connects or starts anything at
construction time; Kokoro loads lazily in its own process). Tests pass stubs for every client.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from hub import db
from hub.graphs.common import REPO_ROOT, Clock, TranscriberLike, TtsLike, WriterLike, utc_now
from hub.graphs.journal import JournalDeps, Notifier
from hub.graphs.mission import MissionDeps, plan_mission, post_walk_trigger
from hub.models.llama_server import LlamaServerClient
from hub.models.ollama import OllamaClient
from hub.models.tts import TtsClient
from hub.notify import NtfyNotifier, NullNotifier
from hub.schemas import WriterBackend
from hub.settings import Settings

__all__ = ["MAX_UPLOAD_BYTES", "AppDeps", "build_deps", "submit_guarded"]

log = structlog.get_logger("hub.api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 90 s of any accepted format fits well below this


def submit_guarded(pool: ThreadPoolExecutor, fn: Callable[[], object], name: str) -> Future[None]:
    """Run ``fn`` on ``pool``; an exception is logged by type only (NFR-P4), never raised."""

    def run() -> None:
        try:
            fn()
        except Exception as exc:
            log.warning("background_failed", job=name, exc_type=type(exc).__name__)

    return pool.submit(run)


@dataclass
class AppDeps:
    settings: Settings
    journal: JournalDeps
    mission: MissionDeps
    planner: ThreadPoolExecutor  # one worker: planning and TTS renders are serialised
    health: dict[str, Callable[[], bool]] = field(default_factory=dict)
    closers: list[Callable[[], None]] = field(default_factory=list)
    static_dir: Path | None = None
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    run_workers: bool = True
    plan_on_startup: bool = True
    retry_interval_s: float = 30.0
    expiry_interval_s: float = 3600.0
    warm_throttle_s: float = 60.0
    rendering: set[str] = field(default_factory=set)
    render_lock: threading.Lock = field(default_factory=threading.Lock)
    last_warm: float = float("-inf")

    @property
    def clock(self) -> Clock:
        return self.journal.clock

    def today(self) -> dt.date:
        return self.clock().astimezone(self.settings.tz).date()

    def groups(self) -> list[str]:
        """Configured family group names, de-duplicated, config order."""
        return list(dict.fromkeys(self.settings.device_groups.values()))

    def connect(self) -> Any:
        return self.journal.connect()

    def close(self) -> None:
        self.planner.shutdown(wait=False, cancel_futures=True)
        for closer in self.closers:
            try:
                closer()
            except Exception as exc:
                log.warning("close_failed", exc_type=type(exc).__name__)


def build_deps(
    settings: Settings,
    *,
    ollama: TranscriberLike | None = None,
    llama: WriterLike | None = None,
    tts: TtsLike | None = None,
    notifier: Notifier | None = None,
    clock: Clock | None = None,
    **options: Any,
) -> AppDeps:
    """Wire the graphs. Anything not passed is built from settings (real clients).

    ``options`` are :class:`AppDeps` fields (``run_workers``, ``plan_on_startup``, ...).
    """
    closers: list[Callable[[], None]] = []
    health: dict[str, Callable[[], bool]] = {}

    if ollama is None:
        real_ollama = OllamaClient(settings)
        ollama = real_ollama
        health["ollama"] = real_ollama.health
        closers.append(real_ollama.close)
    if llama is None and settings.writer_backend is WriterBackend.LLAMA:
        real_llama = LlamaServerClient(settings)
        llama = real_llama
        health["writer"] = real_llama.health
        closers.append(real_llama.close)
    if tts is None:
        real_tts = TtsClient(settings)
        tts = real_tts
        health["tts"] = real_tts.health
        closers.append(real_tts.shutdown)
    if notifier is None:
        notifier = NtfyNotifier(settings) if settings.ntfy_topic else NullNotifier()

    now = clock or utc_now
    planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sair-plan")

    def connect() -> Any:
        return db.connect(settings.db_path)

    def schedule(group: str, date: dt.date) -> None:
        submit_guarded(
            planner, lambda: plan_mission(group, date, mission, replace=False), "plan_mission"
        )

    mission = MissionDeps(
        settings=settings, connect=connect, ollama=ollama, tts=tts, scheduler=schedule, clock=now
    )
    trigger = post_walk_trigger(mission, lambda fn: submit_guarded(planner, fn, "plan_next"))
    journal = JournalDeps(
        settings=settings,
        connect=connect,
        ollama=ollama,
        llama=llama,
        notifier=notifier,
        mission_trigger=trigger,
        clock=now,
    )
    return AppDeps(
        settings=settings,
        journal=journal,
        mission=mission,
        planner=planner,
        health=health,
        closers=closers,
        static_dir=REPO_ROOT / "web" / "dist",
        **options,
    )
