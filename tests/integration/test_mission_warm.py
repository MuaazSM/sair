"""§T FR-M9: fetching the mission warms the transcriber without waiting for it."""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import pytest

from hub.graphs import mission as M
from tests.unit._graphfix import GROUP, StubOllama, mission_deps

pytestmark = pytest.mark.integration


class BlockingWarm(StubOllama):
    def __init__(self) -> None:
        super().__init__()
        self.called = threading.Event()
        self.release = threading.Event()

    def warm(self) -> bool:
        self.warm_calls += 1
        self.called.set()
        self.release.wait(10)
        return True


def test_warm_called_and_mission_flow_does_not_wait(tmp_path: Path) -> None:
    ollama = BlockingWarm()
    deps = mission_deps(tmp_path, ollama=ollama)
    t0 = time.perf_counter()
    thread = M.warm_transcriber(deps)
    mission = M.get_or_fallback_mission(GROUP, dt.date(2026, 10, 10), deps)
    elapsed = time.perf_counter() - t0
    assert mission.text and elapsed < 0.5  # served while warm-up is still blocked
    assert ollama.called.wait(2), "warm() was not called"
    assert thread.is_alive() and thread.daemon
    ollama.release.set()
    thread.join(2)
    assert ollama.warm_calls == 1 and not thread.is_alive()


def test_warm_failure_never_raises(tmp_path: Path) -> None:
    class Broken(StubOllama):
        def warm(self) -> bool:
            raise RuntimeError("down")

    thread = M.warm_transcriber(mission_deps(tmp_path, ollama=Broken()))
    thread.join(2)
    assert not thread.is_alive()
