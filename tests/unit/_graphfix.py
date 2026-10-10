"""Stubs and builders for graph tests: no real models, tmp DB and dirs, synthetic data only."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hub import db
from hub.graphs.journal import JournalDeps
from hub.graphs.mission import MissionDeps
from hub.models._http import CallStats
from hub.models.errors import Permanent, Transient
from hub.prompts.builders import ChatMessage
from hub.schemas import MissionResult
from hub.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
GROUP = "Mom and Dad"
EN_20S = (FIXTURES / "en_20s.txt").read_text(encoding="utf-8").strip()
KOEL_SPAN = "A koel was calling from somewhere near the temple"

GOOD_ENTRY = json.dumps(
    {
        "facts": [
            {"span": KOEL_SPAN, "kind": "sound", "subject": "koel", "certainty": "sure",
             "place": "near the temple"},
        ],
        "mission_result": "not_mentioned",
        "entry": "Mom and Dad heard a koel calling near the temple.",
    }
)  # fmt: skip
# Valid fact, but the entry invents a detail (check 4 fails; the fact is still verified).
BAD_ENTRY = json.dumps(
    {
        "facts": [
            {"span": KOEL_SPAN, "kind": "sound", "subject": "koel", "certainty": "sure",
             "place": "near the temple"},
        ],
        "mission_result": "not_mentioned",
        "entry": "Mom and Dad heard a koel and saw a peacock near the temple.",
    }
)  # fmt: skip
GOOD_PLAN_SENSE = json.dumps(
    {"type": "sense", "references": [], "text": "Listen for one bird you can hear from the path."}
)
UNSAFE_PLAN = json.dumps(
    {"type": "sense", "references": [], "text": "Climb the tree near the temple at noon."}
)

STATS = CallStats(latency_s=0.01, prompt_tokens=11, completion_tokens=7)

Step = str | BaseException | Callable[..., str]


class FakeClock:
    def __init__(self, start: dt.datetime | None = None) -> None:
        self.now = start or dt.datetime(2026, 10, 10, 13, 0, tzinfo=dt.UTC)  # 18:30 IST

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, seconds: float) -> dt.datetime:
        self.now = self.now + dt.timedelta(seconds=seconds)
        return self.now


class StubOllama:
    """Transcriber + JSON chat stub. Steps are consumed in order; the last one repeats."""

    def __init__(
        self,
        transcripts: Sequence[Step] = (EN_20S,),
        writer: Sequence[Step] = (GOOD_ENTRY,),
        planner: Sequence[Step] = (GOOD_PLAN_SENSE,),
    ) -> None:
        self.transcripts = list(transcripts)
        self.writer = list(writer)
        self.planner = list(planner)
        self.transcribe_calls = 0
        self.writer_calls: list[list[ChatMessage]] = []
        self.planner_calls: list[list[ChatMessage]] = []
        self.schemas: list[dict[str, Any]] = []
        self.warm_calls = 0

    @staticmethod
    def _next(steps: list[Step], n: int, *args: Any) -> str:
        step = steps[min(n, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(*args)
        return step

    def transcribe_with_stats(self, wav: Path) -> tuple[str, CallStats]:
        assert wav.is_file()
        n = self.transcribe_calls
        self.transcribe_calls += 1
        return self._next(self.transcripts, n, wav), STATS

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        role: Any = "writer",
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> tuple[str, CallStats]:
        self.schemas.append(schema)
        if role == "planner":
            self.planner_calls.append(list(messages))
            return self._next(self.planner, len(self.planner_calls) - 1, messages), STATS
        self.writer_calls.append(list(messages))
        return self._next(self.writer, len(self.writer_calls) - 1, messages), STATS

    def warm(self) -> bool:
        self.warm_calls += 1
        return True


class StubLlama:
    def __init__(self, outputs: Sequence[Step] = (GOOD_ENTRY,)) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, CallStats]:
        self.calls.append({"grammar": grammar, "max_tokens": max_tokens})
        return StubOllama._next(self.outputs, len(self.calls) - 1, messages), STATS


class StubTts:
    def __init__(self, fail: BaseException | None = None) -> None:
        self.fail = fail
        self.calls = 0

    def render(self, text: str, out_path: Path) -> Path:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ID3fake")
        return out_path


class RecNotifier:
    def __init__(self) -> None:
        self.walks: list[tuple[Any, ...]] = []
        self.health: list[tuple[Any, ...]] = []

    def walk_done(self, group: str, facts: int, mission_result: MissionResult) -> None:
        self.walks.append((group, facts, mission_result))

    def health_ping(self, *args: Any, **kwargs: Any) -> None:
        assert not kwargs
        self.health.append(args)


def make_settings(tmp: Path, **kw: Any) -> Settings:
    base: dict[str, Any] = {
        "sair_env": "test",
        "sair_data_dir": tmp / "data",
        "sair_db_path": tmp / "data" / "sair.db",
        "sair_tmp_dir": tmp / "tmp",
        "sair_audio_dir": tmp / "audio",
    }
    base.update(kw)
    s = Settings(_env_file=None, **base)  # type: ignore[call-arg]
    s.tmp_dir.mkdir(parents=True, exist_ok=True)
    db.init_db(s.db_path).close()
    return s


def connect_fn(s: Settings) -> Callable[[], Any]:
    return lambda: db.connect(s.db_path)


def journal_deps(
    tmp: Path,
    *,
    ollama: StubOllama | None = None,
    llama: StubLlama | None = None,
    notifier: RecNotifier | None = None,
    trigger: Callable[[str], None] | None = None,
    clock: FakeClock | None = None,
    **settings: Any,
) -> JournalDeps:
    s = make_settings(tmp, **settings)
    return JournalDeps(
        settings=s,
        connect=connect_fn(s),
        ollama=ollama or StubOllama(),
        llama=llama,
        notifier=notifier or RecNotifier(),
        mission_trigger=trigger or (lambda g: None),
        clock=clock or FakeClock(),
    )


def mission_deps(
    tmp: Path,
    *,
    ollama: StubOllama | None = None,
    tts: StubTts | None = None,
    scheduler: Callable[[str, dt.date], None] | None = None,
    clock: FakeClock | None = None,
    settings: Settings | None = None,
) -> MissionDeps:
    s = settings or make_settings(tmp)
    return MissionDeps(
        settings=s,
        connect=connect_fn(s),
        ollama=ollama or StubOllama(),
        tts=tts,
        scheduler=scheduler or (lambda g, d: None),
        clock=clock or FakeClock(),
    )


def stage_audio(deps: JournalDeps, name: str, walk_id: str) -> Path:
    """Copy a synthetic fixture into the tmp dir as the app would (<tmp>/<walk_id><ext>)."""
    src = FIXTURES / name
    dst = deps.settings.tmp_dir / f"{walk_id}{src.suffix}"
    shutil.copyfile(src, dst)
    return dst


def add_received_walk(
    deps: JournalDeps, walk_id: str, group: str = GROUP, mission_id: str | None = None
) -> None:
    conn = deps.connect()
    try:
        db.insert_walk(conn, walk_id=walk_id, group=group, recorded_at=deps.clock(),
                       mission_id=mission_id)  # fmt: skip
    finally:
        conn.close()


def files_in(d: Path) -> list[Path]:
    return [p for p in d.rglob("*") if p.is_file()] if d.is_dir() else []


def walk_row(deps: JournalDeps | MissionDeps, walk_id: str) -> Any:
    conn = deps.connect()
    try:
        return db.get_walk_row(conn, walk_id)
    finally:
        conn.close()


def transient() -> Transient:
    return Transient(None, "test")


def permanent() -> Permanent:
    return Permanent(None, "test")
