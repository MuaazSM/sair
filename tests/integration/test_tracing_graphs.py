"""§T NFR-P7 / PRD §10: the graphs emit one transaction per run, one span per node and an
``llm`` span per journal model call, through the real SDK with a fake transport (dev/eval only)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import SecretStr

from hub import db, tracing
from hub.graphs.mission import plan_mission
from hub.schemas import WalkStatus
from hub.settings import Settings
from tests.integration.conftest import Env
from tests.unit._graphfix import EN_20S
from tests.unit.test_tracing_prod_off import FAKE_DSN, CapturingTransport

# NFR-P7: Sentry settings refuse the prod data dir (~/.sair); never created here.
SYNTHETIC = "/nonexistent/sair-synthetic"

pytestmark = pytest.mark.integration

GROUP = "Mom and Dad"


@pytest.fixture
def cap() -> Iterator[Any]:
    transport = CapturingTransport.make()
    s = Settings(  # type: ignore[call-arg]
        _env_file=None, sair_env="eval", sentry_dsn=SecretStr(FAKE_DSN), sair_data_dir=SYNTHETIC
    )
    assert tracing.init_tracing(s, transport=transport) is True
    yield transport
    tracing.shutdown_tracing()


def _transactions(transport: Any) -> list[dict[str, Any]]:
    tracing.shutdown_tracing()  # flush
    out = []
    for env in transport.envelopes:
        for item in env.items:
            p = item.payload.json
            if p and p.get("type") == "transaction":
                out.append(p)
    return out


def _ops(tx: dict[str, Any]) -> list[tuple[str, str]]:
    return [(s.get("op"), s.get("description")) for s in tx.get("spans", [])]


def test_journal_and_mission_runs_are_traced(cap: Any, make_env: Callable[..., Env]) -> None:
    env = make_env()
    wid = env.post_walk().json()["walk_id"]
    assert env.wait_walk(wid)["status"] == "done"
    conn = db.connect(env.deps.settings.db_path)
    try:  # past the seed phase: a miss serves a fallback and schedules planning
        for i in range(3):
            db.insert_walk(conn, walk_id=f"x{i}", group=GROUP, recorded_at=env.clock())
            db.update_walk_status(conn, f"x{i}", WalkStatus.DONE)
    finally:
        conn.close()
    time.sleep(0.3)  # let the post-walk mission run (seed phase) finish
    out = plan_mission(GROUP, env.deps.today(), env.deps.mission, replace=True)
    assert out is not None and env.ollama.planner_calls
    txs = _transactions(cap)
    names = [t.get("transaction") for t in txs]
    assert "journal" in names and "mission" in names

    journal = next(t for t in txs if t.get("transaction") == "journal")
    ops = _ops(journal)
    for node in ("ingest", "transcribe", "write_entry", "ground_check", "store"):
        assert ("graph.node", node) in ops, node
    assert sum(1 for op, _ in ops if op == "llm") >= 2  # transcription + writer
    assert journal["contexts"]["trace"]["data"]["walk_id"] == wid

    # the post-walk run (seed phase) and the planner run
    missions = [_ops(t) for t in txs if t.get("transaction") == "mission"]
    assert all(("graph.node", "mission_cache") in m for m in missions)
    planned = [m for m in missions if ("graph.node", "mission_plan") in m]
    assert planned

    blob = json.dumps(txs)
    for word in ("koel", "temple", EN_20S[:30]):  # no transcript/entry text in any payload
        assert word not in blob
