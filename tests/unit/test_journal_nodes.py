"""One test per journal-graph node (stubbed models, tmp DB/dirs, synthetic fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub import db
from hub.graphs import journal as J
from hub.graphs.common import work_dir_for
from hub.grounding import check_entry
from hub.schemas import (
    MissionResult,
    ReasonCode,
    WalkStatus,
    WriterBackend,
    parse_writer_output,
)
from tests.unit._graphfix import (
    BAD_ENTRY,
    EN_20S,
    GOOD_ENTRY,
    GROUP,
    RecNotifier,
    StubLlama,
    StubOllama,
    add_received_walk,
    files_in,
    journal_deps,
    stage_audio,
    walk_row,
)


def _state(**kw: object) -> J.JournalState:
    base: dict[str, object] = {"walk_id": "w1", "group": GROUP, "mission_id": None,
                               "regen_count": 0, "latency_ms": {}}  # fmt: skip
    base.update(kw)
    return base  # type: ignore[return-value]


def _checked(raw: str) -> dict[str, object]:
    out = parse_writer_output(raw).value
    assert out is not None
    res = check_entry(out, EN_20S, GROUP)
    return {"output": out, "check": res, "best_check": res, "check_passed": res.passed,
            "check_reasons": res.reasons}  # fmt: skip


def test_ingest_decodes_into_work_dir(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    add_received_walk(deps, "w1")
    raw = stage_audio(deps, "en_20s.webm", "w1")
    upd = J.ingest(_state(raw_path=str(raw)), deps)
    wav = Path(str(upd["wav_path"]))
    assert wav.is_file() and wav.parent == work_dir_for(deps.settings.tmp_dir, "w1")
    assert isinstance(upd["wav_path"], str)
    assert 15.0 < float(upd["duration_s"]) < 30.0  # type: ignore[arg-type]
    assert raw.is_file()  # ingest never deletes on success; transcribe does
    assert walk_row(deps, "w1")["duration_s"] == pytest.approx(upd["duration_s"], abs=0.01)


def test_transcribe_deletes_audio_and_stores_transcript(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    add_received_walk(deps, "w1")
    raw = stage_audio(deps, "en_20s.webm", "w1")
    upd = J.ingest(_state(raw_path=str(raw)), deps)
    upd = J.transcribe(_state(raw_path=str(raw), wav_path=upd["wav_path"]), deps)
    assert upd["transcript"] == EN_20S
    assert upd["_tokens_in"] == 11 and upd["_tokens_out"] == 7
    assert files_in(deps.settings.tmp_dir) == []
    assert walk_row(deps, "w1")["transcript"] == EN_20S


def test_write_entry_gemma_uses_json_schema(tmp_path: Path) -> None:
    ollama = StubOllama()
    deps = journal_deps(tmp_path, ollama=ollama)
    upd = J.write_entry(_state(transcript=EN_20S, mission_text="Listen for one bird."), deps)
    assert upd["raw_output"] == GOOD_ENTRY and upd["writer_error"] is None
    schema = json.loads(Path("schemas/entry.schema.json").read_text(encoding="utf-8"))
    assert ollama.schemas == [schema]
    user = ollama.writer_calls[0][1].content
    assert "Walkers: Mom and Dad" in user and "Listen for one bird." in user
    assert EN_20S in user


def test_write_entry_llama_uses_grammar(tmp_path: Path) -> None:
    llama = StubLlama()
    deps = journal_deps(tmp_path, llama=llama, writer_backend=WriterBackend.LLAMA)
    upd = J.write_entry(_state(transcript=EN_20S), deps)
    assert upd["raw_output"] == GOOD_ENTRY
    grammar = Path("grammars/entry.gbnf").read_text(encoding="utf-8")
    assert llama.calls == [{"grammar": grammar, "max_tokens": 400}]


def test_ground_check_pass_and_parse_failure(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    upd = J.ground_check(_state(transcript=EN_20S, raw_output=GOOD_ENTRY), deps)
    assert upd["check_passed"] is True and upd["check_reasons"] == ()
    upd = J.ground_check(_state(transcript=EN_20S, raw_output="{not json"), deps)
    assert upd["check_passed"] is False and upd["check"] is None
    assert upd["check_reasons"] == (ReasonCode.INVALID_JSON,)


def test_regen_passes_reason_codes_as_instructions(tmp_path: Path) -> None:
    ollama = StubOllama()
    deps = journal_deps(tmp_path, ollama=ollama)
    upd = J.regen(_state(transcript=EN_20S, check_reasons=(ReasonCode.UNSUPPORTED_DETAIL,)), deps)
    assert upd["regen_count"] == 1 and upd["raw_output"] == GOOD_ENTRY
    user = ollama.writer_calls[0][1].content
    assert "Mention only things that are in the facts." in user
    assert BAD_ENTRY not in user  # previous output is never passed back


def test_template_from_best_check_and_without_check(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    upd = J.template(_state(transcript=EN_20S, **_checked(BAD_ENTRY)), deps)
    out = upd["final_output"]
    assert upd["writer"] == "template" and upd["final_passed"] is False
    assert out.entry == "Mom and Dad went for a walk. They noticed koel."  # type: ignore[union-attr]
    assert len(out.facts) == 1  # type: ignore[union-attr]
    upd = J.template(_state(transcript=EN_20S), deps)
    assert upd["final_output"].entry == "Mom and Dad went for their evening walk."  # type: ignore[union-attr]


def test_store_model_entry(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    add_received_walk(deps, "w1")
    upd = J.store(_state(transcript=EN_20S, **_checked(GOOD_ENTRY)), deps)
    assert upd["status"] is WalkStatus.DONE and upd["facts"] == 1
    row = walk_row(deps, "w1")
    assert row["status"] == "done" and row["writer"] == "gemma" and row["check_passed"] == 1
    conn = deps.connect()
    try:
        facts = conn.execute("SELECT verified FROM facts WHERE walk_id='w1'").fetchall()
        entry = conn.execute("SELECT text FROM entries WHERE walk_id='w1'").fetchone()
    finally:
        conn.close()
    assert [f["verified"] for f in facts] == [1]
    assert entry["text"] == "Mom and Dad heard a koel calling near the temple."


def test_plan_next_mission_triggers_and_never_raises(tmp_path: Path) -> None:
    calls: list[str] = []
    deps = journal_deps(tmp_path, trigger=calls.append)
    assert J.plan_next_mission(_state(), deps) == {}
    assert calls == [GROUP]

    def boom(group: str) -> None:
        raise RuntimeError("scheduler down")

    deps.mission_trigger = boom
    assert J.plan_next_mission(_state(), deps) == {}


def test_notify_metadata_only(tmp_path: Path) -> None:
    n = RecNotifier()
    deps = journal_deps(tmp_path, notifier=n)
    J.notify(_state(transcript=EN_20S, facts=2, mission_result=MissionResult.DONE), deps)
    assert n.walks == [(GROUP, 2, MissionResult.DONE)] and n.health == []
    J.notify(_state(transcript="I felt dizzy near the gate.", facts=0), deps)
    assert n.health == [(GROUP,)]


def test_graph_nodes_and_no_checkpointer(tmp_path: Path) -> None:
    graph = J.build_journal_graph(journal_deps(tmp_path))
    assert graph.checkpointer is None
    assert set(J.NODES) <= set(graph.get_graph().nodes)


def test_metrics_rows_per_node(tmp_path: Path) -> None:
    deps = journal_deps(tmp_path)
    add_received_walk(deps, "w1")
    raw = stage_audio(deps, "en_20s.webm", "w1")
    J.run_journal("w1", raw, GROUP, None, deps)
    conn = deps.connect()
    try:
        rows = conn.execute("SELECT node, tokens_in FROM metrics WHERE walk_id='w1'").fetchall()
        lat = json.loads(db.get_walk_row(conn, "w1")["latency_ms_json"])
    finally:
        conn.close()
    nodes = [r["node"] for r in rows]
    assert nodes == ["ingest", "transcribe", "write_entry", "ground_check", "store",
                     "plan_next_mission", "notify"]  # fmt: skip
    assert {r["node"]: r["tokens_in"] for r in rows}["write_entry"] == 11
    assert set(lat) == {"ingest", "transcribe", "write_entry", "ground_check"}
