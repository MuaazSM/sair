"""Contamination chains with stub writer/planner (PRD §13.4)."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from eval import chains
from eval.backends import CachedPlanner, CachedWriter, CacheMode, StubPlanner, StubWriter
from eval.chains import (
    ChainDay,
    ChainSpec,
    contamination,
    load_chain,
    run_chain,
    run_chains,
    summarise,
)
from eval.prose_planner import build_prose_planner_prompt
from hub.prompts.builders import REGEN_HEADER, ChatMessage, MissionText, WalkerGroup

EXAMPLES = Path(chains.__file__).resolve().parent / "data" / "chains" / "examples"
G = "Mom and Dad"
CROW_SPAN = "we heard a crow near the gate"

SPEC = ChainSpec(
    id="t01",
    group=G,
    start_date=dt.date(2026, 10, 5),
    first_mission="Listen for one bird you can hear from the path.",
    days=[
        ChainDay(transcript=CROW_SPAN + ", bas itna hi"),
        ChainDay(transcript="nothing much today"),
        ChainDay(transcript="we just walked"),
    ],
)


def invent_parrot(messages: list[ChatMessage], grammar: bool) -> str:
    """Day-1 entry invents a parrot; the only verified fact is the crow."""
    user = messages[-1].content
    if "crow" in user:
        return json.dumps(
            {
                "facts": [
                    {
                        "span": CROW_SPAN,
                        "kind": "sound",
                        "subject": "crow",
                        "certainty": "sure",
                        "place": "near the gate",
                    }
                ],
                "mission_result": "not_mentioned",
                "mission_span": None,
                "entry": "Mom and Dad heard a crow and a green parrot near the gate.",
            }
        )
    return json.dumps(
        {
            "facts": [],
            "mission_result": "not_mentioned",
            "mission_span": None,
            "entry": "Mom and Dad went for a quiet walk.",
        }
    )


def parrot_planner(messages: list[ChatMessage]) -> str:
    """Invents "parrot" whenever the prompt shows it; else revisits the crow fact."""
    user = messages[-1].content
    if "parrot" in user:
        return json.dumps(
            {"type": "sense", "references": [], "text": "Can you spot the parrot near the gate?"}
        )
    ids = re.findall(r'"id": "(f\d+_\d+)"', user)
    if ids:
        return json.dumps(
            {
                "type": "return",
                "references": [ids[0]],
                "text": "Can you spot the crow near the gate?",
            }
        )
    return json.dumps(
        {"type": "sense", "references": [], "text": "Can you spot the crow near the gate?"}
    )


def test_prose_arm_contaminated_fact_arm_clean() -> None:
    prose = run_chain(
        SPEC,
        "D",
        StubWriter(invent_parrot),
        StubPlanner(parrot_planner),
        memory="prose",
        provenance=False,
    )
    fact = run_chain(
        SPEC,
        "D",
        StubWriter(invent_parrot),
        StubPlanner(parrot_planner),
        memory="fact",
        provenance=False,
    )
    assert prose.first_contaminated_day == 1
    assert prose.days[0].orphans == ["parrot"] and prose.days[0].contaminated_raw
    assert fact.first_contaminated_day is None
    assert fact.days[0].mission_out_source == "model"
    # the stub repeats itself, so the FR-M3 repeat rule sends day 2 to regen + fallback
    assert "mission_repeated" in fact.days[1].planner_attempts[0].reasons
    assert fact.days[1].mission_out_source == "fallback"
    assert fact.days[0].mission_out == "Can you spot the crow near the gate?"
    # the served mission feeds the next day's writer
    assert prose.days[1].mission_in == prose.days[0].mission_out


def test_provenance_on_blocks_prose_contamination_with_fallback() -> None:
    res = run_chain(
        SPEC,
        "D",
        StubWriter(invent_parrot),
        StubPlanner(parrot_planner),
        memory="prose",
        provenance=True,
    )
    d1 = res.days[0]
    assert d1.contaminated_raw is True  # what the planner wrote
    assert d1.mission_out_source == "fallback" and not d1.contaminated  # what was served
    assert (
        len(d1.planner_attempts) == 2 and "noun_not_provenanced" in d1.planner_attempts[0].reasons
    )


def test_fact_arm_prompt_never_contains_entry_prose() -> None:
    planner = StubPlanner(parrot_planner)
    run_chain(SPEC, "D", StubWriter(invent_parrot), planner, memory="fact", provenance=False)
    assert all("parrot" not in m.content for call in planner.calls for m in call)


def test_plus_writer_removes_contamination_source() -> None:
    def writer(messages: list[ChatMessage], grammar: bool) -> str:
        return invent_parrot(messages, grammar)

    res = run_chain(
        SPEC,
        "D+",
        StubWriter(writer),
        StubPlanner(parrot_planner),
        memory="prose",
        provenance=False,
    )
    assert res.days[0].writer_origin == "template"
    assert res.first_contaminated_day is None


def test_contamination_definition() -> None:
    ts = ["koyal bol rahi thi mandir ke paas"]
    assert contamination("Can you hear the koel near the temple?", ts, G) == ([], False)
    assert contamination("Can you spot the parrot?", ts, G)[0] == ["parrot"]
    # generic walk words and weekday names are not contamination
    assert contamination("On Tuesday, look at the sky above the gate.", ts, G)[0] == []
    # numbers are concrete details
    assert contamination("Can you count the 3 koels?", ts, G)[0] == ["three"]
    # orphan that only echoes a mission shown to the planner
    orphans, echo = contamination(
        "Can you count the scooters?",
        ts,
        G,
        shown_missions=["Count the parked scooters in the lane."],
    )
    assert orphans and echo


def test_prose_builder_is_eval_only_and_typed() -> None:
    msgs = build_prose_planner_prompt(
        [(dt.date(2026, 10, 5), "Mom and Dad saw a parrot.")],
        [MissionText("Listen for one bird.")],
        group=WalkerGroup(G),
        today=dt.date(2026, 10, 6),
    )
    assert "Diary entries" in msgs[-1].content and "parrot" in msgs[-1].content
    with pytest.raises(TypeError):
        build_prose_planner_prompt([], [], group=G, today=dt.date(2026, 10, 6))  # type: ignore[arg-type]
    hub_src = Path(chains.__file__).resolve().parent.parent / "hub"
    assert not any("prose_planner" in p.read_text("utf-8") for p in hub_src.rglob("*.py"))


def test_example_chain_files_load() -> None:
    specs = [load_chain(p) for p in sorted(EXAMPLES.glob("*.yaml"))]
    assert [s.id for s in specs] == ["ex01", "ex02"]


def test_resumable_with_cache_and_checkpoints(tmp_path: Path) -> None:
    cache, out = tmp_path / "cache", tmp_path / "out"
    w_inner, p_inner = StubWriter(invent_parrot), StubPlanner(parrot_planner)
    writers = {"sair_writer": CachedWriter(w_inner, cache, CacheMode.RESUME)}
    planner = CachedPlanner(p_inner, cache, CacheMode.RESUME)
    first = run_chains([SPEC], ["D"], writers, planner, out_dir=out)
    assert len(list((out / "runs").glob("*.json"))) == 4
    n_calls = len(w_inner.calls) + len(p_inner.calls)

    # simulate a kill: drop one checkpoint; --resume re-runs it from cache only
    victim = sorted((out / "runs").glob("*.json"))[0]
    victim.unlink()
    w2, p2 = StubWriter([]), StubPlanner([])
    again = run_chains(
        [SPEC],
        ["D"],
        {"sair_writer": CachedWriter(w2, cache, CacheMode.CACHED)},
        CachedPlanner(p2, cache, CacheMode.CACHED),
        out_dir=out,
        resume=True,
    )
    assert len(w2.calls) + len(p2.calls) == 0 and n_calls > 0
    assert [r.to_dict() for r in again] == [r.to_dict() for r in first]


def test_summary_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = summarise(
        [
            run_chain(
                SPEC,
                "D",
                StubWriter(invent_parrot),
                StubPlanner(parrot_planner),
                memory=m,
                provenance=False,
            )
            for m in ("prose", "fact")
        ],
        n_resamples=200,
    )
    by_mem = {r["memory"]: r for r in rows}
    assert by_mem["prose"]["contaminated_by_day"][1] == 1.0
    assert by_mem["fact"]["contaminated_by_day"][3] == 0.0
    assert by_mem["prose"]["first_contaminated_day"] == [1]

    cdir = tmp_path / "chains"
    cdir.mkdir()
    (cdir / "t01.yaml").write_text((EXAMPLES / "ex01.yaml").read_text("utf-8"), encoding="utf-8")
    code = chains.main(
        [
            "--chains-dir",
            str(cdir),
            "--systems",
            "D",
            "--out",
            str(tmp_path / "o"),
            "--resamples",
            "100",
        ],
        writers={"sair_writer": StubWriter(invent_parrot)},
        planner=StubPlanner(parrot_planner),
    )
    assert code == 0
    assert (tmp_path / "o" / "contamination.csv").is_file()
    printed = capsys.readouterr().out
    assert "crow" not in printed and "parrot" not in printed


# --------------------------------------------------------------------------- PRD §13.4 headline

CHAINS_DIR = Path(chains.__file__).resolve().parent / "data" / "chains"


def regen_rescue_planner(messages: list[ChatMessage]) -> str:
    """First attempt invents a parrot AND breaks a non-provenance rule (time of day), so the
    safety regen replaces it with a clean mission: raw and served measures diverge."""
    user = messages[-1].content
    if REGEN_HEADER in user:
        return json.dumps(
            {"type": "sense", "references": [], "text": "Can you spot the crow near the gate?"}
        )
    return json.dumps(
        {
            "type": "sense",
            "references": [],
            "text": "Can you spot the parrot near the gate this evening?",
        }
    )


def test_headline_is_raw_planner_output_when_filter_off() -> None:
    # PRD §13.4: "Headline runs use the raw planner output with the FR-M4 provenance filter
    # OFF in both arms" -> the regen/fallback must not hide what the planner first wrote.
    res = run_chain(
        SPEC,
        "A",
        StubWriter(invent_parrot),
        StubPlanner(regen_rescue_planner),
        memory="fact",
        provenance=False,
    )
    d1 = res.days[0]
    assert "time_of_day" in d1.planner_attempts[0].reasons
    assert d1.mission_out == "Can you spot the crow near the gate?"
    assert d1.contaminated_raw is True and d1.raw_orphans == ["parrot"]
    assert d1.contaminated_served is False and d1.served_orphans == []
    assert d1.contaminated is True and d1.orphans == ["parrot"]  # headline = raw
    assert res.headline_measure == "raw_planner_output"
    assert res.first_contaminated_day == 1 == res.first_contaminated_day_raw
    assert res.first_contaminated_day_served is None
    dumped = res.to_dict()
    assert dumped["first_contaminated_day"] == 1 and dumped["first_contaminated_day_served"] is None
    assert chains.ChainResult.from_dict(dumped).to_dict() == dumped

    row = summarise([res], n_resamples=100)[0]
    assert row["headline_measure"] == "raw_planner_output" and row["headline"] is True
    assert row["contaminated_by_day"][1] == 1.0 == row["raw_contaminated_by_day"][1]
    assert row["served_contaminated_by_day"][1] == 0.0
    assert row["first_contaminated_day"] == [1] and row["served_first_contaminated_day"] == [None]
    assert row["pct_contaminated_by_last_day"]["point"] == 1.0
    assert row["served_pct_contaminated_by_last_day"]["point"] == 0.0


def test_headline_is_served_mission_when_filter_on() -> None:
    # PRD §13.4: filter-on runs are reported separately as "what production does".
    res = run_chain(
        SPEC,
        "A",
        StubWriter(invent_parrot),
        StubPlanner(regen_rescue_planner),
        memory="fact",
        provenance=True,
    )
    d1 = res.days[0]
    assert d1.contaminated_raw is True
    assert d1.contaminated is d1.contaminated_served is False
    assert res.headline_measure == "served_mission"
    assert res.first_contaminated_day is None and res.first_contaminated_day_raw == 1
    row = summarise([res], n_resamples=100)[0]
    assert row["headline_measure"] == "served_mission" and row["headline"] is False
    assert row["contaminated_by_day"][1] == 0.0 and row["raw_contaminated_by_day"][1] == 1.0


def test_unparsed_raw_output_is_counted_not_hidden() -> None:
    res = run_chain(
        SPEC,
        "A",
        StubWriter(invent_parrot),
        StubPlanner(["not json"]),
        memory="fact",
        provenance=False,
    )
    assert all(d.contaminated_raw is None and not d.contaminated for d in res.days)
    assert all(d.mission_out_source == "fallback" for d in res.days)
    row = summarise([res], n_resamples=100)[0]
    assert row["raw_unparsed_missions"] == len(SPEC.days)
    assert row["fallback_missions"] == len(SPEC.days)


def test_old_checkpoint_without_split_fields_still_loads() -> None:
    old = {
        "chain_id": "t01",
        "system": "A",
        "memory": "fact",
        "provenance": False,
        "days": [
            {
                "day": 1,
                "mission_in": "m",
                "writer_origin": "template",
                "writer_passed": True,
                "writer_unsupported": 0,
                "entry": None,
                "verified_facts": 0,
                "planner_attempts": [],
                "contaminated": False,
                "contaminated_raw": None,
            }
        ],
    }
    assert chains.ChainResult.from_dict(old).first_contaminated_day is None


def test_stub_cli_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "o"
    code = chains.main(
        [
            "--chains-dir",
            str(EXAMPLES),
            "--systems",
            "A",
            "--stub",
            "--out",
            str(out),
            "--resamples",
            "100",
        ]
    )
    assert code == 0
    rows = json.loads((out / "contamination.json").read_text("utf-8"))["rows"]
    assert {(r["memory"], r["provenance"]) for r in rows} == {
        (m, p) for m in ("prose", "fact") for p in ("off", "on")
    }
    assert all(r["n_chains"] == 2 for r in rows)
    assert all(r["raw_unparsed_missions"] == 0 for r in rows)
    assert chains.main(["--chains-dir", str(EXAMPLES), "--systems", "D", "--stub"]) == 2
    printed = capsys.readouterr().out
    assert "crow" not in printed and "koel" not in printed


# --------------------------------------------------------------------------- committed chain set

_GROUPS = {"Mom and Dad", "Mom", "Dad"}
_ALLOWED_CAPS = {"I", "Mom", "Dad", "I'm", "I've", "I'd", "I'll", "Ok", "OK"}


def test_committed_chain_set() -> None:
    from hub.safety import check_fallback_text

    specs = chains.load_chains(CHAINS_DIR)
    assert [s.id for s in specs] == [f"c{i:02d}" for i in range(1, 21)]
    for s in specs:
        assert s.version == 1 and s.group in _GROUPS and s.place_type
        assert len(s.days) == 7
        assert check_fallback_text(s.first_mission).passed, s.id
        for day in s.days:
            assert 1 <= len(day.transcript.split()) <= 90  # short notes allowed (PRD §12.1)
            # no names: a capitalised Latin token only at a sentence start or a role word
            for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z']+\b", day.transcript):
                assert m.group(0) in _ALLOWED_CAPS, (s.id, "capitalised token")
