"""GBNF grammars vs. JSON schemas (PRD 9.1, 9.2). No model needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub.schemas import parse_mission_plan, parse_writer_output
from tests.unit._gbnf import GbnfError, Grammar

ROOT = Path(__file__).resolve().parents[2]
ENTRY = Grammar.from_file(ROOT / "grammars" / "entry.gbnf")
MISSION = Grammar.from_file(ROOT / "grammars" / "mission.gbnf")


ABSENT = object()


def _fact(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "span": "koel was calling",
        "kind": "sound",
        "subject": "koel",
        "certainty": "sure",
    }
    base.update(kw)
    return base


def _entry(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "facts": [],
        "mission_result": "not_mentioned",
        "mission_span": ABSENT,
        "entry": "A quiet walk.",
    }
    base.update(kw)
    return {k: v for k, v in base.items() if v is not ABSENT}


def _dumps(obj: object, **kw: object) -> str:
    return json.dumps(obj, ensure_ascii=False, **kw)  # type: ignore[arg-type]


VALID_ENTRIES = {
    "prd_9_1": (
        '{"facts": [{"span": "koel was calling near the temple", "kind": "sound", '
        '"subject": "koel", "certainty": "sure", "place": "near the temple"}], '
        '"mission_result": "done", "mission_span": "gulmohar pe phir se phool aa gaye", '
        '"entry": "Mom and Dad walked past the temple and heard a koel calling."}'
    ),
    "zero_facts": _dumps(_entry()),
    "six_facts": _dumps(_entry(facts=[_fact(subject=f"s{i}") for i in range(6)])),
    "devanagari": _dumps(
        _entry(
            facts=[_fact(span="मंदिर के पास कोयल बोल रही थी", subject="कोयल", place="मंदिर के पास")],
            mission_result="done",
            mission_span="गुलमोहर पर फिर फूल आ गए",
            entry="Mom heard a कोयल near the मंदिर.",
        )
    ),
    "escaped_quotes": _dumps(
        _entry(
            entry='Dad said "wow" and walked on\\ slowly.\n', facts=[_fact(span='he said "koel"')]
        )
    ),
    "null_place": _dumps(_entry(facts=[_fact(place=None)])),
    "no_mission_span": _dumps(_entry(facts=[_fact()], mission_result="not_done")),
    "null_mission_span": _dumps(_entry(mission_span=None)),
    "compact": json.dumps(
        _entry(facts=[_fact(place="gate")], mission_result="partly", mission_span="x"),
        ensure_ascii=False,
        separators=(",", ":"),
    ),
    "ascii_escapes": json.dumps(_entry(entry="कोयल"), ensure_ascii=True),
    "unsure_all_kinds": _dumps(
        _entry(
            facts=[
                _fact(kind=k, certainty="unsure")
                for k in ("sight", "sound", "smell", "person", "animal", "plant")
            ]
        )
    ),
    "weather_other": _dumps(_entry(facts=[_fact(kind="weather"), _fact(kind="other", place="")])),
    "newline_pretty": (
        '{\n"facts": [],\n"mission_result": "done",\n"mission_span": "x",\n"entry": "y"\n}'
    ),
}


@pytest.mark.parametrize("name", sorted(VALID_ENTRIES))
def test_valid_entry_accepted(name: str) -> None:
    raw = VALID_ENTRIES[name]
    parsed = parse_writer_output(raw)
    assert parsed.ok, f"fixture invalid against schema: {parsed.reasons}"
    assert ENTRY.accepts(raw)


def test_prd_example_is_schema_valid() -> None:
    assert parse_writer_output(VALID_ENTRIES["prd_9_1"]).ok


INVALID_ENTRIES = {
    "seven_facts": _dumps(_entry(facts=[_fact() for _ in range(7)])),
    "bad_kind": _dumps(_entry(facts=[_fact(kind="taste")])),
    "bad_certainty": _dumps(_entry(facts=[_fact(certainty="maybe")])),
    "bad_result": _dumps(_entry(mission_result="finished")),
    "missing_entry": _dumps({"facts": [], "mission_result": "done"}),
    "missing_facts": _dumps({"mission_result": "done", "entry": "x"}),
    "wrong_order": _dumps({"mission_result": "done", "facts": [], "entry": "x"}),
    "extra_key": _dumps({"facts": [], "mission_result": "done", "entry": "x", "extra": 1}),
    "extra_fact_key": _dumps(_entry(facts=[_fact(confidence=1)])),
    "trailing_text": _dumps(_entry()) + " thanks",
    "trailing_json": _dumps(_entry()) + "{}",
    "leading_text": "Here you go: " + _dumps(_entry()),
    "unterminated_string": '{"facts": [], "mission_result": "done", "entry": "oops}',
    "not_json": "The walk was nice.",
    "markdown_fence": "```json\n" + _dumps(_entry()) + "\n```",
    "empty_span": _dumps(_entry(facts=[_fact(span="")])),
    "overlong_span": _dumps(_entry(facts=[_fact(span="a" * 401)])),
    "overlong_entry": _dumps(_entry(entry="a" * 1201)),
    "raw_newline_in_string": '{"facts": [], "mission_result": "done", "entry": "a\nb"}',
    "bad_escape": '{"facts": [], "mission_result": "done", "entry": "a\\qb"}',
    "number_place": _dumps(_entry(facts=[_fact(place=3)])),
    "single_quotes": "{'facts': [], 'mission_result': 'done', 'entry': 'x'}",
    "empty": "",
}


@pytest.mark.parametrize("name", sorted(INVALID_ENTRIES))
def test_invalid_entry_rejected(name: str) -> None:
    assert not ENTRY.accepts(INVALID_ENTRIES[name])


def test_entry_length_bounds_are_exact() -> None:
    assert ENTRY.accepts(_dumps(_entry(entry="a" * 1200)))
    assert ENTRY.accepts(
        _dumps(_entry(facts=[_fact(span="a" * 400, subject="b" * 80, place="c" * 120)]))
    )
    assert not ENTRY.accepts(_dumps(_entry(facts=[_fact(subject="b" * 81)])))
    assert not ENTRY.accepts(_dumps(_entry(facts=[_fact(place="c" * 121)])))


def _mission(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {"type": "sense", "references": [], "text": "Listen for one bird."}
    base.update(kw)
    return base


VALID_MISSIONS = {
    "prd_9_2": (
        '{"type": "return", "references": ["f_123"], '
        '"text": "On Tuesday you heard a koel near the temple. Is it there again?"}'
    ),
    "no_refs": _dumps(_mission()),
    "six_refs": _dumps(_mission(type="change", references=[f"f_{i}" for i in range(6)])),
    "count": _dumps(_mission(type="count")),
    "seed": _dumps(_mission(type="seed")),
    "devanagari": _dumps(_mission(text="कोयल फिर बोली? आप सुनिए।")),
    "escaped": _dumps(_mission(text='Say "hello" to one tree.')),
    "compact": json.dumps(_mission(references=["a", "b"]), separators=(",", ":")),
}


@pytest.mark.parametrize("name", sorted(VALID_MISSIONS))
def test_valid_mission_accepted(name: str) -> None:
    raw = VALID_MISSIONS[name]
    assert parse_mission_plan(raw).ok
    assert MISSION.accepts(raw)


INVALID_MISSIONS = {
    "bad_type": _dumps(_mission(type="quest")),
    "missing_text": _dumps({"type": "sense", "references": []}),
    "missing_refs": _dumps({"type": "sense", "text": "x"}),
    "wrong_order": _dumps({"text": "x", "type": "sense", "references": []}),
    "extra_key": _dumps({**_mission(), "score": 1}),
    "seven_refs": _dumps(_mission(references=[f"f_{i}" for i in range(7)])),
    "empty_ref": _dumps(_mission(references=[""])),
    "empty_text": _dumps(_mission(text="")),
    "overlong_text": _dumps(_mission(text="a" * 401)),
    "trailing": _dumps(_mission()) + "\n\nDone.",
    "unterminated": '{"type": "sense", "references": [], "text": "abc',
    "not_json": "Listen for a bird.",
    "refs_not_strings": _dumps(_mission(references=[1])),
}


@pytest.mark.parametrize("name", sorted(INVALID_MISSIONS))
def test_invalid_mission_rejected(name: str) -> None:
    assert not MISSION.accepts(INVALID_MISSIONS[name])


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))


def _enums(node: object) -> list[list[str]]:
    found: list[list[str]] = []
    if isinstance(node, dict):
        if "enum" in node:
            found.append(list(node["enum"]))
        for v in node.values():
            found.extend(_enums(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_enums(v))
    return found


@pytest.mark.parametrize(("schema", "grammar"), [("entry", ENTRY), ("mission", MISSION)])
def test_every_schema_enum_value_is_in_grammar(schema: str, grammar: Grammar) -> None:
    enums = _enums(_schema(schema))
    assert enums
    for values in enums:
        for v in values:
            assert f'"\\"{v}\\""' in grammar.text, f"{v} missing from {schema}.gbnf"


@pytest.mark.parametrize(("schema", "grammar"), [("entry", ENTRY), ("mission", MISSION)])
def test_every_schema_key_is_in_grammar(schema: str, grammar: Grammar) -> None:
    def keys(node: object) -> set[str]:
        out: set[str] = set()
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                out |= set(props)
            for v in node.values():
                out |= keys(v)
        elif isinstance(node, list):
            for v in node:
                out |= keys(v)
        return out

    for k in keys(_schema(schema)):
        assert f'\\"{k}\\":' in grammar.text, f"key {k} missing from {schema}.gbnf"


def test_grammars_parse_and_have_root() -> None:
    assert "root" in ENTRY.rule_names
    assert "root" in MISSION.rule_names


def test_recogniser_rejects_recursion_and_undefined() -> None:
    with pytest.raises(GbnfError):
        Grammar('root ::= "a" root')
    with pytest.raises(GbnfError):
        Grammar("root ::= missing")
