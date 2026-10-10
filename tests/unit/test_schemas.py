"""Structural schema tests: PRD §9.1-9.3, NFR-P4 (no input in errors), NFR-P5 (dates only)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hub import schemas as S
from hub.schemas import (
    GROUNDING_CODES,
    MISSION_SAFETY_CODES,
    PIPELINE_CODES,
    BlockCategory,
    Certainty,
    Fact,
    FactKind,
    MissionPlan,
    MissionResult,
    MissionType,
    ReasonCode,
    StatusResponse,
    VerifiedFact,
    WriterOutput,
    parse_mission_plan,
    parse_writer_output,
    reasons_to_instructions,
)

ROOT = Path(__file__).resolve().parents[2]
MARKER = "ZQX-CANARY-7731"

PRD_91 = """{
  "facts": [
    {"span": "koel was calling near the temple", "kind": "sound",
     "subject": "koel", "certainty": "sure", "place": "near the temple"}
  ],
  "mission_result": "done",
  "mission_span": "gulmohar pe phir se phool aa gaye",
  "entry": "Mom and Dad walked past the temple and heard a koel calling."
}"""
PRD_92 = (
    '{"type": "return", "references": ["f_123"], '
    '"text": "On Tuesday you heard a koel near the temple. Is it there again?"}'
)

API_MODELS: list[type[BaseModel]] = [
    S.MissionTodayResponse,
    S.WalkAccepted,
    S.WalkStatusResponse,
    S.FactChip,
    S.DiaryFact,
    S.DiaryEntry,
    S.DiarySummary,
    S.DiaryResponse,
    S.FeedbackRequest,
    S.StatusResponse,
    S.LatencyStat,
    S.GroupCounts,
    S.Healthz,
]
ALL_MODELS: list[type[BaseModel]] = [
    *API_MODELS,
    S.Fact,
    S.WriterOutput,
    S.VerifiedFact,
    S.MissionPlan,
    S.Mission,
]


def _load_gen() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gen_schemas", ROOT / "scripts/gen_schemas.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _leaf_types(tp: Any) -> list[Any]:
    """Every concrete type reachable from an annotation (unions, containers, models)."""
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        return _leaf_types(typing.get_args(tp)[0])
    if origin is not None:
        out: list[Any] = []
        for arg in typing.get_args(tp):
            out.extend(_leaf_types(arg))
        return out
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        out = [tp]
        for f in tp.model_fields.values():
            out.extend(_leaf_types(f.annotation))
        return out
    return [tp]


# ------------------------------------------------------------------ PRD examples


def test_prd_91_example_parses() -> None:
    res = parse_writer_output(PRD_91)
    assert res.ok and res.reasons == ()
    out = res.value
    assert isinstance(out, WriterOutput)
    assert out.facts[0] == Fact(
        span="koel was calling near the temple",
        kind=FactKind.SOUND,
        subject="koel",
        certainty=Certainty.SURE,
        place="near the temple",
    )
    assert out.mission_result is MissionResult.DONE


def test_prd_92_example_parses() -> None:
    res = parse_mission_plan(PRD_92)
    assert isinstance(res.value, MissionPlan)
    assert res.value.type is MissionType.RETURN
    assert res.value.references == ["f_123"]


def test_place_and_mission_span_optional() -> None:
    raw = (
        '{"facts":[{"span":"a b","kind":"sight","subject":"a","certainty":"unsure"}],'
        '"mission_result":"not_mentioned","entry":"Mom walked."}'
    )
    out = parse_writer_output(raw).value
    assert out is not None and out.facts[0].place is None and out.mission_span is None


# ------------------------------------------------------------------ structural


def test_extra_keys_forbidden_and_frozen() -> None:
    data = json.loads(PRD_91)
    data["mood"] = "happy"
    assert parse_writer_output(json.dumps(data)).reasons == (ReasonCode.SCHEMA_INVALID,)
    out = parse_writer_output(PRD_91).value
    assert out is not None
    with pytest.raises(ValidationError):
        out.entry = "x"  # type: ignore[misc]


def test_too_many_facts_code() -> None:
    data = json.loads(PRD_91)
    data["facts"] = data["facts"] * 7
    res = parse_writer_output(json.dumps(data))
    assert res.value is None and res.reasons == (ReasonCode.TOO_MANY_FACTS,)
    data["facts"] = data["facts"][:6]
    assert parse_writer_output(json.dumps(data)).ok


@pytest.mark.parametrize("raw", ["", "not json", "{", "[1,2]", "null", '{"facts": '])
def test_invalid_json_codes(raw: str) -> None:
    res = parse_writer_output(raw)
    assert res.value is None
    assert set(res.reasons) <= {ReasonCode.INVALID_JSON, ReasonCode.SCHEMA_INVALID}
    assert res.reasons


def test_bad_enum_and_empty_span_are_schema_invalid() -> None:
    data = json.loads(PRD_91)
    data["facts"][0]["kind"] = "colour"
    assert parse_writer_output(json.dumps(data)).reasons == (ReasonCode.SCHEMA_INVALID,)
    data = json.loads(PRD_91)
    data["facts"][0]["span"] = ""
    assert parse_writer_output(json.dumps(data)).reasons == (ReasonCode.SCHEMA_INVALID,)


def test_runaway_entry_rejected() -> None:
    data = json.loads(PRD_91)
    data["entry"] = "word " * 1000
    assert parse_writer_output(json.dumps(data)).reasons == (ReasonCode.SCHEMA_INVALID,)


def test_semantic_rules_not_in_pydantic() -> None:
    # Grounding (not Pydantic) handles: missing mission_span with done, 1-word entries.
    raw = '{"facts":[],"mission_result":"done","entry":"Hi"}'
    assert parse_writer_output(raw).ok


def test_mission_plan_bad_type() -> None:
    res = parse_mission_plan('{"type":"walk","references":[],"text":"x"}')
    assert res.reasons == (ReasonCode.SCHEMA_INVALID,)


# ------------------------------------------------------------------ privacy NFR-P4


@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_model_hides_input_and_forbids_extra(model: type[BaseModel]) -> None:
    assert model.model_config.get("hide_input_in_errors") is True
    assert model.model_config.get("extra") == "forbid"
    assert model.model_config.get("frozen") is True


def test_validation_error_str_has_no_input() -> None:
    data = json.loads(PRD_91)
    data["facts"][0]["kind"] = MARKER
    data["mission_result"] = MARKER
    data[MARKER] = MARKER
    data["entry"] = MARKER * 200
    with pytest.raises(ValidationError) as ei:
        WriterOutput.model_validate_json(json.dumps(data))
    assert MARKER not in str(ei.value)
    assert MARKER not in repr(ei.value)
    with pytest.raises(ValidationError) as ei2:
        WriterOutput.model_validate(data)
    assert MARKER not in str(ei2.value)
    with pytest.raises(ValidationError) as ei3:
        MissionPlan.model_validate({"type": MARKER, "text": MARKER, "references": [MARKER * 99]})
    assert MARKER not in str(ei3.value)


def test_invalid_json_error_has_no_input() -> None:
    with pytest.raises(ValidationError) as ei:
        WriterOutput.model_validate_json('{"entry": "' + MARKER + '" garbage')
    assert MARKER not in str(ei.value)


def test_parse_result_carries_codes_only() -> None:
    res = parse_writer_output('{"entry": "' + MARKER + '"')
    assert res.value is None
    assert all(isinstance(c, ReasonCode) for c in res.reasons)
    assert MARKER not in repr(res)


# ------------------------------------------------------------------ dates NFR-P5


@pytest.mark.parametrize("model", [*API_MODELS, S.VerifiedFact, S.Mission])
def test_no_datetime_or_time_fields(model: type[BaseModel]) -> None:
    for leaf in _leaf_types(model):
        assert leaf not in (dt.datetime, dt.time, dt.timedelta), (model.__name__, leaf)


def test_dates_reject_datetimes_with_time() -> None:
    with pytest.raises(ValidationError):
        S.WalkStatusResponse.model_validate({"status": "done", "date": "2026-10-10T18:42:00"})


def test_verified_fact_shape() -> None:
    assert set(VerifiedFact.model_fields) == {
        "id",
        "walk_id",
        "walk_date",
        "group",
        "kind",
        "subject",
        "place",
        "certainty",
        "span",
    }
    assert "asked_again_on" not in VerifiedFact.model_fields
    assert "entry" not in VerifiedFact.model_fields


# ------------------------------------------------------------------ FR-O2 /status


_STATUS_NAME_KEYED = {"model_health", "latency", "per_group"}  # keys are model/node/group names
_SAFE_LEAVES = (bool, int, float, dt.date, type(None))


def _check_no_free_text(tp: Any, path: str) -> None:
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        _check_no_free_text(typing.get_args(tp)[0], path)
        return
    if origin is dict:
        key, val = typing.get_args(tp)
        field = path.split(".")[1]
        key_ok = (key is str and field in _STATUS_NAME_KEYED) or (
            isinstance(key, type) and issubclass(key, Enum)
        )
        assert key_ok, f"{path}: dict key {key}"
        _check_no_free_text(val, path + "[]")
        return
    if origin in (list, tuple, typing.Union, types.UnionType):
        for arg in typing.get_args(tp):
            _check_no_free_text(arg, path)
        return
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        for name, f in tp.model_fields.items():
            _check_no_free_text(f.annotation, f"{path}.{name}")
        return
    if isinstance(tp, type) and issubclass(tp, Enum):
        return
    assert isinstance(tp, type) and issubclass(tp, _SAFE_LEAVES) and tp is not str, (
        f"{path}: free-text-capable type {tp}"
    )


def test_status_response_has_no_free_text() -> None:
    _check_no_free_text(StatusResponse, "StatusResponse")


def test_status_free_text_detector_works() -> None:
    class Bad(BaseModel):
        note: str

    with pytest.raises(AssertionError):
        _check_no_free_text(Bad, "Bad")


def test_status_example_validates() -> None:
    s = StatusResponse(
        model_health={"ollama": True, "llama_server": False},
        latency={"transcribe": S.LatencyStat(n=3, p50_ms=4000.0, target_p95_ms=30000.0)},
        per_group={"Mom and Dad": S.GroupCounts(walks=2, facts=5)},
        reason_counts={ReasonCode.TEMPLATE_FALLBACK: 1},
        fallback_count=0,
        raw_audio_on_disk=0,
        last_canary_result=S.CanaryResult.PASSED,
    )
    assert json.loads(s.model_dump_json())["reason_counts"] == {"template_fallback": 1}


# ------------------------------------------------------------------ reason codes


def test_reason_codes_are_snake_and_partitioned() -> None:
    for c in ReasonCode:
        assert c.name == c.name.upper() and c.value == c.name.lower()
    assert set(ReasonCode) == GROUNDING_CODES | MISSION_SAFETY_CODES | PIPELINE_CODES
    assert not GROUNDING_CODES & MISSION_SAFETY_CODES
    assert not (GROUNDING_CODES | MISSION_SAFETY_CODES) & PIPELINE_CODES


def test_block_categories_match_spec() -> None:
    assert {b.value for b in BlockCategory} == {
        "road_crossing",
        "touch_animal",
        "feed_animal",
        "give_anything",
        "follow_animal",
        "approach_stranger",
        "enter_property",
        "climb",
        "run",
        "phone_outside",
    }


def test_block_categories_match_walk_words_if_present() -> None:
    path = ROOT / "data/walk_words.yaml"
    if not path.exists():
        pytest.skip("data/walk_words.yaml not written yet")
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("blocklist") if isinstance(data, dict) else None
    assert isinstance(block, list), "walk_words.yaml: blocklist must be a list of {id, patterns}"
    ids = [item["id"] for item in block]
    assert len(ids) == len(set(ids))
    assert set(ids) == {b.value for b in BlockCategory}


def test_every_check_code_has_instruction() -> None:
    for c in GROUNDING_CODES | MISSION_SAFETY_CODES:
        assert len(reasons_to_instructions([c])) == 1, c
    for c in PIPELINE_CODES:
        assert reasons_to_instructions([c]) == [], c


def test_instructions_are_short_fixed_and_deduped() -> None:
    codes = [ReasonCode.NEW_NUMBER, ReasonCode.NEW_NUMBER, ReasonCode.ENTRY_TOO_LONG]
    out = reasons_to_instructions(codes)
    assert out == reasons_to_instructions([ReasonCode.NEW_NUMBER, ReasonCode.ENTRY_TOO_LONG])
    for c in ReasonCode:
        for line in reasons_to_instructions([c]):
            assert len(line.split()) <= 14 and "{" not in line


# ------------------------------------------------------------------ JSON schemas


@pytest.mark.parametrize("rel", ["schemas/entry.schema.json", "schemas/mission.schema.json"])
def test_committed_schema_equals_generated(rel: str) -> None:
    gen = _load_gen()
    assert (ROOT / rel).read_text(encoding="utf-8") == gen.render(gen.TARGETS[rel])


@pytest.mark.parametrize("rel", ["schemas/entry.schema.json", "schemas/mission.schema.json"])
def test_schema_is_inlined_and_ollama_friendly(rel: str) -> None:
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert "$ref" not in text and "$defs" not in text
    assert text.endswith("}\n")
    allowed = {
        "type",
        "enum",
        "properties",
        "required",
        "items",
        "maxItems",
        "anyOf",
        "additionalProperties",
        "title",
        "description",
        "default",
        "maxLength",
        "minLength",
    }

    def walk(node: Any, in_props: bool = False) -> None:
        if isinstance(node, dict):
            if not in_props:
                assert set(node) <= allowed, set(node) - allowed
            for k, v in node.items():
                walk(v, in_props=(k == "properties" and not in_props))
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(text))


def test_entry_schema_property_order_puts_entry_last() -> None:
    props = list(json.loads((ROOT / "schemas/entry.schema.json").read_text())["properties"])
    assert props == ["facts", "mission_result", "mission_span", "entry"]


def test_planner_schema_is_generated_from_mission_plan_without_seed() -> None:
    gen = _load_gen()
    planner = json.loads(gen.render(gen.TARGETS["schemas/mission.schema.json"]))
    assert "seed" not in planner["properties"]["type"]["enum"]
    assert planner["required"] == ["type", "references", "text"]
    # the pydantic model itself is unchanged: SEED is a valid stored mission type
    full = MissionPlan.model_json_schema()
    assert "seed" in json.dumps(full)
