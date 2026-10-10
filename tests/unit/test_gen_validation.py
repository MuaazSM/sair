"""Unit tests for training/gen_transcripts.py validation (synthetic data only)."""

from __future__ import annotations

from typing import Any

from training import gen_transcripts as gt

SEED: dict[str, Any] = {
    "id": "t001",
    "group": "Mom and Dad",
    "place_type": "the park",
    "season": "winter",
    "mission": {"type": "sense", "text": "Listen for birds. What do you hear?"},
    "mission_outcome": "done",
    "noticed": [
        {"subject": "koel", "kind": "sound", "certainty": "sure", "place": None},
        {"subject": "crow", "kind": "animal", "certainty": "unsure", "place": None},
    ],
    "lures": ["sparrow"],
    "language_mix": "english",
    "style": [],
}


def make_item(
    transcript: str, facts: list[dict[str, Any]], mspan: str | None = "we did it"
) -> dict[str, Any]:
    return {
        "item_id": "t001-00",
        "seed_id": "t001",
        "group": "Mom and Dad",
        "mission_text": SEED["mission"]["text"],
        "transcript": transcript,
        "gold": {"facts": facts, "mission_result": "done", "mission_span": mspan},
        "meta": {"language_mix": "english", "style": [], "hard": False, "lures": ["sparrow"]},
    }


def fact(span: str, subject: str, certainty: str = "sure") -> dict[str, Any]:
    kind = "sound" if subject == "koel" else "animal"
    return {"span": span, "kind": kind, "subject": subject, "certainty": certainty, "place": None}


GOOD = "Evening walk was nice. We heard a koel singing. Maybe a crow on the wall. We did it."


def good_facts() -> list[dict[str, Any]]:
    return [
        fact("We heard a koel singing", "koel"),
        fact("Maybe a crow on the wall", "crow", "unsure"),
    ]


def test_valid_item_passes() -> None:
    assert gt.validate_item(make_item(GOOD, good_facts()), SEED) is None


def test_span_must_be_exact_substring() -> None:
    facts = good_facts()
    facts[0]["span"] = "We heard a koel sing"
    assert gt.validate_item(make_item(GOOD, facts), SEED) == "span_not_in_transcript"


def test_span_match_uses_grounding_normalisation() -> None:
    facts = good_facts()
    facts[0]["span"] = "  WE  heard a KOEL singing "
    assert gt.validate_item(make_item(GOOD, facts), SEED) is None


def test_subject_must_be_in_span() -> None:
    facts = good_facts()
    facts[0]["span"] = "Evening walk was nice"
    assert gt.validate_item(make_item(GOOD, facts), SEED) == "subject_not_in_span"


def test_hedge_in_sure_span_is_dropped_not_relabelled() -> None:
    item = make_item(GOOD, good_facts())
    item["gold"]["facts"][1]["certainty"] = "sure"
    assert gt.validate_item(item, SEED) == "hedge_label_mismatch"
    assert item["gold"]["facts"][1]["certainty"] == "sure"  # untouched


def test_unsure_without_cue_is_dropped() -> None:
    facts = good_facts()
    facts[1]["span"] = "a crow on the wall"
    assert gt.validate_item(make_item(GOOD, facts), SEED) == "unsure_without_cue"


def test_mission_span_must_be_in_transcript() -> None:
    item = make_item(GOOD, good_facts(), mspan="we finished the mission")
    assert gt.validate_item(item, SEED) == "mission_span_not_in_transcript"


def test_word_count_bounds() -> None:
    assert gt.validate_item(make_item("We did it", []), SEED) == "word_count"
    long = " ".join(["word"] * 130) + " we did it"
    assert gt.validate_item(make_item(long, []), SEED) == "word_count"


def test_very_short_note_allowed() -> None:
    item = make_item("Bas, we did it", [])
    item["meta"]["style"] = ["very_short"]
    assert gt.validate_item(item, SEED) is None


def test_name_heuristic() -> None:
    assert gt.likely_names("Today Ramesh came with us") == ["Ramesh"]
    assert gt.likely_names("We met Sharma near the gate") == ["Sharma"]
    # sentence-initial, Mom/Dad, I, speaker labels and lexicon words are fine
    assert gt.likely_names("Dad: Nice walk. Then Mom said I liked the Koel.") == []
    assert gt.likely_names("Dad: umm. Mom: achha.") == []


def test_name_in_transcript_drops_item() -> None:
    t = GOOD + " We met Ramesh."
    assert gt.validate_item(make_item(t, good_facts()), SEED) == "likely_proper_noun"


def test_lure_leak_dropped() -> None:
    t = GOOD + " Also a sparrow."
    assert gt.validate_item(make_item(t, good_facts()), SEED) == "lure_leaked"


def test_prompt_leak_dropped() -> None:
    t = GOOD + " mission_span ok"
    assert gt.validate_item(make_item(t, good_facts()), SEED) == "prompt_leak"


def test_dedupe_near_identical() -> None:
    a = {"item_id": "a", "transcript": "We heard a koel singing near the park today."}
    b = {"item_id": "b", "transcript": "we heard a koel singing near the park today!"}
    c = {"item_id": "c", "transcript": "Nothing much, just walked and came back for dinner."}
    kept, dropped = gt.dedupe([a, b, c])
    assert [k["item_id"] for k in kept] == ["a", "c"]
    assert dropped == ["b"]


def test_hedge_position() -> None:
    assert gt.hedge_position("maybe a crow on the wall") == "start"
    assert gt.hedge_position("a crow on the wall maybe") == "end"
    assert gt.hedge_position("a big crow maybe on the wall") == "middle"
    assert gt.hedge_position("a crow on the wall") is None


def test_assemble_builds_exact_spans() -> None:
    v = gt.Variant("t001", 0, [0, 1], [], False, {1: ("maybe", "start")})
    parsed = {
        "segments": [
            {"role": "other", "text": "Nice walk."},
            {"role": "item_0", "text": "Dad: We heard a koel."},
            {"role": "item_1", "text": "A crow."},
            {"role": "mission", "text": "We did it."},
        ]
    }
    built = gt.assemble(SEED, v, parsed)
    assert isinstance(built, dict)
    assert built["gold"]["facts"][0]["span"] == "We heard a koel."
    assert built["gold"]["mission_span"] == "We did it."
    assert built["transcript"].startswith("Nice walk. Dad: We heard")
    parsed["segments"].pop(2)
    assert gt.assemble(SEED, v, parsed) == "missing_span"


def test_variants_are_deterministic() -> None:
    assert gt.make_variant(SEED, 3) == gt.make_variant(SEED, 3)
    assert gt.item_seed("s001", 1) == gt.item_seed("s001", 1)
    assert gt.build_prompt(SEED, gt.make_variant(SEED, 2)) == gt.build_prompt(
        SEED, gt.make_variant(SEED, 2)
    )


def test_target_positions_cycle() -> None:
    got = {gt.target_position("t001", 0, i) for i in range(3)}
    assert got == {"start", "middle", "end"}
    assert gt.target_position("t001", 4, 1) == gt.target_position("t001", 4, 1)


def test_hedge_position_mismatch_dropped() -> None:
    item = make_item(GOOD, good_facts())
    item["meta"]["gen"] = {"hedge_targets": ["end"], "number_fact": None, "hard_kind": ""}
    assert gt.validate_item(item, SEED) == "hedge_position_mismatch"
    item["meta"]["gen"]["hedge_targets"] = ["start"]
    assert gt.validate_item(item, SEED) is None


def test_numbers_missing_dropped() -> None:
    item = make_item(GOOD, good_facts())
    item["meta"]["gen"] = {"hedge_targets": [], "number_fact": 0, "hard_kind": ""}
    assert gt.validate_item(item, SEED) == "numbers_missing"
    t = GOOD.replace("a koel", "two koels")
    facts = good_facts()
    facts[0]["span"] = "We heard two koels singing"
    assert gt.validate_item(make_item(t, facts) | {"meta": item["meta"]}, SEED) is None


def test_requested_detail_must_not_be_stated_in_a_fact() -> None:
    item = make_item(GOOD, good_facts())
    item["meta"]["gen"] = {
        "hedge_targets": [],
        "number_fact": None,
        "hard_kind": "detail",
        "hard_text": "say how many koels there were",
    }
    assert gt.validate_item(item, SEED) is None
    t = GOOD
    item2 = make_item(t, [fact("Maybe a crow on the wall", "crow", "unsure")])
    item2["meta"]["gen"] = {
        "hedge_targets": [],
        "number_fact": None,
        "hard_kind": "detail",
        "hard_text": "mention the koel",
    }
    assert gt.validate_item(item2, SEED) is None


def test_hard_variants_are_about_twenty_percent() -> None:
    n = sum(gt.make_variant({**SEED, "id": f"x{k:03d}"}, 0).hard for k in range(1000))
    assert 150 <= n <= 250


def test_assemble_inserts_hedge_at_target_positions() -> None:
    for pos, expect in (
        ("start", "Maybe a crow on the wall."),
        ("end", "a crow on the wall maybe."),
        ("middle", "a crow maybe on the wall"),
    ):
        v = gt.Variant("t001", 0, [1], [], False, {1: ("maybe", pos)})
        segs = (
            [{"role": "item_0_a", "text": "a crow"}, {"role": "item_0_b", "text": "on the wall"}]
            if pos == "middle"
            else [{"role": "item_0", "text": "a crow on the wall."}]
        )
        built = gt.assemble(
            SEED, v, {"segments": segs + [{"role": "mission", "text": "We did it."}]}
        )
        assert isinstance(built, dict)
        span = built["gold"]["facts"][0]["span"]
        assert span.casefold() == expect.casefold().replace("maybe a crow", "maybe a crow")
        assert span in built["transcript"]
