"""Eval metrics on hand-made toy cases with known answers (PRD §13.3)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from eval.backends import StubWriter
from eval.items import EvalItem, Gold, GoldFact
from eval.metrics import (
    Ratio,
    cer,
    checker_recall_details,
    checker_recall_items,
    details_per_entry,
    edit_distance,
    fact_precision,
    fact_recall,
    hedge_preservation,
    invalid_rate,
    latency_percentiles,
    match_facts,
    mission_accuracy,
    point,
    span_jaccard,
    subjects_match,
    template_fallback_rate,
    unsupported_rate,
    valid_json_no_grammar,
    wer,
)
from eval.systems import ItemRecord, run_item
from hub.schemas import Certainty, FactKind, MissionResult

G = "Mom and Dad"
T_CROW = "we heard a crow near the gate, bas itna hi"
T_KOEL = "I think it was a koel near the school, maybe. the sky was orange"


def out(
    entry: str,
    facts: list[dict[str, Any]] | None = None,
    mr: str = "not_mentioned",
    span: str | None = None,
) -> str:
    return json.dumps(
        {"facts": facts or [], "mission_result": mr, "mission_span": span, "entry": entry}
    )


def fact(
    span: str, subject: str, certainty: str = "sure", kind: str = "sound", place: str | None = None
) -> dict[str, Any]:
    return {"span": span, "kind": kind, "subject": subject, "certainty": certainty, "place": place}


def gfact(span: str, subject: str, certainty: Certainty = Certainty.SURE) -> GoldFact:
    return GoldFact(span=span, subject=subject, kind=FactKind.SOUND, certainty=certainty)


def item(
    transcript: str,
    gold_facts: list[GoldFact] | None = None,
    mr: MissionResult = MissionResult.NOT_MENTIONED,
    iid: str = "i1",
    seed: str = "s1",
) -> EvalItem:
    return EvalItem(
        id=iid,
        seed=seed,
        group=G,
        mission=None,
        transcript=transcript,
        gold=Gold(facts=gold_facts or [], mission_result=mr),
    )


def rec(system: str, it: EvalItem, *responses: str, grammar: bool = True) -> ItemRecord:
    return run_item(system, it, StubWriter(list(responses)), grammar=grammar)


# ---------------------------------------------------------------- unsupported details


def test_unsupported_rate_flags_invented_parrot() -> None:
    it = item(T_CROW, [gfact("we heard a crow near the gate", "crow")])
    r = rec(
        "D",
        it,
        out(
            "Mom and Dad heard a crow and a parrot near the gate.",
            [fact("we heard a crow near the gate", "crow", place="near the gate")],
        ),
    )
    assert unsupported_rate(r, it) == Ratio(1, 1)
    assert details_per_entry(r, it) == Ratio(1, 1)


def test_unsupported_rate_zero_for_faithful_entry() -> None:
    it = item(T_CROW)
    r = rec(
        "D",
        it,
        out(
            "Mom and Dad heard a crow near the gate.",
            [fact("we heard a crow near the gate", "crow", place="near the gate")],
        ),
    )
    assert unsupported_rate(r, it) == Ratio(0, 1)
    assert details_per_entry(r, it) == Ratio(0, 1)


def test_details_per_entry_counts_distinct_orphans() -> None:
    it = item(T_CROW)
    r = rec("D", it, out("Mom and Dad saw a parrot, a peacock and a parrot near the gate."))
    # parrot, peacock and gate are all unsupported (no verified facts): 3 distinct
    assert details_per_entry(r, it).num == 3


def test_invalid_output_excluded_from_unsupported_and_counted_invalid() -> None:
    it = item(T_CROW)
    r = rec("D", it, "not json at all")
    assert r.final_origin == "invalid"
    assert unsupported_rate(r, it) == Ratio(0, 0)
    assert invalid_rate(r, it) == Ratio(1, 1)
    assert mission_accuracy(r, it) == Ratio(0, 1)


def test_plus_system_measured_on_final_template_entry() -> None:
    it = item(T_CROW)
    bad = out("Mom and Dad saw a parrot near the gate.")
    r = rec("D+", it, bad, bad)
    assert r.regen_used and r.final_origin == "template"
    assert unsupported_rate(r, it) == Ratio(0, 1)
    assert template_fallback_rate(r, it) == Ratio(1, 1)


def test_template_fallback_rate_zero_when_first_attempt_passes() -> None:
    it = item(T_CROW)
    r = rec("B+", it, out("Mom and Dad went for a walk together."))
    assert r.final_origin == "model" and not r.regen_used
    assert template_fallback_rate(r, it) == Ratio(0, 1)


# ---------------------------------------------------------------- facts


def test_span_jaccard_known_values() -> None:
    assert span_jaccard("a crow near the gate", "a crow near the gate") == 1.0
    assert span_jaccard("crow near the gate", "crow near") == pytest.approx(2 / 4)
    assert span_jaccard("koel", "crow") == 0.0
    assert span_jaccard("Crow, near the GATE!", "crow near the gate") == 1.0


def test_subjects_match_alias_aware() -> None:
    assert subjects_match("koel", "koyal")
    assert subjects_match("koel", "कोयल")
    assert subjects_match("green parrot", "parrot")
    assert not subjects_match("crow", "koel")


def test_fact_precision_recall_toy() -> None:
    gold = [gfact("we heard a crow near the gate", "crow"), gfact("the neem tree", "neem")]
    it = item(T_CROW + " and the neem tree", gold)
    r = rec(
        "D",
        it,
        out(
            "Mom and Dad heard a crow.",
            [
                fact("we heard a crow near the gate", "crow"),
                fact("bas itna hi", "itna", kind="other"),
            ],
        ),
    )
    assert fact_precision(r, it) == Ratio(1, 2)
    assert fact_recall(r, it) == Ratio(1, 2)


def test_fact_match_needs_span_overlap() -> None:
    gold = [gfact("we heard a crow near the gate", "crow")]
    assert match_facts([gfact("crow", "crow")], gold) == []  # Jaccard 1/7 < 0.5
    assert match_facts([gfact("heard a crow near the gate", "crow")], gold) == [(0, 0)]


def test_fact_match_is_one_to_one() -> None:
    gold = [gfact("a crow near the gate", "crow")]
    pred = [gfact("a crow near the gate", "crow"), gfact("crow near the gate", "crow")]
    assert len(match_facts(pred, gold)) == 1


def test_dropped_facts_do_not_count_as_predicted() -> None:
    gold = [gfact("we heard a crow near the gate", "crow")]
    it = item(T_CROW, gold)
    r = rec("D", it, out("Mom and Dad went for a walk.", [fact("a crow on the roof", "crow")]))
    assert fact_precision(r, it) == Ratio(0, 0)
    assert fact_recall(r, it) == Ratio(0, 1)


# ---------------------------------------------------------------- hedges


def test_hedge_preserved() -> None:
    gold = [gfact("I think it was a koel near the school, maybe", "koel", Certainty.UNSURE)]
    it = item(T_KOEL, gold)
    r = rec(
        "D",
        it,
        out(
            "Mom and Dad maybe heard a koel near the school.",
            [fact("I think it was a koel near the school, maybe", "koel", "unsure")],
        ),
    )
    assert hedge_preservation(r, it) == Ratio(1, 1)


def test_hedge_lost_when_entry_states_it_as_sure() -> None:
    gold = [gfact("I think it was a koel near the school, maybe", "koel", Certainty.UNSURE)]
    it = item(T_KOEL, gold)
    r = rec(
        "D",
        it,
        out(
            "Mom and Dad heard a koel near the school.",
            [fact("I think it was a koel near the school, maybe", "koel", "unsure")],
        ),
    )
    assert hedge_preservation(r, it) == Ratio(0, 1)


def test_hedge_lost_when_mentioned_without_fact() -> None:
    gold = [gfact("I think it was a koel near the school, maybe", "koel", Certainty.UNSURE)]
    it = item(T_KOEL, gold)
    r = rec("D", it, out("Mom and Dad maybe heard a koyal."))
    assert hedge_preservation(r, it) == Ratio(0, 1)  # mentioned, but no unsure fact


def test_hedge_unmentioned_fact_excluded() -> None:
    gold = [gfact("I think it was a koel near the school, maybe", "koel", Certainty.UNSURE)]
    it = item(T_KOEL, gold)
    r = rec("D", it, out("Mom and Dad went for a walk."))
    assert hedge_preservation(r, it) == Ratio(0, 0)


# ---------------------------------------------------------------- mission / json


def test_mission_accuracy_after_check7_coercion() -> None:
    it = item(T_CROW, mr=MissionResult.DONE)
    ok = rec("D", it, out("Mom and Dad heard a crow.", mr="done", span="we heard a crow"))
    coerced = rec("D", it, out("Mom and Dad heard a crow.", mr="done", span="we saw a koel"))
    assert mission_accuracy(ok, it) == Ratio(1, 1)
    assert coerced.final_check is not None and coerced.final_check.mission_result == "not_mentioned"
    assert mission_accuracy(coerced, it) == Ratio(0, 1)


def test_valid_json_no_grammar_only_counts_grammar_off_records() -> None:
    it = item(T_CROW)
    on = rec("B", it, "{bad", grammar=True)
    off_bad = rec("B", it, "{bad", grammar=False)
    off_ok = rec("B", it, out("Mom and Dad walked."), grammar=False)
    assert valid_json_no_grammar(on, it) == Ratio(0, 0)
    assert valid_json_no_grammar(off_bad, it) == Ratio(0, 1)
    assert valid_json_no_grammar(off_ok, it) == Ratio(1, 1)


def test_system_a_is_the_no_facts_template() -> None:
    it = item(T_CROW, [gfact("we heard a crow near the gate", "crow")])
    from eval.backends import TemplateWriter

    r = run_item("A", it, TemplateWriter())
    assert r.final is not None and r.final["entry"] == "Mom and Dad went for their evening walk."
    assert unsupported_rate(r, it) == Ratio(0, 1)
    assert fact_recall(r, it) == Ratio(0, 1)
    assert template_fallback_rate(r, it) == Ratio(1, 1)


# ---------------------------------------------------------------- checker vs human


def test_checker_recall_items() -> None:
    v = checker_recall_items([(True, True), (True, False), (False, True), (False, False)])
    assert v == Ratio(1, 2)


def test_checker_recall_details_alias_aware() -> None:
    assert checker_recall_details(["koyal", "the old man"], ["koel"]) == Ratio(1, 2)
    assert checker_recall_details([], ["koel"]) == Ratio(0, 0)


# ---------------------------------------------------------------- WER / CER


def test_wer_known_values() -> None:
    assert wer("the crow was near the gate", "the crow was near the gate") == 0.0
    assert wer("the crow was near the gate", "a crow was near gate") == pytest.approx(2 / 6)
    assert wer("Crow, near the GATE.", "crow near the gate") == 0.0


def test_cer_known_values() -> None:
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert cer("ab", "") == 1.0
    assert cer("", "") == 0.0


def test_wer_cer_devanagari_safe() -> None:
    ref = "कोयल बोल रही थी।"
    assert wer(ref, "कोयल बोल रही थी") == 0.0  # danda is punctuation
    assert wer(ref, "कौआ बोल रही थी") == pytest.approx(1 / 4)
    assert 0 < cer(ref, "कौआ बोल रही थी") < 1
    # NFC: a decomposed nukta form equals the precomposed one
    assert cer("ज़", "ज़") == 0.0


def test_edit_distance_basic() -> None:
    assert edit_distance(list("kitten"), list("sitting")) == 3


# ---------------------------------------------------------------- latency / point


def test_latency_percentiles() -> None:
    recs = []
    for k in range(1, 11):
        r = ItemRecord(
            item_id=str(k), seed="s", system="D", grammar=True, latency={"writer": float(k)}
        )
        recs.append(r)
    lat = latency_percentiles(recs)
    assert lat["writer"]["n"] == 10
    assert lat["writer"]["p50_ms"] == pytest.approx(5.5)
    assert lat["writer"]["p95_ms"] == pytest.approx(9.55)


def test_point_micro_average() -> None:
    assert point([Ratio(1, 2), Ratio(0, 1), Ratio(3, 3)]) == pytest.approx(4 / 6)
    assert point([Ratio(0, 0)]) is None


def test_record_roundtrip() -> None:
    it = item(T_CROW)
    r = rec("D+", it, out("Mom and Dad saw a parrot."), out("Mom and Dad heard a crow."))
    again = ItemRecord.from_dict(json.loads(json.dumps(r.to_dict())))
    assert again == r
