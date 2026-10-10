"""Unit tests for hub/grounding.py (PRD §8 checks 1-7, templates, FR-J5).

All data here is synthetic and uses generic places (NFR-P6).
"""

from __future__ import annotations

import hashlib
import random
import time
import unicodedata
from pathlib import Path

import pytest

from hub import grounding as g
from hub.schemas import (
    GROUNDING_CODES,
    Certainty,
    EntryOrigin,
    Fact,
    FactKind,
    MissionResult,
    ReasonCode,
    WriterOutput,
)

R = ReasonCode


def F(
    span: str,
    subject: str,
    certainty: str = "sure",
    place: str | None = None,
    kind: str = "sight",
) -> Fact:
    return Fact(
        span=span,
        subject=subject,
        certainty=Certainty(certainty),
        place=place,
        kind=FactKind(kind),
    )


def W(
    entry: str,
    facts: list[Fact] | None = None,
    mission_result: MissionResult = MissionResult.NOT_MENTIONED,
    mission_span: str | None = None,
) -> WriterOutput:
    return WriterOutput(
        facts=facts or [],
        mission_result=mission_result,
        mission_span=mission_span,
        entry=entry,
    )


def run(
    entry: str,
    transcript: str,
    facts: list[Fact] | None = None,
    group: str = "Mom and Dad",
    origin: EntryOrigin = EntryOrigin.MODEL,
    **kw: object,
) -> g.CheckResult:
    return g.check_entry(W(entry, facts, **kw), transcript, group, origin=origin)  # type: ignore[arg-type]


# ----------------------------------------------------------------- normalise


def test_normalise_collapses_case_and_whitespace() -> None:
    assert g.normalise("  Koel   WAS\tcalling\n near ") == "koel was calling near"


def test_normalise_nfc_devanagari_composed_and_decomposed_equal() -> None:
    composed = "क़"  # क़ precomposed (a composition exclusion)
    decomposed = "क़"  # क + nukta
    assert composed != decomposed
    assert g.normalise(composed) == g.normalise(decomposed)
    assert unicodedata.is_normalized("NFC", g.normalise(composed))


# ----------------------------------------------------------------- check 1


def test_c1_span_exact_after_case_and_whitespace_collapse_is_verified() -> None:
    t = "Koel   was calling\nnear the TEMPLE"
    r = run(
        "Mom and Dad heard a koel near the temple.",
        t,
        [F("koel was calling near the temple", "koel", place="the temple")],
    )
    assert r.passed, r.reasons
    assert len(r.verified_facts) == 1 and r.dropped_facts == 0


def test_c1_span_not_in_transcript_drops_fact_but_does_not_fail_alone() -> None:
    r = run(
        "Mom and Dad had a quiet walk together.",
        "it was quiet today",
        [F("koel was calling", "koel")],
    )
    assert R.SPAN_NOT_IN_TRANSCRIPT in r.reasons
    assert r.verified_facts == () and r.dropped_facts == 1
    assert r.passed


def test_c1_span_nfc_decomposed_transcript_matches_composed_span() -> None:
    transcript = "क़िला के पास कोयल"  # decomposed nukta
    span = "क़िला के पास कोयल"  # precomposed
    r = run("Mom and Dad heard a koel.", transcript, [F(span, "कोयल")])
    assert r.passed, r.reasons
    assert len(r.verified_facts) == 1


def test_c1_span_must_sit_on_latin_token_boundaries() -> None:
    r = run("Mom and Dad had a quiet walk together.", "the category was long", [F("cat", "cat")])
    assert R.SPAN_NOT_IN_TRANSCRIPT in r.reasons
    assert r.verified_facts == ()


def test_c1_whitespace_only_span_is_dropped() -> None:
    r = run("Mom and Dad had a quiet walk together.", "a koel", [F("   ", "koel")])
    assert R.SPAN_NOT_IN_TRANSCRIPT in r.reasons
    assert r.dropped_facts == 1


# ----------------------------------------------------------------- check 2


@pytest.mark.parametrize(
    ("span", "subject", "place"),
    [
        pytest.param("koel was calling near the temple", "koel", "near the temple", id="substring"),
        pytest.param("koel was calling", "koyal", None, id="alias-koyal-koel"),
        pytest.param("कोयल बोल रही थी", "koel", None, id="alias-across-scripts"),
        pytest.param(
            "gulmohar pe phool aa gaye", "gul mohar", None, id="multiword-alias-gul-mohar"
        ),
        pytest.param("gul mohar pe phool aa gaye", "gulmohar", None, id="multiword-alias-reverse"),
        pytest.param(
            "mandir ke paas koyal bol rahi thi", "koel", "temple", id="place-alias-temple-mandir"
        ),
        pytest.param("ek kauwa baitha tha", "the crow", None, id="article-skipped"),
        pytest.param("laal gulab dikha", "red rose", None, id="two-aliases-red-rose"),
    ],
)
def test_c2_subject_and_place_pass(span: str, subject: str, place: str | None) -> None:
    r = g.check_entry(
        W("Mom and Dad had a quiet walk together.", [F(span, subject, place=place)]),
        span,
        "Mom and Dad",
    )
    assert len(r.verified_facts) == 1, r.reasons


@pytest.mark.parametrize(
    ("span", "subject", "place", "code"),
    [
        pytest.param(
            "koel was calling", "parrot", None, R.SUBJECT_NOT_IN_SPAN, id="subject-absent"
        ),
        pytest.param(
            "the category was long", "cat", None, R.SUBJECT_NOT_IN_SPAN, id="no-cat-in-category"
        ),
        pytest.param(
            "koyal bol rahi thi", "big koel", None, R.SUBJECT_NOT_IN_SPAN, id="invented-adjective"
        ),
        pytest.param(
            "koel was calling", "koel", "near the temple", R.PLACE_NOT_IN_SPAN, id="place-absent"
        ),
        pytest.param(
            "mandir ke paas koyal",
            "koel",
            "near the temple",
            R.PLACE_NOT_IN_SPAN,
            id="place-preposition-invented",
        ),
    ],
)
def test_c2_subject_or_place_fail(
    span: str, subject: str, place: str | None, code: ReasonCode
) -> None:
    r = g.check_entry(
        W("Mom and Dad had a quiet walk together.", [F(span, subject, place=place)]),
        span,
        "Mom and Dad",
    )
    assert code in r.reasons
    assert r.verified_facts == ()
    assert r.passed  # a drop alone is non-failing


# ----------------------------------------------------------------- check 3


@pytest.mark.parametrize(
    "span",
    [
        pytest.param("maybe a koel was calling", id="en-maybe"),
        pytest.param("i think it was a koel", id="en-i-think"),
        pytest.param("shayad koyal thi", id="hinglish-shayad"),
        pytest.param("koyal lagta hai", id="hinglish-lagta-hai"),
        pytest.param("शायद कोयल थी", id="hindi-shayad"),
        pytest.param("कोयल लगता है", id="hindi-lagta-hai"),
        pytest.param("kuch koyal jaisi awaaz", id="ambiguous-kuch-jaisi-treated-as-hedge"),
        pytest.param("I Think it was a KOEL", id="case-insensitive"),
    ],
)
def test_c3_hedged_span_marked_sure_is_not_verified(span: str) -> None:
    r = g.check_entry(
        W("Mom and Dad had a quiet walk together.", [F(span, "koel")]), span, "Mom and Dad"
    )
    assert R.HEDGE_CERTAINTY_MISMATCH in r.reasons
    assert r.verified_facts == ()


def test_c3_hedge_cues_respect_latin_token_boundaries() -> None:
    span = "koel near mayberry lane"  # "maybe" inside "mayberry" is not a cue
    r = g.check_entry(
        W("Mom and Dad had a quiet walk together.", [F(span, "koel")]), span, "Mom and Dad"
    )
    assert R.HEDGE_CERTAINTY_MISMATCH not in r.reasons
    assert len(r.verified_facts) == 1


def test_c3_hedged_unsure_with_entry_hedge_passes() -> None:
    t = "shayad koyal thi gate ke paas"
    r = run(
        "Mom and Dad maybe heard a koel.",
        t,
        [F("shayad koyal thi", "koyal", "unsure", kind="sound")],
    )
    assert r.passed, r.reasons
    assert len(r.verified_facts) == 1


def test_c3_entry_states_unsure_fact_as_sure_fails() -> None:
    t = "shayad koyal thi gate ke paas"
    r = run("Mom and Dad heard a koel.", t, [F("shayad koyal thi", "koyal", "unsure")])
    assert not r.passed
    assert R.HEDGE_MISSING_IN_ENTRY in r.reasons


def test_c3_hedge_in_a_different_sentence_does_not_count() -> None:
    t = "shayad koyal thi"
    r = run(
        "Mom and Dad maybe walked far. They heard a koel.",
        t,
        [F("shayad koyal thi", "koyal", "unsure")],
    )
    assert R.HEDGE_MISSING_IN_ENTRY in r.reasons


def test_c3_unsure_subject_not_mentioned_in_entry_is_fine() -> None:
    t = "shayad koyal thi"
    r = run("Mom and Dad had a quiet walk together.", t, [F("shayad koyal thi", "koyal", "unsure")])
    assert r.passed, r.reasons


def test_c3_unsure_without_cue_still_needs_entry_hedge() -> None:
    t = "a heron by the lake"
    r = run(
        "Mom and Dad saw a heron by the lake.",
        t,
        [F("a heron by the lake", "heron", "unsure", place="lake")],
    )
    assert R.HEDGE_MISSING_IN_ENTRY in r.reasons


# ----------------------------------------------------------------- check 4


def test_c4_detail_mapped_to_verified_fact_passes() -> None:
    t = "koel was calling near the temple"
    r = run("Mom and Dad heard a koel near the temple.", t, [F(t, "koel", place="near the temple")])
    assert r.passed, r.reasons
    assert r.unsupported_details == ()


def test_c4_alias_maps_entry_detail_to_fact() -> None:
    t = "koyal bol rahi thi mandir ke paas"
    r = run("Mom and Dad heard a koel near the temple.", t, [F(t, "koyal", place="mandir")])
    assert r.passed, r.reasons


def test_c4_orphan_detail_fails_and_is_collected() -> None:
    t = "koel was calling"
    r = run("Mom and Dad heard a koel and saw a parrot.", t, [F(t, "koel")])
    assert not r.passed
    assert R.UNSUPPORTED_DETAIL in r.reasons
    assert r.unsupported_details == ("parrot",)


def test_c4_dropped_fact_leaves_orphan_detail() -> None:
    t = "koel was calling"
    r = run("Mom and Dad heard a koel.", t, [F("a koel sang loudly", "koel")])
    assert R.SPAN_NOT_IN_TRANSCRIPT in r.reasons
    assert R.UNSUPPORTED_DETAIL in r.reasons
    assert not r.passed


def test_c4_detail_in_transcript_but_not_in_any_fact_still_fails() -> None:
    t = "koel was calling near the park"
    r = run("Mom and Dad heard a koel in the park.", t, [F("koel was calling", "koel")])
    assert R.UNSUPPORTED_DETAIL in r.reasons
    assert r.unsupported_details == ("park",)


def test_c4_generic_walk_word_that_is_a_lexicon_token_needs_a_fact() -> None:
    t = "we walked round the park"
    bad = run("Mom and Dad walked round the park.", t, [])
    assert R.UNSUPPORTED_DETAIL in bad.reasons
    good = run("Mom and Dad walked round the park.", t, [F("round the park", "park", kind="other")])
    assert good.passed, good.reasons


def test_c4_non_lexicon_generic_words_are_not_details() -> None:
    r = run("Mom and Dad saw trees, flowers and birds along the lane.", "nice walk", [])
    assert r.passed, r.reasons


def test_c4_group_words_are_never_details() -> None:
    r = run("Rose and Teen went for a long walk.", "nice walk", [], group="Rose and Teen")
    assert r.passed, r.reasons


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("Mom and Dad do this walk together.", id="do-verb"),
        pytest.param("Mom and Dad took one more slow round.", id="one-more"),
        pytest.param("Mom and Dad were both very happy.", id="both-happy"),
        pytest.param("Mom and Dad walked well and slowly.", id="well-adverb"),
        pytest.param("Mom and Dad stopped for the first time.", id="first-time"),
        pytest.param("Mom and Dad met many friends.", id="many-friends"),
        pytest.param("Mom and Dad shared a few laughs.", id="a-few-laughs"),
        pytest.param("Mom and Dad walked for half an hour.", id="half-an-hour"),
        pytest.param("Mom and Dad rose and walked slowly home.", id="rose-verb"),
    ],
)
def test_c4_ambiguous_english_words_not_details_alone(entry: str) -> None:
    r = run(entry, "nice walk", [])
    assert r.passed, r.reasons


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        pytest.param("Mom and Dad saw one crow on the way.", R.NEW_NUMBER, id="one-crow"),
        pytest.param("Mom and Dad saw many crows on the way.", R.NEW_NUMBER, id="many-crows"),
        pytest.param("Mom and Dad saw a pair of crows.", R.NEW_NUMBER, id="pair-of-crows"),
    ],
)
def test_c4_ambiguous_number_word_counts_next_to_counted_noun(entry: str, code: ReasonCode) -> None:
    t = "a crow sat quietly"
    r = run(entry, t, [F("a crow sat quietly", "crow", kind="animal")])
    assert code in r.reasons
    assert R.UNSUPPORTED_DETAIL in r.reasons
    assert not r.passed


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("Mom and Dad rested by the well.", id="the-well"),
        pytest.param("Mom and Dad saw a rose.", id="a-rose"),
    ],
)
def test_c4_ambiguous_word_counts_after_determiner(entry: str) -> None:
    r = run(entry, "nice walk", [])
    assert R.UNSUPPORTED_DETAIL in r.reasons


# ----------------------------------------------------------------- check 5


def test_c5_new_number_three_crows() -> None:
    t = "crows were cawing"
    r = run("Mom and Dad heard three crows cawing.", t, [F(t, "crows", kind="sound")])
    assert R.NEW_NUMBER in r.reasons
    assert not r.passed


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("Mom and Dad saw 2 parrots on the gulmohar.", id="digit-vs-hindi-word"),
        pytest.param(
            "Mom and Dad saw two parrots on the gulmohar.", id="english-word-vs-hindi-word"
        ),
    ],
)
def test_c5_number_words_match_digits_across_languages(entry: str) -> None:
    t = "do tote dikhe gulmohar pe"
    r = run(entry, t, [F(t, "tote", place="gulmohar", kind="animal")])
    assert r.passed, r.reasons


def test_c5_devanagari_digits_count_as_numbers() -> None:
    t = "३ कौए दीवार पर"
    r = run("Mom and Dad saw 3 crows.", t, [F(t, "कौए", kind="animal")])
    assert r.passed, r.reasons
    bad = run("Mom and Dad saw 4 crows.", t, [F(t, "कौए", kind="animal")])
    assert R.NEW_NUMBER in bad.reasons


def test_c5_new_time_absent_from_transcript() -> None:
    t = "aarti ho rahi thi"
    r = run("Mom and Dad heard the aarti at 6 pm.", t, [F(t, "aarti", kind="sound")])
    assert R.NEW_TIME in r.reasons
    assert not r.passed


def test_c5_time_present_in_transcript_passes() -> None:
    t = "6 baje aarti ho rahi thi"
    r = run("Mom and Dad heard the aarti at 6 pm.", t, [F(t, "aarti", kind="sound")])
    assert r.passed, r.reasons


def test_c5_day_name_absent_counts_as_new_time() -> None:
    r = run("On Tuesday Mom and Dad walked slowly.", "nice walk", [])
    assert R.NEW_TIME in r.reasons


def test_c5_new_person_name_absent() -> None:
    t = "ek uncle apne kutte ke saath the"
    r = run("Mom and Dad met Sharma uncle and his dog.", t, [F(t, "kutte", kind="animal")])
    assert R.NEW_PERSON_NAME in r.reasons
    assert R.UNSUPPORTED_DETAIL in r.reasons
    assert "Sharma" in r.unsupported_details


def test_c5_person_name_in_transcript_and_fact_passes() -> None:
    t = "sharma uncle apne kutte ke saath the"
    r = run("Mom and Dad met Sharma uncle with his dog.", t, [F(t, "sharma uncle", kind="person")])
    assert r.passed, r.reasons


def test_c5_sentence_initial_capital_and_safe_words_are_not_names() -> None:
    r = run("Later, Nani and Mom walked. Then Dad and I joined.", "nice walk", [], group="Nani")
    assert r.passed, r.reasons


# ----------------------------------------------------------------- check 6


def test_c6_template_exempt_but_model_output_fails_shape() -> None:
    entry = "Nani walked today."  # 3 words
    model = run(entry, "nice walk", [], group="Nani")
    assert R.ENTRY_TOO_SHORT in model.reasons and not model.passed
    tmpl = run(entry, "nice walk", [], group="Nani", origin=EntryOrigin.TEMPLATE)
    assert tmpl.passed, tmpl.reasons


def test_c6_word_count_bounds() -> None:
    ok5 = run("Mom and Dad walked slowly.", "x", [])
    assert ok5.passed, ok5.reasons
    words70 = "Mom and Dad walked " + "slowly " * 65 + "home."
    assert len(words70.split()) == 70
    assert run(words70, "x", []).passed
    words71 = "Mom and Dad walked " + "slowly " * 66 + "home."
    r = run(words71, "x", [])
    assert R.ENTRY_TOO_LONG in r.reasons and not r.passed


def test_c6_too_many_sentences() -> None:
    r = run("Mom and Dad walked. It was quiet. They talked. They smiled. They came home.", "x", [])
    assert R.TOO_MANY_SENTENCES in r.reasons


def test_c6_abbreviations_do_not_split_sentences() -> None:
    t = "dr. rao was walking"
    r = run(
        "Mom and Dad walked. It was quiet. They talked. They saw Dr. Rao.",
        t,
        [F(t, "dr. rao", kind="person")],
    )
    assert r.passed, r.reasons
    assert len(g._split_sentences("They left at 6 p.m. and came home. Mr. Rao waved.")) == 2


def test_c6_empty_entry() -> None:
    r = run("", "x", [])
    assert {R.ENTRY_TOO_SHORT, R.TOO_FEW_SENTENCES, R.GROUP_NOT_MENTIONED} <= set(r.reasons)


@pytest.mark.parametrize(
    ("entry", "group"),
    [
        pytest.param("They walked slowly to the end.", "Mom and Dad", id="absent"),
        pytest.param("Mom walked slowly to the end.", "Mom and Dad", id="partial-group-strict"),
    ],
)
def test_c6_group_not_mentioned(entry: str, group: str) -> None:
    r = run(entry, "x", [], group=group)
    assert R.GROUP_NOT_MENTIONED in r.reasons


def test_c6_group_match_is_case_insensitive() -> None:
    assert run("mom and dad walked slowly home.", "x", []).passed


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("Mom and Dad read about it at https://example.org today.", id="https"),
        pytest.param("Mom and Dad read about it at www.example.net today.", id="www"),
        pytest.param("Mom and Dad read about it on walks.com today.", id="dot-com"),
    ],
)
def test_c6_url_in_entry(entry: str) -> None:
    assert R.URL_IN_ENTRY in run(entry, "x", []).reasons


def test_c6_question_in_entry() -> None:
    r = run("Mom and Dad walked slowly, didn't they?", "x", [])
    assert R.QUESTION_IN_ENTRY in r.reasons and not r.passed


# ----------------------------------------------------------------- check 7


def test_c7_done_without_span_is_coerced_not_failed() -> None:
    r = run("Mom and Dad walked slowly home.", "x", [], mission_result=MissionResult.DONE)
    assert r.mission_result is MissionResult.NOT_MENTIONED
    assert R.MISSION_SPAN_MISSING in r.reasons
    assert r.mission_span is None
    assert r.passed


def test_c7_span_not_in_transcript_is_coerced() -> None:
    r = run(
        "Mom and Dad walked slowly home.",
        "gulmohar pe phir se phool aa gaye",
        [],
        mission_result=MissionResult.PARTLY,
        mission_span="we counted the crows",
    )
    assert r.mission_result is MissionResult.NOT_MENTIONED
    assert R.MISSION_SPAN_NOT_IN_TRANSCRIPT in r.reasons


def test_c7_valid_span_keeps_result() -> None:
    t = "Gulmohar pe   phir se phool aa gaye"
    r = run(
        "Mom and Dad walked slowly home.",
        t,
        [],
        mission_result=MissionResult.DONE,
        mission_span="gulmohar pe phir se phool aa gaye",
    )
    assert r.mission_result is MissionResult.DONE
    assert r.mission_span == "gulmohar pe phir se phool aa gaye"
    assert r.reasons == ()


def test_c7_not_mentioned_with_span_drops_the_span() -> None:
    r = run("Mom and Dad walked slowly home.", "abc", [], mission_span="abc")
    assert r.mission_result is MissionResult.NOT_MENTIONED and r.mission_span is None


# --------------------------------------------------------------- templates


def test_template_without_facts() -> None:
    assert (
        g.build_template_entry("Mom and Dad", [], False)
        == "Mom and Dad went for their evening walk."
    )
    assert (
        g.build_template_entry("Mom and Dad", [], True)
        == "Mom and Dad went for their evening walk."
    )


@pytest.mark.parametrize(
    ("subjects", "expected"),
    [
        pytest.param([("koel", "sure")], "They noticed koel.", id="one"),
        pytest.param([("koel", "sure"), ("crow", "sure")], "They noticed koel and crow.", id="two"),
        pytest.param(
            [("koel", "sure"), ("crow", "sure"), ("parrot", "sure")],
            "They noticed koel, crow and parrot.",
            id="three",
        ),
        pytest.param(
            [("koel", "sure"), ("heron", "unsure")],
            "They noticed koel, maybe heron.",
            id="sure-unsure",
        ),
        pytest.param(
            [("heron", "unsure"), ("owl", "unsure")],
            "They noticed maybe heron, maybe owl.",
            id="only-unsure",
        ),
        pytest.param([("koel", "sure"), ("Koel", "sure")], "They noticed koel.", id="dedupe"),
        pytest.param(
            [("koel", "sure"), ("koel", "unsure")], "They noticed maybe koel.", id="unsure-wins"
        ),
    ],
)
def test_template_joining(subjects: list[tuple[str, str]], expected: str) -> None:
    facts = [F(s, s, c) for s, c in subjects]
    assert g.build_template_entry("Mom", facts, False) == f"Mom went for a walk. {expected}"


def test_template_mission_clause_only_when_done() -> None:
    facts = [F("koel", "koel")]
    assert (
        g.build_template_entry("Dad", facts, True)
        == "Dad went for a walk and did today's mission. They noticed koel."
    )


def test_template_output_keeps_verified_facts_and_mission() -> None:
    t = "koel was calling near the temple. gulmohar pe phir se phool aa gaye"
    out = W(
        "Mom and Dad saw a parrot.",
        [F("koel was calling", "koel"), F("a parrot flew", "parrot")],
        mission_result=MissionResult.DONE,
        mission_span="gulmohar pe phir se phool aa gaye",
    )
    chk = g.check_entry(out, t, "Mom and Dad")
    assert not chk.passed
    tmpl = g.template_output("Mom and Dad", chk)
    assert tmpl.entry == "Mom and Dad went for a walk and did today's mission. They noticed koel."
    assert tmpl.facts == [F("koel was calling", "koel")]
    assert tmpl.mission_result is MissionResult.DONE
    again = g.check_entry(tmpl, t, "Mom and Dad", origin=EntryOrigin.TEMPLATE)
    assert again.passed, again.reasons


_POOL: list[tuple[str, str, str, str | None]] = [
    ("koel was calling near the temple", "koel", "sure", "near the temple"),
    ("shayad koyal thi", "koyal", "unsure", None),
    ("do tote dikhe gulmohar pe", "tote", "sure", "gulmohar"),
    ("कोयल बोल रही थी", "koel", "sure", None),
    ("maybe a heron by the lake", "heron", "unsure", "lake"),
    ("Sharma uncle was walking his dog", "Sharma uncle", "sure", None),
    ("three crows on the wall", "three crows", "sure", None),
    ("the first rain smell", "first rain", "sure", None),
    ("gul mohar pe phool aa gaye", "gulmohar", "sure", None),
    ("6 baje aarti ho rahi thi", "aarti", "sure", None),
    ("i think it was a mongoose", "mongoose", "unsure", None),
    ("laal gulab dikha", "red rose", "sure", None),
    ("ek kauwa baitha tha", "The crow", "sure", None),
    ("Raju's cat sat by the gate", "Raju's cat", "sure", "by the gate"),
    ("koyal lagta hai", "कोयल", "unsure", None),
]


def test_template_property_random_subsets_pass_checks_4_and_5() -> None:
    rng = random.Random(20261010)
    transcript = ". ".join(p[0] for p in _POOL)
    for _ in range(300):
        picked = rng.sample(_POOL, rng.randint(0, 6))
        facts = [F(s, subj, c, place=pl) for s, subj, c, pl in picked]
        group = rng.choice(["Mom and Dad", "Mom", "Dad", "Nani"])
        done = rng.random() < 0.5
        out = W(
            "x",
            facts,
            mission_result=MissionResult.DONE if done else MissionResult.NOT_MENTIONED,
            mission_span="gul mohar pe phool aa gaye" if done else None,
        )
        chk = g.check_entry(out, transcript, group)
        assert len(chk.verified_facts) == len(facts), chk.reasons
        tmpl = g.template_output(group, chk)
        res = g.check_entry(tmpl, transcript, group, origin=EntryOrigin.TEMPLATE)
        assert res.passed, (res.reasons, tmpl.entry)
        assert res.unsupported_details == ()


# --------------------------------------------------------------- extract_details


def test_extract_details_on_sentences() -> None:
    d = g.extract_details("Mom and Dad saw two parrots near the temple at 6 pm on Tuesday.")
    assert d == {"parrot", "two", "temple", "six", "time:6", "day:tuesday"}
    assert g.extract_details("They met Sharma uncle by the gul mohar.") == {
        "name:sharma",
        "gulmohar",
    }
    assert g.extract_details("Mom and Dad had a quiet walk.") == set()
    assert g.extract_details("They saw a house crow.") == {"house_crow"}


def test_extract_details_inclusive_mode_counts_ambiguous_words() -> None:
    assert "two" in g.extract_details("do tote dikhe", inclusive=True)
    assert "two" not in g.extract_details("we do walk")
    assert {"house_crow", "crow"} <= g.extract_details("a house crow", inclusive=True)


# --------------------------------------------------------------- lexicon


def test_lexicon_hash_stable_and_is_sha256_of_bytes(tmp_path: Path) -> None:
    p = tmp_path / "lex.yaml"
    p.write_bytes(b"entries:\n  - {id: zebu, category: animal, aliases: [zebu]}\n")
    h = g.lexicon_hash(p)
    assert h == hashlib.sha256(p.read_bytes()).hexdigest()
    assert g.lexicon_hash(p) == h
    assert g.lexicon_hash() == hashlib.sha256(g.DEFAULT_LEXICON.read_bytes()).hexdigest()


def test_load_lexicon_is_cached() -> None:
    assert g.load_lexicon() is g.load_lexicon()


def test_custom_lexicon_is_used(tmp_path: Path) -> None:
    p = tmp_path / "lex.yaml"
    p.write_text(
        "entries:\n  - {id: zebu, category: animal, aliases: [zebu, nandi bail]}\n",
        encoding="utf-8",
    )
    lex = g.load_lexicon(p)
    out = W("Mom and Dad saw a zebu.", [F("nandi bail baitha tha", "zebu")])
    r = g.check_entry(out, "nandi bail baitha tha", "Mom and Dad", lex=lex)
    assert r.passed, r.reasons
    assert g.extract_details("a koel", lex) == set()


# --------------------------------------------------------------- privacy and API


def test_bad_lexicon_path_error_has_no_content(tmp_path: Path) -> None:
    missing = tmp_path / "secret_place_name.yaml"
    with pytest.raises(g.GroundingError) as exc:
        g.load_lexicon(missing)
    assert "secret" not in str(exc.value) and str(missing) not in str(exc.value)


def test_malformed_lexicon_error_has_no_content(tmp_path: Path) -> None:
    p = tmp_path / "lex.yaml"
    p.write_text("entries: SECRETWORD\n", encoding="utf-8")
    with pytest.raises(g.GroundingError) as exc:
        g.load_lexicon(p)
    assert "SECRETWORD" not in str(exc.value)


def test_bad_input_types_raise_without_content() -> None:
    with pytest.raises(TypeError) as e1:
        g.check_entry({"entry": "SECRET koel"}, "SECRET transcript", "Mom")  # type: ignore[arg-type]
    assert "SECRET" not in str(e1.value)
    with pytest.raises(TypeError) as e2:
        g.check_entry(W("Mom walked."), b"SECRET", "Mom")  # type: ignore[arg-type]
    assert "SECRET" not in str(e2.value)


def test_check_entry_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    run("Mom and Dad heard three parrots at 6 pm?", "koel", [F("nope", "koel")])
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == ""


def test_failing_codes_are_grounding_codes_and_exclude_drops_and_coercions() -> None:
    assert g.FAILING_CODES <= GROUNDING_CODES
    for code in (
        R.SPAN_NOT_IN_TRANSCRIPT,
        R.SUBJECT_NOT_IN_SPAN,
        R.PLACE_NOT_IN_SPAN,
        R.HEDGE_CERTAINTY_MISMATCH,
        R.MISSION_SPAN_MISSING,
        R.MISSION_SPAN_NOT_IN_TRANSCRIPT,
    ):
        assert code not in g.FAILING_CODES
    assert R.UNSUPPORTED_DETAIL in g.FAILING_CODES and R.HEDGE_MISSING_IN_ENTRY in g.FAILING_CODES


def test_reasons_are_deduped_in_stable_order() -> None:
    r = run(
        "Mom and Dad saw a parrot and a cat?",
        "nothing here",
        [F("aaa", "a"), F("bbb", "b")],
        mission_result=MissionResult.DONE,
    )
    assert r.reasons == (
        R.SPAN_NOT_IN_TRANSCRIPT,
        R.UNSUPPORTED_DETAIL,
        R.QUESTION_IN_ENTRY,
        R.MISSION_SPAN_MISSING,
    )
    assert r.dropped_facts == 2
    assert all(isinstance(c, ReasonCode) for c in r.reasons)


def test_check_entry_warm_under_5ms() -> None:
    words = [
        *("koyal", "bol", "rahi", "thi", "mandir", "ke", "paas"),
        *("aur", "do", "tote", "gulmohar", "pe", "the", "shayad"),
    ]
    transcript = " ".join(words[i % len(words)] for i in range(200))
    facts = [
        F("koyal bol rahi thi mandir ke paas", "koyal", place="mandir"),
        F("do tote gulmohar pe", "tote"),
    ]
    entry = (
        "Mom and Dad heard a koel near the temple and saw two parrots on the gulmohar. "
        + " ".join(["They walked slowly and talked about the evening together."] * 5)
    )
    entry = " ".join(entry.split()[:70])
    out = W(entry, facts)
    g.check_entry(out, transcript, "Mom and Dad")
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        g.check_entry(out, transcript, "Mom and Dad")
    per_call_ms = (time.perf_counter() - t0) * 1000 / n
    assert per_call_ms < 5.0, per_call_ms


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param("they left at 6:30 sharp", "time:6:30", id="clock"),
        pytest.param("they left at 7 a.m. sharp", "time:7", id="am-dotted"),
        pytest.param("saat baje aarti", "time:7", id="hindi-word-baje"),
        pytest.param("seven o'clock bell", "time:7", id="o-clock-word"),
        pytest.param("बजे घंटी बजी", "time:clock", id="bare-baje"),
        pytest.param("at noon", "time:noon", id="noon"),
        pytest.param("They came back in March.", "month:march", id="month"),
        pytest.param("about 2.5 rounds", "num:2.5", id="decimal"),
        pytest.param("about 1,200 steps", "num:1200", id="thousands"),
    ],
)
def test_extract_details_times_and_numbers(text: str, key: str) -> None:
    assert key in g.extract_details(text)


def test_extract_details_lowercase_month_word_is_not_a_month_in_strict_mode() -> None:
    assert "month:may" not in g.extract_details("they may walk")
    assert "month:may" in g.extract_details("they may walk", inclusive=True)


def test_devanagari_group_words_are_excluded() -> None:
    assert g.extract_details("नानी ने कोयल सुनी", group="नानी") == {"koel"}


@pytest.mark.parametrize(
    ("subject", "ok"),
    [
        pytest.param("कोयल थी", True, id="deva-extra-word-in-span"),
        pytest.param("बड़ी कोयल", False, id="deva-invented-word"),
        pytest.param("kauwa", False, id="alias-id-absent-from-span"),
        pytest.param("?!", False, id="no-words"),
        pytest.param("the", False, id="only-article"),
    ],
)
def test_c2_devanagari_and_degenerate_subjects(subject: str, ok: bool) -> None:
    span = "शायद koyal थी"
    out = W("Mom and Dad had a quiet walk together.", [F(span, subject, "unsure")])
    r = g.check_entry(out, span, "Mom and Dad")
    assert (len(r.verified_facts) == 1) is ok, r.reasons


def test_lexicon_hash_missing_file_error_has_no_path(tmp_path: Path) -> None:
    missing = tmp_path / "secret.yaml"
    with pytest.raises(g.GroundingError) as exc:
        g.lexicon_hash(missing)
    assert "secret" not in str(exc.value)


def test_malformed_hedges_file_is_rejected(tmp_path: Path) -> None:
    lex = tmp_path / "lex.yaml"
    lex.write_text(
        "entries:\n  - {id: zebu, category: animal, aliases: [zebu]}\n", encoding="utf-8"
    )
    bad_hedges = tmp_path / "hedges.yaml"
    bad_hedges.write_text("cues: {en: [1, 2]}\nentry_hedge_words: [maybe]\n", encoding="utf-8")
    with pytest.raises(g.GroundingError):
        g.load_lexicon(lex, bad_hedges)
    bad_entry = tmp_path / "lex2.yaml"
    bad_entry.write_text("entries:\n  - {category: animal, aliases: [zebu]}\n", encoding="utf-8")
    with pytest.raises(g.GroundingError):
        g.load_lexicon(bad_entry)


def test_template_ignores_blank_subjects() -> None:
    assert g.build_template_entry("Mom", [F("koel", "...")], False) == (
        "Mom went for their evening walk."
    )
