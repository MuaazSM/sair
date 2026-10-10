"""FR-M1, FR-M4, FR-M5, FR-M6 (a-d), FR-M6b: mission safety filter (hub/safety.py).

Table-driven. Every rule has at least one pass and one fail case. Missions here are
synthetic and use generic places only (NFR-P6).
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
import unicodedata

import pytest

from hub import safety
from hub.safety import SafetyResult, check_fallback_text, check_mission
from hub.schemas import (
    MISSION_SAFETY_CODES,
    BlockCategory,
    Certainty,
    FactKind,
    MissionPlan,
    MissionType,
    ReasonCode,
    VerifiedFact,
)

R = ReasonCode
B = BlockCategory
SURE = Certainty.SURE
UNSURE = Certainty.UNSURE


def vf(
    fid: str,
    subject: str,
    place: str | None = None,
    *,
    certainty: Certainty = SURE,
    kind: FactKind = FactKind.SIGHT,
    span: str | None = None,
) -> VerifiedFact:
    return VerifiedFact(
        id=fid,
        walk_id="w_1",
        walk_date=dt.date(2026, 10, 6),
        group="Mom and Dad",
        kind=kind,
        subject=subject,
        place=place,
        certainty=certainty,
        span=span or " ".join(p for p in (subject, place) if p),
    )


FACTS: dict[str, VerifiedFact] = {
    f.id: f
    for f in (
        vf("f_koel", "koel", "near the temple", kind=FactKind.SOUND),
        vf(
            "f_koyal_unsure",
            "koyal",
            "school ke paas",
            certainty=UNSURE,
            kind=FactKind.SOUND,
            span="shayad koyal thi school ke paas",
        ),
        vf("f_koel_dev", "कोयल", "मंदिर के पास", kind=FactKind.SOUND),
        vf("f_gul", "gulmohar", "near the temple", kind=FactKind.PLANT),
        vf("f_dog", "dog", "near the gate", kind=FactKind.ANIMAL),
        vf("f_mango_dev", "आम का पेड़", None, kind=FactKind.PLANT),
        vf("f_bell", "bell", "near the temple", kind=FactKind.SOUND),
    )
}


def plan(text: str, refs: list[str] | None = None, t: MissionType | None = None) -> MissionPlan:
    refs = refs or []
    return MissionPlan(
        type=t or (MissionType.RETURN if refs else MissionType.SENSE),
        references=refs,
        text=text,
    )


def run(text: str, refs: list[str] | None = None, **kw: object) -> SafetyResult:
    return check_mission(plan(text, refs), FACTS, **kw)  # type: ignore[arg-type]


# ----------------------------------------------------------------- passing missions

PASSING: list[tuple[str, list[str]]] = [
    # PRD §9.2 example: context sentence + copula question.
    ("On Tuesday you heard a koel near the temple. Is it there again?", ["f_koel"]),
    # PRD FR-M5 example, unsure fact posed as a question; alias koel <-> koyal.
    ("You thought you heard a koel near the school. Can you find out?", ["f_koyal_unsure"]),
    ("Can you listen for the koel near the temple again?", ["f_koel"]),
    # Cross-script alias: fact in Devanagari, mission in English.
    ("Can you hear the koel near the temple again?", ["f_koel_dev"]),
    # Multi-word Devanagari alias: आम का पेड़ <-> mango tree.
    ("Can you find the mango tree on your walk?", ["f_mango_dev"]),
    ("Last week you saw a gulmohar near the temple. Is it still flowering?", ["f_gul"]),
    ("Look at the dog near your gate.", ["f_dog"]),
    # Imperatives with a subordinate clause ("you pass", "as you walk") and generic nouns.
    ("Count the gates you pass on your walk.", []),
    ("As you walk, notice which trees have new leaves.", []),
    # stand/pause as a preamble to one noticing verb is one task.
    ("Stand by the gate and listen to the birds around you.", []),
    ("Pause at a corner and watch the clouds above you.", []),
    # Colour and number words are not nouns; FR-M4 is about nouns.
    ("Spot three yellow flowers on your way.", []),
    ("Can you smell something sweet near the gulmohar?", ["f_gul"]),
    ("Do you hear the bell near the temple again?", ["f_bell"]),
    ("Compare the sounds you hear at the start and the end of your walk.", []),
]


@pytest.mark.parametrize(("text", "refs"), PASSING)
def test_passing_missions(text: str, refs: list[str]) -> None:
    res = run(text, refs)
    assert res.passed, res.reasons
    assert res.reasons == ()
    assert res.block_categories == ()


# ---------------------------------------------------------------- FR-M6(a) verbs

VERB_FAILS: list[str] = [
    "Remember the gate you passed on your walk.",
    "Touch the leaves of a tree near your gate.",
    "Pick a leaf you like on your walk.",
    "Can you pick a flower on your walk?",
    "Try to spot a bird near your gate.",
    # No task at all: a context sentence only.
    "On your walk you passed many trees.",
    # Opener verb the parser hangs off a later clause; still the task verb.
    "Touch the wall, is it warm where you stand?",
    # Allowlisted root, but the action asked of "you" in a relative clause is not.
    "Find a flower you can touch on your walk.",
]


@pytest.mark.parametrize("text", VERB_FAILS)
def test_verb_not_allowed(text: str) -> None:
    res = run(text)
    assert not res.passed
    assert R.VERB_NOT_ALLOWED in res.reasons


@pytest.mark.parametrize(
    "text",
    [
        # en_core_web_sm parses "pass" as ROOT and "Pause" as a noun; the opener rule applies.
        "Pause for a moment at one spot you usually pass quickly.",
        "Listen closely, is that a bird near you?",
        # "well" is a lexicon alias (a well) but here an adverb: not a concrete noun.
        "Notice how well you can hear the birds.",
    ],
)
def test_parser_quirks_pass(text: str) -> None:
    res = check_fallback_text(text)
    assert res.passed, res.reasons


def test_reported_past_verbs_do_not_count_as_task() -> None:
    # "passed" and "thought" are context; the task verb is "find".
    res = run("Last time you passed a gulmohar near the temple. Can you find it again?", ["f_gul"])
    assert res.passed, res.reasons


# ---------------------------------------------------------------- FR-M6(b) blocklist

BLOCK_FAILS: list[tuple[str, BlockCategory]] = [
    ("Give the puppy a biscuit on your way.", B.FEED_ANIMAL),
    ("Pet the dog near your gate.", B.TOUCH_ANIMAL),
    ("Follow the cows you see on the lane.", B.FOLLOW_ANIMAL),
    ("Ask the watchman what you can hear.", B.APPROACH_STRANGER),
    ("Walk into the society garden and notice what you see.", B.ENTER_PROPERTY),
    ("Climb the low wall and look at the sky with your eyes.", B.CLIMB),
    ("Jog to the corner and count the trees you pass.", B.RUN),
    ("Take a photo of the sky on your walk.", B.PHONE_OUTSIDE),
    ("Bring your phone and listen for birds.", B.PHONE_OUTSIDE),
    ("Cross the road and look at the gate you like.", B.ROAD_CROSSING),
    ("Can you feed the pigeons near your gate?", B.FEED_ANIMAL),
    ("Hand a flower to the person you see first.", B.GIVE_ANYTHING),
    ("Stand in the middle of the road and listen to you", B.ROAD_CROSSING),
    ("Sadak paar karke dekho, what do you see?", B.ROAD_CROSSING),
    ("कुत्ते को छूना, can you?", B.TOUCH_ANIMAL),
]


@pytest.mark.parametrize(("text", "category"), BLOCK_FAILS)
def test_blocklist_paraphrases(text: str, category: BlockCategory) -> None:
    res = run(text)
    assert not res.passed
    assert R.BLOCKLISTED in res.reasons
    assert category in res.block_categories


def test_look_at_dog_is_not_blocklisted() -> None:
    res = check_fallback_text("Look at the dog near the gate.")
    assert R.BLOCKLISTED not in res.reasons
    assert res.block_categories == ()


def test_blocklist_runs_on_nfc_text() -> None:
    # U+095C (precomposed ड़) is NFC-decomposed; the pattern must still match.
    text = "सड़क पार karo, can you see the gate?"
    assert unicodedata.normalize("NFC", text) != text
    res = check_fallback_text(text)
    assert B.ROAD_CROSSING in res.block_categories


# ----------------------------------------------------------------- FR-M4 provenance


@pytest.mark.parametrize(
    ("text", "refs"),
    [
        ("Can you hear the koel near the mosque?", ["f_koel"]),  # place not from fact
        ("Look for the red car near your gate.", []),  # car: not generic, no reference
        ("Can you spot a peacock near your gate?", []),  # lexicon noun, no reference
        ("Can you hear the koel near Lodhi Garden?", ["f_koel"]),  # proper noun
        ("Can you find the gulmohar near your gate?", ["f_koel"]),  # noun of a non-referenced fact
        ("Can you hear the कोयल near your gate?", []),  # Devanagari lexicon alias, no reference
    ],
)
def test_noun_not_provenanced(text: str, refs: list[str]) -> None:
    res = run(text, refs)
    assert R.NOUN_NOT_PROVENANCED in res.reasons
    assert not res.passed


def test_provenance_off_only_skips_noun_check() -> None:
    # PRD §13.4 headline runs: filter OFF. Every other rule still applies.
    contaminated = "Can you hear the koel near the mosque?"
    assert run(contaminated, ["f_koel"], provenance=False).passed
    still_bad = run("Can you feed the koel near the mosque?", ["f_koel"], provenance=False)
    assert R.BLOCKLISTED in still_bad.reasons
    assert R.NOUN_NOT_PROVENANCED not in still_bad.reasons


def test_reference_unknown() -> None:
    res = run("Can you hear the koel near the temple?", ["f_missing"])
    assert R.REFERENCE_UNKNOWN in res.reasons


def test_return_without_reference() -> None:
    res = check_mission(plan("Can you count the gates you pass?", [], MissionType.RETURN), FACTS)
    assert R.RETURN_WITHOUT_REFERENCE in res.reasons


# ----------------------------------------------------------------- FR-M5 unsure facts


@pytest.mark.parametrize(
    "text",
    [
        "You heard a koel near the school. Is it there again?",  # asserts as certain
        "You heard a koel near the school. Can you hear it again?",  # unhedged context
        "Listen for the koel near the school, you two.",  # not a question
        "You thought you heard a koel near the school. Find out.",  # no question mark
        "You thought you heard a koel near the school. Is it there again?",  # question lacks "you"
    ],
)
def test_unsure_not_question(text: str) -> None:
    res = run(text, ["f_koyal_unsure"])
    assert R.UNSURE_NOT_QUESTION in res.reasons


@pytest.mark.parametrize(
    "text",
    [
        "Maybe you heard a koel near the school. Can you find out?",
        "Can you hear the koel near the school again?",
    ],
)
def test_unsure_question_passes(text: str) -> None:
    assert run(text, ["f_koyal_unsure"]).passed


# ------------------------------------------------------------ addressed to the group


@pytest.mark.parametrize(
    "text",
    [
        "Dad thought he heard a koel near the school. Can you find out?",
        "Mom saw the gulmohar near the temple. Can you see it?",
        "Look at the sky above the gate.",  # no "you" at all
        "She heard the bell near the temple. Do you hear it?",
    ],
)
def test_not_addressed_to_group(text: str) -> None:
    res = run(text, ["f_koyal_unsure", "f_gul", "f_bell"])
    assert R.NOT_ADDRESSED_TO_GROUP in res.reasons


# ------------------------------------------------------------------ FR-M6b time of day


@pytest.mark.parametrize(
    "text",
    [
        "Can you hear the koel near the temple tonight?",
        "Watch the sky at sunset on your walk.",
        "Can you count the lights you see after 6 pm?",
        "This evening, look at the trees on your walk.",
        "Shaam ko, can you listen for birds?",
        "Can you count the birds you see at 5:30?",
        "Can you count the birds you see before 7 a.m.?",
        "Watch the sky change at dusk on your walk.",
    ],
)
def test_time_of_day(text: str) -> None:
    res = check_fallback_text(text)
    assert R.TIME_OF_DAY in res.reasons


@pytest.mark.parametrize(
    "text",
    [
        "I am sure you will hear it.",
        "Can you count the gates you pass?",
        "Can you spot the flamingo-pink sign, ma'am?",
    ],
)
def test_no_false_time_of_day(text: str) -> None:
    assert not safety.mentions_time_of_day(text)


# -------------------------------------------------------- FR-M1 / FR-M6(d) shape


def test_too_long() -> None:
    text = (
        "As you walk along your usual path past the gate and the trees, can you count "
        "every single bird and every flower and every sign that you see on the way back home?"
    )
    assert len(text.split()) > 30
    assert R.MISSION_TOO_LONG in check_fallback_text(text).reasons


def test_exactly_thirty_words_is_ok() -> None:
    text = (
        "As you walk your usual path today, can you count the gates, the benches and the "
        "signs you pass between the first corner and the last bend near your home?"
    )
    assert len(re.findall(r"[\w'-]+", text)) == 30
    assert check_fallback_text(text).passed
    longer = text.replace("last bend", "very last bend")
    assert R.MISSION_TOO_LONG in check_fallback_text(longer).reasons


@pytest.mark.parametrize(
    "text",
    [
        "Look at the sky. Listen for birds around you.",  # two task sentences
        "Count the gates you pass and listen for the birds.",  # two conjoined task verbs
        "Notice the trees. Count the birds. Can you hear the wind?",  # three sentences
        "Can you find a bird and count the trees you pass?",  # two tasks in one question
    ],
)
def test_multiple_tasks(text: str) -> None:
    assert R.MULTIPLE_TASKS in check_fallback_text(text).reasons


# ------------------------------------------------------------------- repeats


@pytest.mark.parametrize(
    "recent",
    [
        "Can you count the gates you pass?",
        "can you COUNT the gates you pass ?",
        "Can you count all the gates you pass?",
    ],
)
def test_mission_repeated(recent: str) -> None:
    res = run("Can you count the gates you pass?", recent_missions=[recent])
    assert R.MISSION_REPEATED in res.reasons


def test_different_recent_mission_is_not_repeat() -> None:
    res = run(
        "Can you count the gates you pass?",
        recent_missions=["Can you listen for birds near your gate?"],
    )
    assert res.passed


# -------------------------------------------------------------- fallback/seed texts


def test_fallback_text_generic_nouns_only() -> None:
    assert check_fallback_text("Can you count the gates you pass on your walk?").passed
    res = check_fallback_text("Can you find the koel near the temple?")
    assert R.NOUN_NOT_PROVENANCED in res.reasons


def test_fallback_text_blocklisted() -> None:
    res = check_fallback_text("Feed the birds you see on your walk.")
    assert not res.passed
    assert B.FEED_ANIMAL in res.block_categories


# ------------------------------------------------------------------- result hygiene


def test_reasons_are_enum_codes_only() -> None:
    res = run("Dad, feed the koel near the mosque tonight and climb the wall!", ["f_koel"])
    assert not res.passed
    assert all(isinstance(r, ReasonCode) and r in MISSION_SAFETY_CODES for r in res.reasons)
    assert len(set(res.reasons)) == len(res.reasons)
    assert "mosque" not in repr(res)


def test_spacy_loads_without_network() -> None:
    code = (
        "import socket\n"
        "def _deny(*a, **k):\n"
        "    raise RuntimeError('network blocked')\n"
        "socket.socket.connect = _deny\n"
        "socket.create_connection = _deny\n"
        "from hub.safety import check_fallback_text\n"
        "r = check_fallback_text('Can you count the gates you pass?')\n"
        "assert r.passed, r.reasons\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().endswith("ok")
