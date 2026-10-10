"""FR-M3, FR-J3, FR-J5: prompt builders are typed so entry prose cannot reach a prompt."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from hub.prompts import builders
from hub.prompts.builders import (
    PROMPT_VERSION,
    ChatMessage,
    MissionText,
    Transcript,
    WalkerGroup,
    build_planner_prompt,
    build_writer_prompt,
    prompt_fingerprint,
)
from hub.schemas import (
    Certainty,
    DiaryEntry,
    FactKind,
    MissionResult,
    ReasonCode,
    VerifiedFact,
    reasons_to_instructions,
)

CANARY = "zebra-ultramarine-4471"
TODAY = dt.date(2026, 10, 10)  # a Saturday


def fact(fid: str = "f_1", *, certainty: Certainty = Certainty.SURE) -> VerifiedFact:
    return VerifiedFact(
        id=fid,
        walk_id="w_1",
        walk_date=dt.date(2026, 10, 6),  # a Tuesday
        group="Mom and Dad",
        kind=FactKind.SOUND,
        subject="koyal",
        place="mandir ke paas",
        certainty=certainty,
        span="koyal bol rahi thi mandir ke paas",
    )


def entry_with_canary() -> DiaryEntry:
    return DiaryEntry(
        walk_id="w_1",
        date=dt.date(2026, 10, 6),
        group="Mom and Dad",
        mission_text="Can you count the gates you pass?",
        entry=f"Mom and Dad saw a {CANARY} near the gate.",
        facts=[],
        mission_result=MissionResult.NOT_MENTIONED,
    )


def text_of(msgs: list[ChatMessage]) -> str:
    return "\n".join(m.content for m in msgs)


# ------------------------------------------------------------------ writer prompt


def test_writer_roles_and_order() -> None:
    msgs = build_writer_prompt(
        WalkerGroup("Mom and Dad"),
        MissionText("Can you count the gates you pass?"),
        Transcript("haan aaj humne paanch gate gine"),
    )
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == builders.writer_system_prompt()


def test_writer_user_message_carries_inputs_verbatim() -> None:
    transcript = 'shayad "koyal" thi — कोयल मंदिर के पास बोल रही थी\nand a breeze'
    msgs = build_writer_prompt(
        WalkerGroup("Mom"), MissionText("Listen for one bird as you walk."), Transcript(transcript)
    )
    user = msgs[1].content
    assert transcript in user
    assert "Listen for one bird as you walk." in user
    assert "Mom" in user
    assert transcript not in msgs[0].content


def test_writer_without_mission() -> None:
    msgs = build_writer_prompt(WalkerGroup("Dad"), None, Transcript("bas ghoom ke aaye"))
    assert "none" in msgs[1].content.lower()


@pytest.mark.parametrize(
    "bad",
    [
        {"group": "Mom and Dad"},
        {"mission": "Can you count the gates?"},
        {"transcript": "plain string"},
        {"transcript": entry_with_canary()},
        {"mission": entry_with_canary()},
        {"group": WalkerGroup("Mom"), "transcript": MissionText("not a transcript")},
    ],
)
def test_writer_rejects_untyped_inputs(bad: dict[str, object]) -> None:
    args: dict[str, object] = {
        "group": WalkerGroup("Mom and Dad"),
        "mission": MissionText("Can you count the gates you pass?"),
        "transcript": Transcript("gine paanch gate"),
    }
    args.update(bad)
    with pytest.raises(TypeError) as exc:
        build_writer_prompt(**args)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert msg.startswith("build_writer_prompt:")
    assert CANARY not in msg
    assert "gate" not in msg and "plain string" not in msg


# ------------------------------------------------------------------ planner prompt


def test_planner_roles_and_structured_facts() -> None:
    msgs = build_planner_prompt(
        [fact("f_1"), fact("f_2", certainty=Certainty.UNSURE)],
        [MissionText("Can you count the gates you pass?")],
        group=WalkerGroup("Mom and Dad"),
        today=TODAY,
    )
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == builders.planner_system_prompt()
    user = msgs[1].content
    records = [json.loads(line) for line in user.splitlines() if line.startswith("{")]
    assert [r["id"] for r in records] == ["f_1", "f_2"]
    assert records[0] == {
        "id": "f_1",
        "walk_date": "Tuesday 2026-10-06",
        "kind": "sound",
        "subject": "koyal",
        "place": "mandir ke paas",
        "certainty": "sure",
        "span": "koyal bol rahi thi mandir ke paas",
    }
    assert records[1]["certainty"] == "unsure"
    assert "Can you count the gates you pass?" in user
    assert "Saturday 2026-10-10" in user
    assert "Mom and Dad" in user


def test_planner_with_no_facts_or_missions() -> None:
    msgs = build_planner_prompt([], [], group=WalkerGroup("Mom"), today=TODAY)
    assert "none" in msgs[1].content.lower()


def test_planner_more_than_five_recent_missions_raises() -> None:
    recent = [MissionText(f"Can you count the gates you pass, take {i}?") for i in range(6)]
    with pytest.raises(ValueError) as exc:
        build_planner_prompt([], recent, group=WalkerGroup("Mom"), today=TODAY)
    assert "gates" not in str(exc.value)


@pytest.mark.parametrize(
    "facts",
    [
        [entry_with_canary()],
        [f"prose about a {CANARY}"],
        [fact(), {"id": "f_9", "subject": CANARY}],
        f"prose about a {CANARY}",  # a str is a Sequence too
    ],
)
def test_planner_rejects_non_verified_facts(facts: object) -> None:
    with pytest.raises(TypeError) as exc:
        build_planner_prompt(facts, [], group=WalkerGroup("Mom"), today=TODAY)  # type: ignore[arg-type]
    assert str(exc.value).startswith("build_planner_prompt:")
    assert CANARY not in str(exc.value)


@pytest.mark.parametrize(
    "recent",
    [
        ["Can you count the gates you pass?"],
        [entry_with_canary()],
        f"prose about a {CANARY}",
    ],
)
def test_planner_rejects_untyped_recent_missions(recent: object) -> None:
    with pytest.raises(TypeError) as exc:
        build_planner_prompt([], recent, group=WalkerGroup("Mom"), today=TODAY)  # type: ignore[arg-type]
    assert CANARY not in str(exc.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group": "Mom"},
        {"today": "2026-10-10"},
        {"today": dt.datetime(2026, 10, 10, 18, 30)},  # dates, never times (NFR-P5)
        {"regen": ["verb_not_allowed"]},
    ],
)
def test_planner_rejects_bad_keywords(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {"group": WalkerGroup("Mom"), "today": TODAY}
    args.update(kwargs)
    with pytest.raises(TypeError):
        build_planner_prompt([], [], **args)  # type: ignore[arg-type]


# --------------------------------------------------------------- canary isolation


def test_canary_entry_text_cannot_reach_any_prompt() -> None:
    entry = entry_with_canary()
    attempts = [
        lambda: build_writer_prompt(WalkerGroup("Mom"), None, entry),  # type: ignore[arg-type]
        lambda: build_writer_prompt(WalkerGroup("Mom"), entry, Transcript("x")),  # type: ignore[arg-type]
        lambda: build_writer_prompt(entry, None, Transcript("x")),  # type: ignore[arg-type]
        lambda: build_writer_prompt(WalkerGroup("Mom"), None, entry.entry),  # type: ignore[arg-type]
        lambda: build_planner_prompt([entry], [], group=WalkerGroup("Mom"), today=TODAY),  # type: ignore[list-item]
        lambda: build_planner_prompt([], [entry.entry], group=WalkerGroup("Mom"), today=TODAY),  # type: ignore[list-item]
    ]
    for attempt in attempts:
        with pytest.raises(TypeError) as exc:
            attempt()
        assert CANARY not in str(exc.value)
        assert CANARY not in repr(exc.value)
    ok = build_planner_prompt([fact()], [], group=WalkerGroup("Mom"), today=TODAY)
    ok += build_writer_prompt(WalkerGroup("Mom"), None, Transcript("koyal bol rahi thi"))
    assert CANARY not in text_of(ok)


@pytest.mark.parametrize("wrapper", [WalkerGroup, MissionText, Transcript])
def test_wrappers_validate_and_hide_content(wrapper: type) -> None:
    with pytest.raises(TypeError) as exc:
        wrapper(entry_with_canary())
    assert CANARY not in str(exc.value)
    with pytest.raises(ValueError):
        wrapper("   ")
    value = wrapper(f"secret {CANARY}")
    assert CANARY not in repr(value)
    assert CANARY not in str(value)


def test_wrapper_length_limits() -> None:
    with pytest.raises(ValueError):
        WalkerGroup("x" * 61)
    with pytest.raises(ValueError):
        MissionText("x" * 401)


def test_chat_message_repr_hides_content() -> None:
    msg = ChatMessage(role="user", content=CANARY)
    assert CANARY not in repr(msg)
    with pytest.raises(TypeError):
        ChatMessage(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ChatMessage(role="user", content=entry_with_canary())  # type: ignore[arg-type]


# --------------------------------------------------------------- regeneration


def test_regen_appends_only_fixed_instructions() -> None:
    codes = [ReasonCode.SPAN_NOT_IN_TRANSCRIPT, ReasonCode.ENTRY_TOO_LONG, ReasonCode.REGENERATED]
    base = build_writer_prompt(WalkerGroup("Mom"), None, Transcript("koyal bol rahi thi"))
    regen = build_writer_prompt(
        WalkerGroup("Mom"), None, Transcript("koyal bol rahi thi"), regen=codes
    )
    assert regen[0] == base[0]
    assert regen[1].content.startswith(base[1].content)
    extra = regen[1].content[len(base[1].content) :]
    for line in reasons_to_instructions(codes):
        assert line in extra
    lines = [ln[2:] for ln in extra.splitlines() if ln.startswith("- ")]
    assert lines == reasons_to_instructions(codes)
    allowed_header = builders.REGEN_HEADER
    leftover = extra.replace(allowed_header, "")
    for line in lines:
        leftover = leftover.replace(f"- {line}", "")
    assert leftover.strip() == ""


def test_planner_regen_instructions() -> None:
    msgs = build_planner_prompt(
        [],
        [],
        group=WalkerGroup("Mom"),
        today=TODAY,
        regen=[ReasonCode.TIME_OF_DAY, ReasonCode.MULTIPLE_TASKS],
    )
    assert "Do not mention any time of day." in msgs[1].content
    assert "Give exactly one task." in msgs[1].content


def test_regen_rejects_non_codes() -> None:
    with pytest.raises(TypeError) as exc:
        build_writer_prompt(
            WalkerGroup("Mom"),
            None,
            Transcript("x"),
            regen=[f"previous output {CANARY}"],  # type: ignore[list-item]
        )
    assert CANARY not in str(exc.value)


# --------------------------------------------------------------- templates


def test_fingerprint_stable_and_distinct() -> None:
    w1, w2 = prompt_fingerprint("writer"), prompt_fingerprint("writer")
    p1 = prompt_fingerprint("planner")
    assert w1 == w2
    assert w1 != p1
    assert len(w1) == 64 and int(w1, 16) >= 0
    assert PROMPT_VERSION
    with pytest.raises(ValueError):
        prompt_fingerprint("other")  # type: ignore[arg-type]


def test_templates_have_required_rules() -> None:
    writer = builders.writer_system_prompt()
    planner = builders.planner_system_prompt()
    for needle in ("JSON", "span", "mission_span", "not_mentioned", "unsure", "70"):
        assert needle in writer
    for needle in ("30 words", "notice", "watch", "references", "you", "time of day", "JSON"):
        assert needle in planner
