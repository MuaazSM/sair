"""Prompt builders for the writer and the mission planner (PRD FR-M3, FR-J3, FR-J5).

Typed so prose cannot be passed:

* the writer gets exactly a :class:`WalkerGroup`, an optional :class:`MissionText` and a
  :class:`Transcript` (FR-J3), never previous entries;
* the planner gets only ``VerifiedFact`` records and at most 5 :class:`MissionText`
  values (FR-M3), never entry prose.

The wrappers are frozen dataclasses that validate at runtime, and every builder checks
the type of every argument at runtime too, so a bare ``str`` or an entry object (e.g. a
``DiaryEntry``) raises ``TypeError``. Error messages and ``repr`` never contain content
(NFR-P4).

Regeneration (FR-J5, FR-M6) appends only the fixed instructions from
``hub.schemas.reasons_to_instructions``; previous model output is never passed back.

The system prompts are files next to this module (``writer_system.md``,
``planner_system.md``) and are also used for training data and the fine-tune, so they
must stay stable: bump :data:`PROMPT_VERSION` and record :func:`prompt_fingerprint`
whenever either changes.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from hub.schemas import MAX_GROUP_CHARS, MAX_MISSION_CHARS, ReasonCode, VerifiedFact
from hub.schemas import reasons_to_instructions as _reasons_to_instructions

__all__ = [
    "MAX_RECENT_MISSIONS",
    "PROMPT_VERSION",
    "REGEN_HEADER",
    "ChatMessage",
    "MissionText",
    "Role",
    "Transcript",
    "WalkerGroup",
    "build_planner_prompt",
    "build_writer_prompt",
    "planner_system_prompt",
    "prompt_fingerprint",
    "writer_system_prompt",
]

PROMPT_VERSION = "1"
MAX_RECENT_MISSIONS = 5  # FR-M3
REGEN_HEADER = "Your last reply had problems. Fix them:"

_DIR = Path(__file__).resolve().parent
_TEMPLATES = {"writer": "writer_system.md", "planner": "planner_system.md"}

Role = Literal["system", "user", "assistant"]
_ROLES = frozenset(get_args(Role))


# ------------------------------------------------------------------ typed inputs


def _check_text(owner: str, value: object, max_chars: int | None) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: text must be str")
    if not value.strip():
        raise ValueError(f"{owner}: text must not be empty")
    if max_chars is not None and len(value) > max_chars:
        raise ValueError(f"{owner}: text longer than {max_chars} characters")


@dataclass(frozen=True, slots=True)
class WalkerGroup:
    """The walker group name from config (PRD §3), e.g. "Mom and Dad"."""

    name: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_text("WalkerGroup", self.name, MAX_GROUP_CHARS)

    def __repr__(self) -> str:
        return "WalkerGroup(<redacted>)"


@dataclass(frozen=True, slots=True)
class MissionText:
    """The text of a served mission (today's, or one of the last 5)."""

    text: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_text("MissionText", self.text, MAX_MISSION_CHARS)

    def __repr__(self) -> str:
        return "MissionText(<redacted>)"


@dataclass(frozen=True, slots=True)
class Transcript:
    """The verbatim transcript of one voice note (FR-J2)."""

    text: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_text("Transcript", self.text, None)

    def __repr__(self) -> str:
        return "Transcript(<redacted>)"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise TypeError("ChatMessage: role must be system, user or assistant")
        if not isinstance(self.content, str):
            raise TypeError("ChatMessage: content must be str")

    def __repr__(self) -> str:
        return f"ChatMessage(role={self.role!r}, content=<redacted>)"


# ------------------------------------------------------------------ templates


@functools.cache
def _template(which: str) -> str:
    raw = (_DIR / _TEMPLATES[which]).read_text(encoding="utf-8")
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def writer_system_prompt() -> str:
    """The rendered writer system prompt (also the training prompt)."""
    return _template("writer")


def planner_system_prompt() -> str:
    """The rendered planner system prompt."""
    return _template("planner")


def prompt_fingerprint(which: Literal["writer", "planner"] = "writer") -> str:
    """sha256 hex of the rendered system prompt; record it with training runs and evals."""
    if which not in _TEMPLATES:
        raise ValueError("prompt_fingerprint: which must be writer or planner")
    return hashlib.sha256(_template(which).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ helpers


def _require(owner: str, name: str, value: object, kind: type) -> None:
    if not isinstance(value, kind):
        raise TypeError(f"{owner}: {name} must be {kind.__name__}")


def _require_seq(owner: str, name: str, value: object, kind: type) -> list[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{owner}: {name} must be a sequence of {kind.__name__}")
    items = list(value)
    for item in items:
        if not isinstance(item, kind):
            raise TypeError(f"{owner}: {name} must contain only {kind.__name__}")
    return items


def _regen_block(owner: str, regen: object) -> str:
    codes = _require_seq(owner, "regen", regen, ReasonCode)
    lines = _reasons_to_instructions([c for c in codes if isinstance(c, ReasonCode)])
    if not lines:
        return ""
    return "\n\n" + REGEN_HEADER + "\n" + "\n".join(f"- {line}" for line in lines)


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _day(d: dt.date) -> str:
    """Weekday + ISO date, locale-independent ("Tuesday 2026-10-06")."""
    return f"{_WEEKDAYS[d.weekday()]} {d.isoformat()}"


def _fact_record(f: VerifiedFact) -> str:
    record = {
        "id": f.id,
        "walk_date": _day(f.walk_date),
        "kind": f.kind.value,
        "subject": f.subject,
        "place": f.place,
        "certainty": f.certainty.value,
        "span": f.span,
    }
    return json.dumps(record, ensure_ascii=False)


# ------------------------------------------------------------------ builders


def build_writer_prompt(
    group: WalkerGroup,
    mission: MissionText | None,
    transcript: Transcript,
    *,
    regen: Sequence[ReasonCode] = (),
) -> list[ChatMessage]:
    """Writer prompt from exactly the FR-J3 inputs: group, today's mission, transcript."""
    owner = "build_writer_prompt"
    _require(owner, "group", group, WalkerGroup)
    if mission is not None:
        _require(owner, "mission", mission, MissionText)
    _require(owner, "transcript", transcript, Transcript)
    user = (
        f"Walkers: {group.name}\n"
        f"Today's mission: {mission.text if mission is not None else '(none)'}\n"
        "Transcript:\n"
        "<<<\n"
        f"{transcript.text}\n"
        ">>>"
    ) + _regen_block(owner, regen)
    return [ChatMessage("system", writer_system_prompt()), ChatMessage("user", user)]


def build_planner_prompt(
    facts: Sequence[VerifiedFact],
    recent_missions: Sequence[MissionText],
    *,
    group: WalkerGroup,
    today: dt.date,
    regen: Sequence[ReasonCode] = (),
) -> list[ChatMessage]:
    """Planner prompt from exactly the FR-M3 inputs: verified facts and the last 5 missions.

    Facts are rendered as JSON records, never prose. ``today`` must be a date (a
    ``datetime`` is refused: the planner never sees a time, FR-M6b/NFR-P5).
    """
    owner = "build_planner_prompt"
    fact_items = _require_seq(owner, "facts", facts, VerifiedFact)
    recent_items = _require_seq(owner, "recent_missions", recent_missions, MissionText)
    if len(recent_items) > MAX_RECENT_MISSIONS:
        raise ValueError(f"{owner}: at most {MAX_RECENT_MISSIONS} recent missions")
    _require(owner, "group", group, WalkerGroup)
    if isinstance(today, dt.datetime) or not isinstance(today, dt.date):
        raise TypeError(f"{owner}: today must be a date")

    fact_lines = [_fact_record(f) for f in fact_items if isinstance(f, VerifiedFact)]
    recent_lines = [f"- {m.text}" for m in recent_items if isinstance(m, MissionText)]
    user = (
        f"Walkers: {group.name}\n"
        f"Today: {_day(today)}\n"
        "Verified facts (one JSON record per line):\n"
        f"{chr(10).join(fact_lines) if fact_lines else '(none)'}\n"
        "Recent missions (do not repeat):\n"
        f"{chr(10).join(recent_lines) if recent_lines else '(none)'}"
    ) + _regen_block(owner, regen)
    return [ChatMessage("system", planner_system_prompt()), ChatMessage("user", user)]
