"""Typed records shared by the hub: writer/planner outputs, memory, API bodies, reason codes.

Split of responsibilities (PRD §8, §9):

* This module does **structural** validation only: types, enums, the 0-6 fact cap
  (§9.1) and generous ``max_length`` guards that stop runaway generations.
* **Semantic** rules live in ``hub/grounding.py`` and ``hub/safety.py``: span exactness,
  subject/place inside span, hedges, the detail lexicon, word and sentence counts, and
  the ``mission_span`` requirement. Those checks must *drop facts* or *coerce*
  ``mission_result`` to ``not_mentioned`` rather than raise, so they are deliberately
  not Pydantic validators.

Privacy (NFR-P4): every model sets ``hide_input_in_errors=True`` so a ``ValidationError``
never embeds transcript, entry, mission or fact text. Callers that parse model output
use :func:`parse_writer_output` / :func:`parse_mission_plan`, which return reason codes
only and never the error text. API models carry dates, never times (NFR-P5).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

__all__ = [
    "BlockCategory",
    "CanaryResult",
    "Certainty",
    "DiaryEntry",
    "DiaryFact",
    "DiaryResponse",
    "DiarySummary",
    "EntryOrigin",
    "Fact",
    "FactChip",
    "FactKind",
    "FeedbackRequest",
    "GROUNDING_CODES",
    "GroupCounts",
    "Healthz",
    "LatencyStat",
    "MISSION_SAFETY_CODES",
    "Mission",
    "MissionPlan",
    "MissionResult",
    "MissionSource",
    "MissionTodayResponse",
    "MissionType",
    "PIPELINE_CODES",
    "ParseResult",
    "ReasonCode",
    "StatusResponse",
    "VerifiedFact",
    "WalkAccepted",
    "WalkStatus",
    "WalkStatusResponse",
    "WriterBackend",
    "WriterOutput",
    "parse_mission_plan",
    "parse_writer_output",
    "reasons_to_instructions",
]

# Runaway-generation guards (structural only; §8 word limits are enforced in grounding).
MAX_FACTS = 6
MAX_SPAN_CHARS = 400
MAX_SUBJECT_CHARS = 80
MAX_PLACE_CHARS = 120
MAX_ENTRY_CHARS = 1200
MAX_MISSION_CHARS = 400
MAX_REFERENCES = 6
MAX_ID_CHARS = 64
MAX_GROUP_CHARS = 60
MAX_NOTE_CHARS = 500

_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class _Model(BaseModel):
    """Base for every Sair record: no extra keys, immutable, inputs hidden in errors."""

    model_config = _CONFIG

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        # Pydantic's own extra_forbidden error puts the unknown *key* in ``loc``, and a
        # model can emit content as a key name. Reject first, with no key in the error.
        if isinstance(data, dict) and not set(data) <= cls.model_fields.keys():
            raise PydanticCustomError("extra_forbidden", "Extra inputs are not permitted")
        return data


Id = Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]
GroupName = Annotated[str, Field(min_length=1, max_length=MAX_GROUP_CHARS)]


# --------------------------------------------------------------------------- enums


class FactKind(StrEnum):
    SIGHT = "sight"
    SOUND = "sound"
    SMELL = "smell"
    PERSON = "person"
    ANIMAL = "animal"
    PLANT = "plant"
    WEATHER = "weather"
    OTHER = "other"


class Certainty(StrEnum):
    SURE = "sure"
    UNSURE = "unsure"


class MissionResult(StrEnum):
    DONE = "done"
    PARTLY = "partly"
    NOT_DONE = "not_done"
    NOT_MENTIONED = "not_mentioned"


class MissionType(StrEnum):
    RETURN = "return"
    CHANGE = "change"
    SENSE = "sense"
    COUNT = "count"
    SEED = "seed"


class MissionSource(StrEnum):
    MODEL = "model"
    SEED = "seed"
    FALLBACK = "fallback"


class WalkStatus(StrEnum):
    """DB ``walks.status`` (§9.4); also the API status (no separate ``writing`` value)."""

    RECEIVED = "received"  # acknowledged; graph running. Client shows "Writing it down..."
    DONE = "done"  # entry stored (model or template)
    LOST = "lost"  # permanent audio failure or retry window expired; see ``reason``


class WriterBackend(StrEnum):
    GEMMA = "gemma"
    LLAMA = "llama"


class EntryOrigin(StrEnum):
    """Who built the entry text. Template entries are exempt from §8 check 6 only."""

    MODEL = "model"
    TEMPLATE = "template"


class CanaryResult(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class BlockCategory(StrEnum):
    """FR-M6(b) blocklist categories; ids match ``data/walk_words.yaml``."""

    ROAD_CROSSING = "road_crossing"
    TOUCH_ANIMAL = "touch_animal"
    FEED_ANIMAL = "feed_animal"
    GIVE_ANYTHING = "give_anything"
    FOLLOW_ANIMAL = "follow_animal"
    APPROACH_STRANGER = "approach_stranger"
    ENTER_PROPERTY = "enter_property"
    CLIMB = "climb"
    RUN = "run"
    PHONE_OUTSIDE = "phone_outside"


class ReasonCode(StrEnum):
    """Enum codes only, never text (NFR-P4). Safe for logs, metrics and ``fail_reasons``."""

    # --- grounding, PRD §8 (writer output)
    INVALID_JSON = "invalid_json"  # output was not parseable JSON
    SCHEMA_INVALID = "schema_invalid"  # JSON did not match §9.1 / §9.2 structure
    TOO_MANY_FACTS = "too_many_facts"  # more than 6 facts (§9.1)
    SPAN_NOT_IN_TRANSCRIPT = "span_not_in_transcript"  # check 1: fact dropped
    SUBJECT_NOT_IN_SPAN = "subject_not_in_span"  # check 2: fact dropped
    PLACE_NOT_IN_SPAN = "place_not_in_span"  # check 2: fact dropped
    HEDGE_CERTAINTY_MISMATCH = "hedge_certainty_mismatch"  # check 3: hedged span marked sure
    HEDGE_MISSING_IN_ENTRY = "hedge_missing_in_entry"  # check 3: entry states unsure as sure
    UNSUPPORTED_DETAIL = "unsupported_detail"  # check 4: lexicon/proper-noun orphan in entry
    NEW_NUMBER = "new_number"  # check 5: number absent from transcript
    NEW_TIME = "new_time"  # check 5: time absent from transcript
    NEW_PERSON_NAME = "new_person_name"  # check 5: person name absent from transcript
    ENTRY_TOO_SHORT = "entry_too_short"  # check 6: < 5 words
    ENTRY_TOO_LONG = "entry_too_long"  # check 6: > 70 words
    TOO_FEW_SENTENCES = "too_few_sentences"  # check 6: < 1 sentence
    TOO_MANY_SENTENCES = "too_many_sentences"  # check 6: > 4 sentences
    GROUP_NOT_MENTIONED = "group_not_mentioned"  # check 6: walker group absent
    URL_IN_ENTRY = "url_in_entry"  # check 6
    QUESTION_IN_ENTRY = "question_in_entry"  # check 6
    MISSION_SPAN_MISSING = "mission_span_missing"  # check 7: coerced to not_mentioned
    MISSION_SPAN_NOT_IN_TRANSCRIPT = "mission_span_not_in_transcript"  # check 7: coerced

    # --- mission safety, FR-M1/M4/M5/M6/M6b (planner output)
    VERB_NOT_ALLOWED = "verb_not_allowed"  # FR-M6(a): main verb outside allowlist
    BLOCKLISTED = "blocklisted"  # FR-M6(b): see BlockCategory for which rule
    NOUN_NOT_PROVENANCED = "noun_not_provenanced"  # FR-M4: noun not in facts or walk words
    UNSURE_NOT_QUESTION = "unsure_not_question"  # FR-M5: unsure fact not posed as question
    NOT_ADDRESSED_TO_GROUP = "not_addressed_to_group"  # mission does not say "you"
    TIME_OF_DAY = "time_of_day"  # FR-M6b: mentions a time of day
    MISSION_TOO_LONG = "mission_too_long"  # FR-M1/M6(d): > 30 words
    MULTIPLE_TASKS = "multiple_tasks"  # FR-M1/M6(d): more than one task sentence
    REFERENCE_UNKNOWN = "reference_unknown"  # references an id not among supplied facts
    RETURN_WITHOUT_REFERENCE = "return_without_reference"  # type=return with no references
    MISSION_REPEATED = "mission_repeated"  # same as one of the last 5 missions (FR-M3)

    # --- pipeline outcomes (journal + mission graphs)
    AUDIO_TOO_SHORT = "audio_too_short"  # FR-J1: < 1 s, permanent
    AUDIO_TOO_LONG = "audio_too_long"  # FR-J1: > 90 s, permanent
    AUDIO_SILENT = "audio_silent"  # FR-J2: VAD/energy gate, permanent ("Couldn't hear")
    AUDIO_UNDECODABLE = "audio_undecodable"  # ffmpeg could not decode, permanent
    TRANSCRIBE_TRANSIENT = "transcribe_transient"  # model unreachable/timeout, retried
    TRANSCRIBE_PERMANENT = "transcribe_permanent"  # non-retryable transcription error
    TRANSCRIPT_EMPTY = "transcript_empty"  # transcription returned no text
    WRITER_TRANSIENT = "writer_transient"  # writer unreachable/timeout
    WRITER_PERMANENT = "writer_permanent"  # non-retryable writer error
    REGENERATED = "regenerated"  # FR-J5/FR-M6: one regeneration was used
    TEMPLATE_FALLBACK = "template_fallback"  # FR-J5: template entry stored
    RETRY_WINDOW_EXPIRED = "retry_window_expired"  # FR-J6: 10 min elapsed, audio deleted
    STALE_AUDIO_SWEPT = "stale_audio_swept"  # FR-J6: startup sweep deleted temp audio
    PLANNER_TRANSIENT = "planner_transient"  # planner model unreachable/timeout
    PLANNER_PERMANENT = "planner_permanent"  # non-retryable planner error
    MISSION_FALLBACK = "mission_fallback"  # FR-M6/M7: fixed fallback mission served


GROUNDING_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.INVALID_JSON,
        ReasonCode.SCHEMA_INVALID,
        ReasonCode.TOO_MANY_FACTS,
        ReasonCode.SPAN_NOT_IN_TRANSCRIPT,
        ReasonCode.SUBJECT_NOT_IN_SPAN,
        ReasonCode.PLACE_NOT_IN_SPAN,
        ReasonCode.HEDGE_CERTAINTY_MISMATCH,
        ReasonCode.HEDGE_MISSING_IN_ENTRY,
        ReasonCode.UNSUPPORTED_DETAIL,
        ReasonCode.NEW_NUMBER,
        ReasonCode.NEW_TIME,
        ReasonCode.NEW_PERSON_NAME,
        ReasonCode.ENTRY_TOO_SHORT,
        ReasonCode.ENTRY_TOO_LONG,
        ReasonCode.TOO_FEW_SENTENCES,
        ReasonCode.TOO_MANY_SENTENCES,
        ReasonCode.GROUP_NOT_MENTIONED,
        ReasonCode.URL_IN_ENTRY,
        ReasonCode.QUESTION_IN_ENTRY,
        ReasonCode.MISSION_SPAN_MISSING,
        ReasonCode.MISSION_SPAN_NOT_IN_TRANSCRIPT,
    }
)
MISSION_SAFETY_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.VERB_NOT_ALLOWED,
        ReasonCode.BLOCKLISTED,
        ReasonCode.NOUN_NOT_PROVENANCED,
        ReasonCode.UNSURE_NOT_QUESTION,
        ReasonCode.NOT_ADDRESSED_TO_GROUP,
        ReasonCode.TIME_OF_DAY,
        ReasonCode.MISSION_TOO_LONG,
        ReasonCode.MULTIPLE_TASKS,
        ReasonCode.REFERENCE_UNKNOWN,
        ReasonCode.RETURN_WITHOUT_REFERENCE,
        ReasonCode.MISSION_REPEATED,
    }
)
PIPELINE_CODES: frozenset[ReasonCode] = frozenset(
    set(ReasonCode) - GROUNDING_CODES - MISSION_SAFETY_CODES
)

# Fixed regeneration instructions (FR-J5, FR-M6). Written in code; never contain content.
_INSTRUCTIONS: dict[ReasonCode, str] = {
    ReasonCode.INVALID_JSON: "Reply with one valid JSON object only.",
    ReasonCode.SCHEMA_INVALID: "Use exactly the required JSON fields and allowed values.",
    ReasonCode.TOO_MANY_FACTS: "List at most 6 facts.",
    ReasonCode.SPAN_NOT_IN_TRANSCRIPT: "Copy each span exactly, word for word, from the note.",
    ReasonCode.SUBJECT_NOT_IN_SPAN: "Take each subject from the words of its own span.",
    ReasonCode.PLACE_NOT_IN_SPAN: "Take each place from the words of its own span, or omit it.",
    ReasonCode.HEDGE_CERTAINTY_MISMATCH: "If a span says maybe or I think, mark it unsure.",
    ReasonCode.HEDGE_MISSING_IN_ENTRY: "Write unsure things with 'maybe' in the entry.",
    ReasonCode.UNSUPPORTED_DETAIL: "Mention only things that are in the facts.",
    ReasonCode.NEW_NUMBER: "Do not add numbers that are not in the note.",
    ReasonCode.NEW_TIME: "Do not add times that are not in the note.",
    ReasonCode.NEW_PERSON_NAME: "Do not add names that are not in the note.",
    ReasonCode.ENTRY_TOO_SHORT: "Write at least 5 words.",
    ReasonCode.ENTRY_TOO_LONG: "Keep the entry under 70 words.",
    ReasonCode.TOO_FEW_SENTENCES: "Write at least one full sentence.",
    ReasonCode.TOO_MANY_SENTENCES: "Use at most 4 sentences.",
    ReasonCode.GROUP_NOT_MENTIONED: "Name the walkers in the entry.",
    ReasonCode.URL_IN_ENTRY: "Do not include links.",
    ReasonCode.QUESTION_IN_ENTRY: "Do not ask questions in the entry.",
    ReasonCode.MISSION_SPAN_MISSING: "If the mission was done, copy the words that say so.",
    ReasonCode.MISSION_SPAN_NOT_IN_TRANSCRIPT: "Copy the mission span exactly from the note.",
    ReasonCode.VERB_NOT_ALLOWED: "Start the task with notice, look, listen, count or find.",
    ReasonCode.BLOCKLISTED: "Ask only for looking, listening or counting from the path.",
    ReasonCode.NOUN_NOT_PROVENANCED: "Mention only things from the facts or common walk words.",
    ReasonCode.UNSURE_NOT_QUESTION: "Ask about unsure things as a question to 'you'.",
    ReasonCode.NOT_ADDRESSED_TO_GROUP: "Speak to the walkers as 'you'.",
    ReasonCode.TIME_OF_DAY: "Do not mention any time of day.",
    ReasonCode.MISSION_TOO_LONG: "Keep the mission under 30 words.",
    ReasonCode.MULTIPLE_TASKS: "Give exactly one task.",
    ReasonCode.REFERENCE_UNKNOWN: "Reference only the fact ids you were given.",
    ReasonCode.RETURN_WITHOUT_REFERENCE: "A return mission must reference one fact id.",
    ReasonCode.MISSION_REPEATED: "Do not repeat a recent mission.",
}


def reasons_to_instructions(codes: Sequence[ReasonCode]) -> list[str]:
    """Map codes to short fixed instructions for the one regeneration, deduplicated in order.

    Pipeline codes have no instruction and are skipped.
    """
    out: list[str] = []
    for code in codes:
        text = _INSTRUCTIONS.get(code)
        if text is not None and text not in out:
            out.append(text)
    return out


# ------------------------------------------------------------- writer output (§9.1)


class Fact(_Model):
    span: str = Field(min_length=1, max_length=MAX_SPAN_CHARS)
    kind: FactKind
    subject: str = Field(min_length=1, max_length=MAX_SUBJECT_CHARS)
    certainty: Certainty
    place: str | None = Field(default=None, max_length=MAX_PLACE_CHARS)


class WriterOutput(_Model):
    """Raw writer output. Structural only; grounding drops facts and coerces mission_result."""

    facts: list[Fact] = Field(max_length=MAX_FACTS)
    mission_result: MissionResult
    mission_span: str | None = Field(default=None, max_length=MAX_SPAN_CHARS)
    entry: str = Field(max_length=MAX_ENTRY_CHARS)


# ------------------------------------------------------------------ memory (FR-M3)


class VerifiedFact(_Model):
    """A fact that passed §8 checks 1-3; the only memory the planner ever sees."""

    id: Id
    walk_id: Id
    walk_date: dt.date
    group: GroupName
    kind: FactKind
    subject: str = Field(min_length=1, max_length=MAX_SUBJECT_CHARS)
    place: str | None = Field(default=None, max_length=MAX_PLACE_CHARS)
    certainty: Certainty
    span: str = Field(min_length=1, max_length=MAX_SPAN_CHARS)


# ------------------------------------------------------------ missions (§9.2, §9.4)


class MissionPlan(_Model):
    """Raw planner output (§9.2). Safety rules live in hub/safety.py."""

    type: MissionType
    references: list[Id] = Field(default_factory=list, max_length=MAX_REFERENCES)
    text: str = Field(min_length=1, max_length=MAX_MISSION_CHARS)


class Mission(_Model):
    """A stored/served mission (§9.4 ``missions``)."""

    id: Id
    date: dt.date
    group: GroupName
    type: MissionType
    text: str = Field(min_length=1, max_length=MAX_MISSION_CHARS)
    references: list[Id] = Field(default_factory=list, max_length=MAX_REFERENCES)
    source: MissionSource


# ------------------------------------------------------------------- API (§9.3)


class MissionTodayResponse(_Model):
    mission_id: Id
    text: str = Field(max_length=MAX_MISSION_CHARS)
    type: MissionType
    audio_path: str | None = None  # None while TTS has not rendered yet


class WalkAccepted(_Model):
    walk_id: Id
    status: WalkStatus = WalkStatus.RECEIVED


class FactChip(_Model):
    """A verified fact as shown to the family (solid = sure, dotted = unsure)."""

    subject: str = Field(max_length=MAX_SUBJECT_CHARS)
    kind: FactKind
    certainty: Certainty
    place: str | None = Field(default=None, max_length=MAX_PLACE_CHARS)


class WalkStatusResponse(_Model):
    status: WalkStatus
    date: dt.date
    entry: str | None = Field(default=None, max_length=MAX_ENTRY_CHARS)
    facts: list[FactChip] | None = None
    mission_result: MissionResult | None = None
    reason: ReasonCode | None = None  # set only when status=lost (e.g. audio_silent)


class DiaryFact(FactChip):
    asked_again_on: list[dt.date] = Field(default_factory=list)  # FR-D2


class DiaryEntry(_Model):
    walk_id: Id
    date: dt.date
    group: GroupName
    mission_text: str | None = Field(default=None, max_length=MAX_MISSION_CHARS)
    entry: str = Field(max_length=MAX_ENTRY_CHARS)
    facts: list[DiaryFact] = Field(default_factory=list)
    mission_result: MissionResult


class DiarySummary(_Model):
    walks_this_month: int = Field(ge=0)
    things_noticed: int = Field(ge=0)


class DiaryResponse(_Model):
    summary: DiarySummary
    entries: list[DiaryEntry]


class FeedbackRequest(_Model):
    correct: bool
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class LatencyStat(_Model):
    n: int = Field(ge=0)
    p50_ms: float | None = None
    p95_ms: float | None = None
    target_p50_ms: float | None = None
    target_p95_ms: float | None = None


class GroupCounts(_Model):
    walks: int = Field(ge=0)
    facts: int = Field(ge=0)


class StatusResponse(_Model):
    """FR-O2: metadata only. Dict keys are model/node/group names; no free-text values."""

    model_health: dict[str, bool]
    latency: dict[str, LatencyStat]
    per_group: dict[str, GroupCounts]
    grounding_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    grounding_n: int = Field(default=0, ge=0)
    reason_counts: dict[ReasonCode, int] = Field(default_factory=dict)
    fallback_count: int = Field(ge=0)
    feedback_pending: int = Field(default=0, ge=0)
    raw_audio_on_disk: int = Field(ge=0)
    last_canary_result: CanaryResult | None = None
    last_canary_date: dt.date | None = None


class Healthz(_Model):
    ok: bool = True


# ------------------------------------------------------------------- parsing


@dataclass(frozen=True, slots=True)
class ParseResult[T: BaseModel]:
    """Parsed value or reason codes. Never carries error text or input."""

    value: T | None
    reasons: tuple[ReasonCode, ...] = ()

    @property
    def ok(self) -> bool:
        return self.value is not None


def _parse[T: BaseModel](model: type[T], raw: str, list_field: str | None) -> ParseResult[T]:
    try:
        return ParseResult(model.model_validate_json(raw))
    except ValidationError as exc:
        reasons: list[ReasonCode] = []
        for err in exc.errors(include_input=False, include_url=False, include_context=False):
            etype = err["type"]
            if etype in ("json_invalid", "json_type"):
                code = ReasonCode.INVALID_JSON
            elif etype == "too_long" and list_field is not None and err["loc"] == (list_field,):
                code = ReasonCode.TOO_MANY_FACTS
            else:
                code = ReasonCode.SCHEMA_INVALID
            if code not in reasons:
                reasons.append(code)
        return ParseResult(None, tuple(reasons) or (ReasonCode.SCHEMA_INVALID,))
    except (ValueError, TypeError):
        return ParseResult(None, (ReasonCode.INVALID_JSON,))


def parse_writer_output(raw: str) -> ParseResult[WriterOutput]:
    """Parse raw writer JSON. Returns codes only on failure (NFR-P4)."""
    return _parse(WriterOutput, raw, "facts")


def parse_mission_plan(raw: str) -> ParseResult[MissionPlan]:
    """Parse raw planner JSON. Returns codes only on failure (NFR-P4)."""
    return _parse(MissionPlan, raw, None)
