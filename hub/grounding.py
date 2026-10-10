"""Grounding check for writer output (PRD §8, normative) and the template fallback (FR-J5).

Pure and deterministic: no I/O at call time except the cached lexicon load. This module
never logs or prints, and every exception it raises carries a fixed message with no
transcript, entry, fact or file content (NFR-P4).

A fact is *verified* iff it passes checks 1-3; only verified facts enter memory (FR-M3).
``passed`` is False iff any code in :data:`FAILING_CODES` was raised (checks 3-entry, 4, 5, 6).
Fact drops (checks 1-3) and check-7 coercions are recorded in ``reasons`` but do not fail.

Interpretations (where PRD §8 is silent the stricter reading is chosen):

I1  Normalisation (§8.1) is exactly NFC + casefold + whitespace collapse + strip. Curly
    apostrophes and hyphens are *not* folded for span checks; they are folded only for
    lexicon/hedge token matching (``sea-face`` == ``sea face``; ``don’t`` == ``don't``).
I2  Span and mission-span containment is substring containment that must also sit on
    Latin token boundaries ("cat" is not found in "category"). Devanagari edges need no
    boundary. A span that normalises to "" is treated as absent / missing.
I3  Check 2: subject/place passes if it is a boundaried substring of its span, or if it is
    fully *covered*: every lexicon alias in it has its lexicon id present in the span (any
    script), and every remaining word is either an article (the/a/an) or itself present in
    the span. So "big koel" fails against "koyal bol rahi thi" ("big" was invented), and
    "near the temple" fails against "mandir ke paas" ("near" is not in the span; the writer
    must use "mandir" or "temple"). Blank place == no place.
I4  Lexicon matching: Latin aliases match on whole tokens (longest alias first); Devanagari
    aliases match as substrings (``\\b`` is unreliable with matras). On the fact/transcript
    side matching is *inclusive* (all overlapping aliases, ambiguous words included), which
    only makes "present in the evidence" easier. On the entry side matching is *strict*
    (longest non-overlapping, with the ambiguity rules in I8).
I5  Check 3: hedge cues are all cues in data/hedges.yaml, *including* those marked
    ``# review`` (bare "kuch", "jaisa", "jaise", "lagta", "kind of", ...). Over-triggering
    only forces certainty=unsure, which is the conservative direction.
I6  Check 3 entry rule applies to every fact that passed checks 1-2 and whose span has a cue
    OR whose certainty is unsure (even if it was then dropped for HEDGE_CERTAINTY_MISMATCH).
    Every entry sentence that mentions the subject (boundaried phrase, or any lexicon id of
    the subject, cross-script) must contain an entry hedge word from hedges.yaml. A subject
    the entry never mentions needs no hedge. A hedge in a different sentence does not count.
I7  Check 4 details = lexicon ids (all categories in the frozen lexicon, generic setting words
    such as park/gate/sky/bench included) + numbers + time expressions + capitalised proper
    nouns. A detail maps iff it appears in a *verified* fact's span, subject or place
    (alias-aware). Appearing in the transcript is not enough (PRD: "map to a verified fact").
I8  Ambiguous English collisions (entry side only). AMBIGUOUS_NUMBER_ALIASES (one, do, few,
    first, half, many, both, pair, teen, ...) count as a number only when the next token
    (skipping of/the/a/an) starts a lexicon alias ("one crow", "a pair of crows").
    AMBIGUOUS_WORD_ALIASES (well, rose, os, ...) count only after a determiner or a lexicon
    word ("the well", "a red rose", not "as well", "the moon rose").
I9  Numbers: ASCII and Devanagari digits and lexicon number words share one canonical key
    (the lexicon id, e.g. "two"; "num:<n>" outside the lexicon). So "2 parrots" is
    supported by "do tote". Digits inside a time also count as numbers. Number words outside
    the lexicon are not recognised.
I10 Times: "6 pm", "6:30", "6 a.m.", "<number> o'clock|baje|बजे" (key time:H[:MM]; am/pm is
    ignored), bare o'clock/baje (time:clock), noon/midnight, day names (day:<name>) and
    capitalised month names (month:<name>). A day or month name absent from the transcript is
    NEW_TIME. Devanagari day names are not parsed, so "Tuesday" against "मंगलवार" fails.
I11 Person names: a capitalised Latin token that is not sentence-initial, not a walker-group
    word, not a counted lexicon alias, not a day/month, and not in SAFE_CAPITALISED
    (I, Mom, Dad) or NAME_STOPWORDS (The, They, ...). Every such proper noun is treated as a
    person name for NEW_PERSON_NAME (places cannot be told apart). Presence is a boundaried
    case-insensitive match in the transcript; a possessive "'s" is stripped.
    Sentence-initial tokens (check_entry only): capitalisation carries no signal there, so a
    sentence-initial token is a name candidate only if it is none of: a walker-group word,
    SAFE_CAPITALISED, NAME_STOPWORDS, SENTENCE_STARTERS (a closed list of common English
    sentence openers: After, Both, Today, Near, Later, Some, ...), any lexicon alias (counted
    or not), a hedge cue or entry hedge word, a walk_words.yaml generic noun or allowlisted
    verb (any simple inflection), a day/month, a number, or an "-ing" word of 5+ letters
    ("Walking", "Heading"). A candidate that appears in the transcript (boundaried,
    case-insensitive) is dropped, because a common word the transcript uses is far likelier
    than a name; one that does not is NEW_PERSON_NAME + UNSUPPORTED_DETAIL ("Ramesh walked
    with Mom and Dad."). ``extract_details`` does not report sentence-initial candidates, so
    its keys (used by eval) are unchanged.
I12 Walker-group words (each token of ``group`` except "and") are never details or names,
    anywhere in the entry.
I13 Check 6 sentences: split after . ! ? … । ॥ when followed by whitespace, a closing quote or
    the end; a "." after a known abbreviation (Mr, Dr, St, p.m, e.g, ...) or a single-letter
    initial (not "I") does not end a sentence. Words = whitespace tokens containing a letter
    or digit. Group mention = the exact group string, case-insensitive, boundaried
    ("Mom and Dad" requires "mom and dad"). URL = http(s)://, www., or a dotted host with a
    common TLD. Question = any "?" (also full-width). English (§9.1 "entry: English"): the
    entry fails ENTRY_NOT_ENGLISH when more than 20% of its letters (``str.isalpha``) are
    Devanagari; quoted Devanagari words inside an English entry stay well below that.
I14 Check 7: coercion to not_mentioned is recorded (MISSION_SPAN_MISSING /
    MISSION_SPAN_NOT_IN_TRANSCRIPT) but does not fail the entry. A span given with
    not_mentioned is discarded silently.
I15 Templates: sure subjects joined "a", "a and b", "a, b and c"; each unsure subject is
    appended as ", maybe X" (or "maybe X" first if none is sure). Subjects are deduplicated
    by normalised text; a subject unsure in any fact is listed as unsure. The no-fact template
    has no mission clause (PRD verbatim). Templates are exempt from check 6 only (incl. the
    English check: a template may embed a Devanagari subject verbatim).
I16 ``repr``/``str`` of a CheckResult show only passed, reason codes, counts and the mission
    result: never fact text, the mission span or unsupported_details (NFR-P4 hardening).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from hub.schemas import (
    Certainty,
    EntryOrigin,
    Fact,
    MissionResult,
    ReasonCode,
    WriterOutput,
)

__all__ = [
    "AMBIGUOUS_NUMBER_ALIASES",
    "AMBIGUOUS_WORD_ALIASES",
    "CheckResult",
    "DEFAULT_HEDGES",
    "DEFAULT_LEXICON",
    "FAILING_CODES",
    "GroundingError",
    "SENTENCE_STARTERS",
    "Lexicon",
    "build_template_entry",
    "check_entry",
    "extract_details",
    "lexicon_hash",
    "load_lexicon",
    "normalise",
    "template_output",
]

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON = _ROOT / "data" / "lexicon.yaml"
DEFAULT_HEDGES = _ROOT / "data" / "hedges.yaml"
DEFAULT_WALK_WORDS = _ROOT / "data" / "walk_words.yaml"

R = ReasonCode

FAILING_CODES: frozenset[ReasonCode] = frozenset(
    {
        R.HEDGE_MISSING_IN_ENTRY,  # check 3 (entry side)
        R.UNSUPPORTED_DETAIL,  # check 4
        R.NEW_NUMBER,  # check 5
        R.NEW_TIME,
        R.NEW_PERSON_NAME,
        R.ENTRY_TOO_SHORT,  # check 6 (model output only)
        R.ENTRY_TOO_LONG,
        R.TOO_FEW_SENTENCES,
        R.TOO_MANY_SENTENCES,
        R.GROUP_NOT_MENTIONED,
        R.URL_IN_ENTRY,
        R.QUESTION_IN_ENTRY,
        R.ENTRY_NOT_ENGLISH,
    }
)

MIN_WORDS, MAX_WORDS = 5, 70
MIN_SENTENCES, MAX_SENTENCES = 1, 4
MAX_DEVANAGARI_LETTER_SHARE = 0.2

AMBIGUOUS_NUMBER_ALIASES: frozenset[str] = frozenset(
    {
        "one", "do", "few", "a few", "first", "third", "half", "many", "both", "pair",
        "pairs", "a couple", "a couple of", "couple of", "teen", "char", "das", "tees",
        "sau", "nau",
    }
)  # fmt: skip
AMBIGUOUS_WORD_ALIASES: frozenset[str] = frozenset(
    {"well", "rose", "os", "hon", "bail", "vat", "vad", "bel", "mor", "pul", "tara", "chand"}
)
_DETERMINERS = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "some", "its", "their", "his", "her",
     "our", "my", "your", "one", "old", "big", "small", "little", "new"}
)  # fmt: skip
_SKIP_BEFORE_NOUN = frozenset({"of", "the", "a", "an"})
_ARTICLES = frozenset({"the", "a", "an"})
SAFE_CAPITALISED: frozenset[str] = frozenset({"i", "mom", "dad"})
NAME_STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "they", "he", "she", "it", "we", "you", "this", "that", "these",
     "those", "their", "his", "her", "our", "my", "and", "but", "or", "then", "am", "pm",
     "i'm", "i've", "i'd"}
)  # fmt: skip
SENTENCE_STARTERS: frozenset[str] = frozenset(
    {"there", "here", "its", "it's", "there's", "that's", "they're", "we're", "let's",
     "now", "today", "yesterday", "later", "earlier", "soon", "afterwards", "afterward",
     "after", "before", "during", "while", "when", "whenever", "as", "once", "since",
     "until", "till", "on", "in", "at", "by", "near", "nearby", "along", "across", "around",
     "past", "beside", "behind", "under", "over", "above", "below", "beyond", "through",
     "throughout", "from", "with", "without", "within", "outside", "inside", "into", "onto",
     "towards", "toward", "up", "down", "out", "off", "among", "between", "about", "like",
     "unlike", "despite", "back", "home", "both", "all", "each", "every", "everyone",
     "everybody", "everything", "nothing", "nobody", "none", "someone", "somebody",
     "something", "somewhere", "some", "many", "few", "most", "much", "more", "other",
     "another", "such", "no", "not", "neither", "either", "also", "still", "even", "just",
     "only", "so", "yet", "because", "though", "although", "if", "again", "overall",
     "finally", "first", "next", "last", "together", "instead", "meanwhile", "suddenly",
     "slowly", "quietly", "gently", "usually", "eventually", "luckily", "happily",
     "actually", "really", "probably", "apparently", "sadly", "thankfully", "fortunately",
     "unfortunately", "sometimes", "often", "always", "never", "what", "which", "who",
     "whose", "where", "how", "why", "yes", "oh", "wow", "very", "quite", "almost",
     "halfway", "nice", "quiet", "lovely", "beautiful", "warm", "cool", "cold", "fresh",
     "little", "big", "small", "old", "new", "long", "short", "good", "great", "whole",
     "evening", "walk", "walks", "walked", "went", "came", "saw", "heard", "felt", "noticed",
     "spotted", "watched", "stopped", "paused", "returned", "headed", "passed"}
)  # fmt: skip
_GROUP_STOP = frozenset({"and", "&"})
_DAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"}
)  # fmt: skip
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "mt", "vs", "etc", "e.g", "i.e",
     "a.m", "p.m", "approx"}
)  # fmt: skip
_NUMBER_ID_VALUES: Mapping[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty_five": 25, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}  # fmt: skip

_LATIN_CH = "0-9A-Za-zÀ-ɏ"
_TOKEN_RE = re.compile(rf"[{_LATIN_CH}]+(?:['’][{_LATIN_CH}]+)*")
_DEVA_RE = re.compile(r"[ऀ-ॿ]")
_DEVA_WORD_RE = re.compile(r"[ऀ-ॣ०-ॿ]+")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:,\d{3})*(?:\.\d+)?)(?!\w)")
_AMPM_RE = re.compile(r"(?<!\w)(\d{1,2})(?:[:.](\d{2}))?\s?(a\.m\.|p\.m\.|am|pm)(?!\w)")
_CLOCK_RE = re.compile(r"(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])")
_OCLOCK_RE = re.compile(r"(?<![a-z])(o'clock|oclock|baje)(?![a-z])|बजे")
_NOON_RE = re.compile(r"(?<![a-z])(noon|midnight)(?![a-z])")
_TERM_RE = re.compile(r"[.!?…।॥]+")
_CLOSERS = "\"')]’”"
_URL_RE = re.compile(
    r"(?i)(?:https?://|\bwww\.|\b[a-z0-9-]+\.(?:com|org|net|in|io|co|uk|ly|app|dev|me|info|"
    r"gov|edu|biz|xyz)\b)"
)
_QUESTION_CHARS = ("?", "？", "¿")


class GroundingError(ValueError):
    """Configuration error (lexicon or hedges). Messages are fixed; they never carry content."""


# --------------------------------------------------------------------------- text helpers


def normalise(text: str) -> str:
    """NFC, casefold, collapse whitespace, strip (PRD §8.1). Devanagari is kept intact."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _low(token: str) -> str:
    return token.casefold().replace("’", "'")


def _is_latin_alnum(ch: str) -> bool:
    return (ch.isascii() and ch.isalnum()) or "À" <= ch <= "ɏ"


def _find_bounded(hay: str, needle: str) -> bool:
    """Substring search that respects Latin token boundaries at both ends (I2)."""
    if not needle:
        return False
    left_alnum = _is_latin_alnum(needle[0])
    right_alnum = _is_latin_alnum(needle[-1])
    start = 0
    while (i := hay.find(needle, start)) != -1:
        j = i + len(needle)
        ok_left = i == 0 or not (left_alnum and _is_latin_alnum(hay[i - 1]))
        ok_right = j == len(hay) or not (right_alnum and _is_latin_alnum(hay[j]))
        if ok_left and ok_right:
            return True
        start = i + 1
    return False


def _tokens(text: str) -> list[tuple[int, str, str]]:
    """Latin tokens as (start, raw, lowered) on NFC text."""
    return [(m.start(), m.group(), _low(m.group())) for m in _TOKEN_RE.finditer(text)]


def _token_key(text: str) -> str:
    return " ".join(low for _, _, low in _tokens(normalise(text)))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _TERM_RE.finditer(text):
        end = m.end()
        nxt = text[end : end + 1]
        if nxt and not (nxt.isspace() or nxt in _CLOSERS):
            continue
        if m.group() == ".":
            prev = re.search(r"(\S+)$", text[start : m.start()])
            word = prev.group(1).lower().lstrip("(\"'") if prev else ""
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha() and word != "i"):
                continue
        spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if any(c.isalnum() for c in text[s:e])]


def _split_sentences(text: str) -> list[str]:
    nfc = unicodedata.normalize("NFC", text)
    return [nfc[s:e].strip() for s, e in _sentence_spans(nfc)]


def _word_count(text: str) -> int:
    return sum(1 for w in text.split() if any(c.isalnum() for c in w))


# --------------------------------------------------------------------------- lexicon


@dataclass(frozen=True, eq=False)
class Lexicon:
    """The frozen detail lexicon plus hedge cues, pre-indexed for matching."""

    latin: Mapping[str, str]  # token key ("gul mohar") -> lexicon id
    deva: tuple[tuple[str, str], ...]  # (alias, id), longest first
    max_ngram: int
    categories: Mapping[str, str]  # id -> category
    number_ids: Mapping[int, str]  # value -> number id
    id_values: Mapping[str, int]  # number id -> value
    hedge_latin: frozenset[str]
    hedge_deva: tuple[str, ...]
    entry_hedges: frozenset[str]

    def alias_at(self, lows: Sequence[str], i: int) -> tuple[str, int] | None:
        """Longest Latin alias starting at token i, as (key, length)."""
        for n in range(min(self.max_ngram, len(lows) - i), 0, -1):
            key = " ".join(lows[i : i + n])
            if key in self.latin:
                return key, n
        return None


def _read_yaml(path: Path) -> Any:
    data: Any = None
    failed = False
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError):
        failed = True
    if failed:  # raised outside the except block so no content-bearing context is chained
        raise GroundingError("grounding config file could not be read")
    return data


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise GroundingError("grounding config file is malformed")
    return value


@lru_cache(maxsize=8)
def load_lexicon(path: Path = DEFAULT_LEXICON, hedges_path: Path = DEFAULT_HEDGES) -> Lexicon:
    """Load and index the lexicon and hedge cues (cached per path pair)."""
    raw = _read_yaml(path)
    hed = _read_yaml(hedges_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise GroundingError("grounding config file is malformed")
    if not isinstance(hed, dict) or not isinstance(hed.get("cues"), dict):
        raise GroundingError("grounding config file is malformed")

    latin: dict[str, str] = {}
    deva: dict[str, str] = {}
    categories: dict[str, str] = {}
    for entry in raw["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise GroundingError("grounding config file is malformed")
        eid: str = entry["id"]
        categories[eid] = str(entry.get("category", ""))
        for alias in _str_list(entry.get("aliases")):
            norm = normalise(alias)
            key = _token_key(norm)
            if _DEVA_RE.search(norm) and not key:
                deva.setdefault(norm, eid)
            elif key:
                latin.setdefault(key, eid)

    hedge_latin: set[str] = set()
    hedge_deva: list[str] = []
    for cues in hed["cues"].values():
        for cue in _str_list(cues):
            norm = normalise(cue)
            if _DEVA_RE.search(norm):
                hedge_deva.append(norm)
            elif key := _token_key(norm):
                hedge_latin.add(key)
    entry_hedges = {k for w in _str_list(hed.get("entry_hedge_words")) if (k := _token_key(w))}

    id_values = {i: v for i, v in _NUMBER_ID_VALUES.items() if i in categories}
    return Lexicon(
        latin=latin,
        deva=tuple(sorted(deva.items(), key=lambda kv: -len(kv[0]))),
        max_ngram=max((k.count(" ") + 1 for k in latin), default=1),
        categories=categories,
        number_ids={v: i for i, v in id_values.items()},
        id_values=id_values,
        hedge_latin=frozenset(hedge_latin),
        hedge_deva=tuple(hedge_deva),
        entry_hedges=frozenset(entry_hedges),
    )


def _inflected(word: str) -> set[str]:
    forms = {word, word + "s", word + "es", word + "ed", word + "d", word + "ing"}
    if word.endswith("e"):
        forms.add(word[:-1] + "ing")
    return forms


@lru_cache(maxsize=4)
def _common_words(path: Path = DEFAULT_WALK_WORDS) -> frozenset[str]:
    """walk_words.yaml generic nouns + allowlisted verbs (simple inflections), lowercased."""
    raw = _read_yaml(path)
    if not isinstance(raw, dict):
        raise GroundingError("grounding config file is malformed")
    words: set[str] = set()
    for w in _str_list(raw.get("generic_nouns")):
        words.add(_low(w))
    for v in _str_list(raw.get("verb_allowlist")):
        words |= _inflected(_low(v))
    return frozenset(words)


def lexicon_hash(path: Path = DEFAULT_LEXICON) -> str:
    """sha256 hex digest of the lexicon file bytes (PRD §12.1.6)."""
    data = b""
    failed = False
    try:
        data = Path(path).read_bytes()
    except (OSError, TypeError):
        failed = True
    if failed:
        raise GroundingError("grounding config file could not be read")
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- details


@dataclass(frozen=True, slots=True)
class _Detail:
    key: str  # canonical: lexicon id, "num:<n>", "time:...", "day:...", "month:...", "name:..."
    surface: str  # the entry words; local eval output only, never logged
    kind: str  # "lex" | "num" | "time" | "name" | "initial_name" (I11, check_entry only)


def _num_key(value: str, lex: Lexicon) -> str:
    clean = value.replace(",", "")
    try:
        n = int(clean)
    except ValueError:
        return f"num:{float(clean):g}"
    return lex.number_ids.get(n, f"num:{n}")


def _word_value(word: str, lex: Lexicon) -> int | None:
    w = word.strip(".,;:!?\"'()")
    if w.isdigit():
        return int(w)
    nw = normalise(w)
    eid = lex.latin.get(nw) or dict(lex.deva).get(nw)
    return lex.id_values.get(eid) if eid else None


def _kind(eid: str, lex: Lexicon) -> str:
    return "num" if lex.categories.get(eid) == "number" else "lex"


def _time_details(nfc: str, lex: Lexicon, inclusive: bool) -> list[_Detail]:
    low = _low(nfc)
    out: list[_Detail] = []
    taken: list[tuple[int, int]] = []
    for m in _AMPM_RE.finditer(low):
        h, mm = int(m.group(1)), m.group(2)
        key = f"time:{h}" if not mm or mm == "00" else f"time:{h}:{mm}"
        out.append(_Detail(key, nfc[m.start() : m.end()], "time"))
        taken.append(m.span())
    for m in _CLOCK_RE.finditer(low):
        if any(a <= m.start() < b for a, b in taken):
            continue
        h, mm = int(m.group(1)), m.group(2)
        key = f"time:{h}" if mm == "00" else f"time:{h}:{mm}"
        out.append(_Detail(key, nfc[m.start() : m.end()], "time"))
    for m in _OCLOCK_RE.finditer(low):
        before = low[: m.start()].split()
        value = _word_value(before[-1], lex) if before else None
        key = f"time:{value}" if value is not None else "time:clock"
        out.append(_Detail(key, nfc[m.start() : m.end()], "time"))
    for m in _NOON_RE.finditer(low):
        out.append(_Detail(f"time:{m.group(1)}", nfc[m.start() : m.end()], "time"))
    for _, raw, lw in _tokens(nfc):
        if lw in _DAYS:
            out.append(_Detail(f"day:{lw}", raw, "time"))
        elif lw in _MONTHS and (inclusive or raw[0].isupper()):
            out.append(_Detail(f"month:{lw}", raw, "time"))
    return out


def _initial_candidate(raw: str, lw: str, lows: Sequence[str], idx: int, lex: Lexicon) -> bool:
    """I11: is this sentence-initial token a possible person name?"""
    base = lw[:-2] if lw.endswith("'s") else lw
    return not (
        not raw[0].isupper()
        or base in SAFE_CAPITALISED
        or base in NAME_STOPWORDS
        or base in SENTENCE_STARTERS
        or base in _DAYS
        or base in _MONTHS
        or any(c.isdigit() for c in base)
        or lex.alias_at(lows, idx) is not None
        or base in lex.latin
        or base in lex.hedge_latin
        or base in lex.entry_hedges
        or base in _common_words()
        or (len(base) >= 5 and base.endswith("ing"))
    )


def _scan(
    text: str,
    lex: Lexicon,
    *,
    inclusive: bool,
    group: str | None = None,
    initial_names: bool = False,
) -> list[_Detail]:
    nfc = unicodedata.normalize("NFC", text)
    toks = _tokens(nfc)
    lows = [t[2] for t in toks]
    group_words = {
        w for w in (_low(t) for t in normalise(group or "").split()) if w not in _GROUP_STOP
    }
    excluded = [lw in group_words for lw in lows]
    consumed = [False] * len(toks)
    out: list[_Detail] = []

    # Latin lexicon aliases
    i = 0
    while i < len(toks):
        if excluded[i]:
            i += 1
            continue
        if inclusive:
            for n in range(1, min(lex.max_ngram, len(toks) - i) + 1):
                key = " ".join(lows[i : i + n])
                if key in lex.latin and not any(excluded[i : i + n]):
                    out.append(_Detail(lex.latin[key], key, "lex"))
            i += 1
            continue
        hit = lex.alias_at(lows, i)
        if hit is None or any(excluded[i : i + hit[1]]):
            i += 1
            continue
        key, n = hit
        if key in AMBIGUOUS_NUMBER_ALIASES:
            j = i + n
            while j < len(toks) and lows[j] in _SKIP_BEFORE_NOUN:
                j += 1
            nxt = lex.alias_at(lows, j) if j < len(toks) else None
            counted = nxt is not None and nxt[0] not in AMBIGUOUS_NUMBER_ALIASES
        elif key in AMBIGUOUS_WORD_ALIASES:
            prev = lows[i - 1] if i > 0 else ""
            counted = prev in _DETERMINERS or prev in lex.latin
        else:
            counted = True
        if counted:
            eid = lex.latin[key]
            surface = nfc[toks[i][0] : toks[i + n - 1][0] + len(toks[i + n - 1][1])]
            out.append(_Detail(eid, surface, _kind(eid, lex)))
            for k in range(i, i + n):
                consumed[k] = True
        i += n

    # Devanagari aliases (substring)
    if _DEVA_RE.search(nfc):
        rest = normalise(nfc)
        for w in group_words:
            if _DEVA_RE.search(w):
                rest = rest.replace(w, " ")
        for alias, eid in lex.deva:
            if inclusive:
                if alias in rest:
                    out.append(_Detail(eid, alias, "lex"))
                continue
            while alias in rest:
                out.append(_Detail(eid, alias, _kind(eid, lex)))
                rest = rest.replace(alias, "\x00" * len(alias), 1)

    # digits (ASCII and Devanagari)
    for m in _NUM_RE.finditer(nfc):
        out.append(_Detail(_num_key(m.group(1), lex), m.group(1), "num"))

    out.extend(_time_details(nfc, lex, inclusive))

    # capitalised proper nouns
    initial: set[int] = set()
    spans = _sentence_spans(nfc)
    for s, _e in spans:
        for idx, (start, _, _) in enumerate(toks):
            if start >= s:
                initial.add(idx)
                break
    for idx, (_, raw, lw) in enumerate(toks):
        if (
            initial_names
            and idx in initial
            and not excluded[idx]
            and not consumed[idx]
            and _initial_candidate(raw, lw, lows, idx, lex)
        ):
            name = lw[:-2] if lw.endswith("'s") else lw
            out.append(_Detail(f"name:{name}", raw, "initial_name"))
            continue
        if (
            not raw[0].isupper()
            or idx in initial
            or excluded[idx]
            or consumed[idx]
            or lw in SAFE_CAPITALISED
            or lw in NAME_STOPWORDS
            or lw in _DAYS
            or lw in _MONTHS
            or lw.isdigit()
        ):
            continue
        name = lw[:-2] if lw.endswith("'s") else lw
        out.append(_Detail(f"name:{name}", raw, "name"))
    return out


def extract_details(
    text: str,
    lex: Lexicon | None = None,
    *,
    inclusive: bool = False,
    group: str | None = None,
) -> set[str]:
    """Canonical concrete details in ``text`` (PRD §8.4/§8.5, §13.4).

    Keys: lexicon ids ("koel", "two"), "num:<n>", "time:<h>[:<mm>]", "time:clock",
    "time:noon", "day:<name>", "month:<name>", "name:<lowercased>". ``inclusive=False``
    (entry/mission side) applies the ambiguity rules (I8); ``inclusive=True`` (evidence
    side: transcripts, spans) counts every alias occurrence.
    """
    return {d.key for d in _scan(text, lex or load_lexicon(), inclusive=inclusive, group=group)}


# --------------------------------------------------------------------------- check helpers


def _covered(phrase: str, span_n: str, span_keys: set[str], lex: Lexicon) -> bool:
    """Check 2: phrase is a boundaried substring of the span, or fully alias-covered (I3)."""
    pn = normalise(phrase)
    if not pn:
        return False
    if _find_bounded(span_n, pn):
        return True
    lows = [t[2] for t in _tokens(pn)]
    matched = False
    i = 0
    while i < len(lows):
        hit = lex.alias_at(lows, i)
        if hit is not None:
            if lex.latin[hit[0]] not in span_keys:
                return False
            matched = True
            i += hit[1]
            continue
        if lows[i] in _ARTICLES:
            i += 1
            continue
        if not _find_bounded(span_n, lows[i]):
            return False
        matched = True
        i += 1
    rest = _TOKEN_RE.sub(" ", pn)
    for alias, eid in lex.deva:
        if alias in rest:
            if eid not in span_keys:
                return False
            matched = True
            rest = rest.replace(alias, " ")
    for word in _DEVA_WORD_RE.findall(rest):
        if word not in span_n:
            return False
        matched = True
    return matched


def _padded(text_n: str) -> str:
    return " " + " ".join(t[2] for t in _tokens(text_n)) + " "


def _has_cue(text_n: str, latin: Iterable[str], deva: Iterable[str]) -> bool:
    padded = _padded(text_n)
    return any(f" {c} " in padded for c in latin) or any(c in text_n for c in deva)


def _lex_keys(details: Iterable[_Detail]) -> set[str]:
    return {d.key for d in details if d.kind in ("lex", "num")}


def _dedupe[T](items: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(items))


# --------------------------------------------------------------------------- check_entry


@dataclass(frozen=True, repr=False)
class CheckResult:
    passed: bool
    reasons: tuple[ReasonCode, ...]  # enum codes only (deduped, stable order)
    # facts passing checks 1-3 (only these enter memory); text-bearing, kept out of repr
    verified_facts: tuple[Fact, ...] = field(repr=False)
    dropped_facts: int
    mission_result: MissionResult  # possibly coerced to NOT_MENTIONED by check 7
    mission_span: str | None = field(repr=False)  # transcript words; kept out of repr
    # LOCAL EVAL OUTPUT ONLY. Entry words: never log, print, trace or send anywhere (NFR-P4).
    unsupported_details: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        """Codes and counts only (I16): safe if a result is ever logged by mistake."""
        return (
            f"CheckResult(passed={self.passed}, "
            f"reasons={tuple(str(c) for c in self.reasons)!r}, "
            f"verified_facts={len(self.verified_facts)}, dropped_facts={self.dropped_facts}, "
            f"mission_result={str(self.mission_result)!r}, "
            f"mission_span={'set' if self.mission_span is not None else None!r}, "
            f"unsupported_details={len(self.unsupported_details)})"
        )


def check_entry(
    output: WriterOutput,
    transcript: str,
    group: str,
    *,
    origin: EntryOrigin = EntryOrigin.MODEL,
    lex: Lexicon | None = None,
) -> CheckResult:
    """Run PRD §8 checks 1-7 on one writer output. Never raises on content; never logs."""
    if not isinstance(output, WriterOutput):
        raise TypeError("output must be a WriterOutput")
    if not isinstance(transcript, str) or not isinstance(group, str):
        raise TypeError("transcript and group must be str")
    lex = lex or load_lexicon()
    reasons: list[ReasonCode] = []
    t_norm = normalise(transcript)

    # ---- checks 1-3 (per fact)
    verified: list[Fact] = []
    evidence: list[tuple[str, set[str]]] = []  # (normalised text, keys) of verified facts
    needs_hedge: list[Fact] = []
    dropped = 0
    for fact in output.facts:
        span_n = normalise(fact.span)
        if not _find_bounded(t_norm, span_n):
            reasons.append(R.SPAN_NOT_IN_TRANSCRIPT)
            dropped += 1
            continue
        span_keys = {d.key for d in _scan(fact.span, lex, inclusive=True)}
        if not _covered(fact.subject, span_n, span_keys, lex):
            reasons.append(R.SUBJECT_NOT_IN_SPAN)
            dropped += 1
            continue
        place = fact.place if fact.place and fact.place.strip() else None
        if place is not None and not _covered(place, span_n, span_keys, lex):
            reasons.append(R.PLACE_NOT_IN_SPAN)
            dropped += 1
            continue
        hedged = _has_cue(span_n, lex.hedge_latin, lex.hedge_deva)
        if hedged or fact.certainty is Certainty.UNSURE:
            needs_hedge.append(fact)
        if hedged and fact.certainty is Certainty.SURE:
            reasons.append(R.HEDGE_CERTAINTY_MISMATCH)
            dropped += 1
            continue
        verified.append(fact)
        keys = set(span_keys)
        for extra in (fact.subject, place or ""):
            keys |= {d.key for d in _scan(extra, lex, inclusive=True)}
        evidence.append(
            (" | ".join(normalise(x) for x in (fact.span, fact.subject, place or "")), keys)
        )

    entry = unicodedata.normalize("NFC", output.entry)

    # ---- check 3 (entry side)
    sentences = [entry[s:e] for s, e in _sentence_spans(entry)]
    for fact in needs_hedge:
        subj_n = normalise(fact.subject)
        subj_keys = _lex_keys(_scan(fact.subject, lex, inclusive=True))
        for sent in sentences:
            sent_n = normalise(sent)
            mentioned = _find_bounded(sent_n, subj_n) or bool(
                subj_keys & _lex_keys(_scan(sent, lex, inclusive=True))
            )
            if mentioned and not _has_cue(sent_n, lex.entry_hedges, ()):
                reasons.append(R.HEDGE_MISSING_IN_ENTRY)
                break

    # ---- checks 4 and 5
    details = _scan(entry, lex, inclusive=False, group=group, initial_names=True)
    t_keys = {d.key for d in _scan(transcript, lex, inclusive=True)}
    orphans: list[str] = []
    new_codes: list[ReasonCode] = []
    for d in details:
        if d.kind == "initial_name":
            if _find_bounded(t_norm, d.key.removeprefix("name:")):
                continue  # I11: a word the transcript uses is not an invented name
            orphans.append(d.surface)
            new_codes.append(R.NEW_PERSON_NAME)
            continue
        if d.kind == "name":
            name = d.key.removeprefix("name:")
            supported = any(_find_bounded(txt, name) for txt, _ in evidence)
            present = _find_bounded(t_norm, name)
            code = R.NEW_PERSON_NAME
        else:
            supported = any(d.key in keys for _, keys in evidence)
            present = d.key in t_keys
            code = R.NEW_NUMBER if d.kind == "num" else R.NEW_TIME
        if not supported:
            orphans.append(d.surface)
        if d.kind != "lex" and not present:
            new_codes.append(code)
    if orphans:
        reasons.append(R.UNSUPPORTED_DETAIL)
    reasons.extend(c for c in (R.NEW_NUMBER, R.NEW_TIME, R.NEW_PERSON_NAME) if c in new_codes)

    # ---- check 6 (model output only)
    if origin is EntryOrigin.MODEL:
        words = _word_count(entry)
        if words < MIN_WORDS:
            reasons.append(R.ENTRY_TOO_SHORT)
        elif words > MAX_WORDS:
            reasons.append(R.ENTRY_TOO_LONG)
        if len(sentences) < MIN_SENTENCES:
            reasons.append(R.TOO_FEW_SENTENCES)
        elif len(sentences) > MAX_SENTENCES:
            reasons.append(R.TOO_MANY_SENTENCES)
        if not _find_bounded(normalise(entry), normalise(group)):
            reasons.append(R.GROUP_NOT_MENTIONED)
        if _URL_RE.search(entry):
            reasons.append(R.URL_IN_ENTRY)
        if any(q in entry for q in _QUESTION_CHARS):
            reasons.append(R.QUESTION_IN_ENTRY)
        letters = [c for c in entry if c.isalpha()]
        deva = sum(1 for c in letters if _DEVA_RE.match(c))
        if letters and deva / len(letters) > MAX_DEVANAGARI_LETTER_SHARE:
            reasons.append(R.ENTRY_NOT_ENGLISH)

    # ---- check 7
    result = output.mission_result
    m_span: str | None = output.mission_span
    if result is MissionResult.NOT_MENTIONED:
        m_span = None
    elif m_span is None or not normalise(m_span):
        reasons.append(R.MISSION_SPAN_MISSING)
        result, m_span = MissionResult.NOT_MENTIONED, None
    elif not _find_bounded(t_norm, normalise(m_span)):
        reasons.append(R.MISSION_SPAN_NOT_IN_TRANSCRIPT)
        result, m_span = MissionResult.NOT_MENTIONED, None

    codes = _dedupe(reasons)
    return CheckResult(
        passed=not any(c in FAILING_CODES for c in codes),
        reasons=codes,
        verified_facts=tuple(verified),
        dropped_facts=dropped,
        mission_result=result,
        mission_span=m_span,
        unsupported_details=_dedupe(orphans),
    )


# --------------------------------------------------------------------------- templates


def _join_and(items: Sequence[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_template_entry(group: str, verified_facts: Sequence[Fact], mission_done: bool) -> str:
    """PRD §8 template entries, built by code from verified facts only (I15)."""
    name = " ".join(group.split())
    shown: dict[str, str] = {}
    unsure: set[str] = set()
    for fact in verified_facts:
        subject = " ".join(fact.subject.split()).rstrip(".,;:!?")
        key = normalise(subject)
        if not key:
            continue
        shown.setdefault(key, subject)
        if fact.certainty is Certainty.UNSURE:
            unsure.add(key)
    if not shown:
        return f"{name} went for their evening walk."
    sure = [s for k, s in shown.items() if k not in unsure]
    maybe = [f"maybe {s}" for k, s in shown.items() if k in unsure]
    noticed = ", ".join(([_join_and(sure)] if sure else []) + maybe)
    clause = " and did today's mission" if mission_done else ""
    return f"{name} went for a walk{clause}. They noticed {noticed}."


def template_output(group: str, check: CheckResult) -> WriterOutput:
    """The FR-J5 fallback: verified facts kept, template entry, check-7 mission result."""
    done = check.mission_result is MissionResult.DONE
    return WriterOutput(
        facts=list(check.verified_facts),
        mission_result=check.mission_result,
        mission_span=check.mission_span
        if check.mission_result is not MissionResult.NOT_MENTIONED
        else None,
        entry=build_template_entry(group, check.verified_facts, done),
    )
