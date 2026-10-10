"""Mission safety filter (PRD FR-M1, FR-M4, FR-M5, FR-M6 a-d, FR-M6b).

Pure, deterministic and offline. Inputs are a planner output (or a hand-written
fallback/seed text) plus the verified facts it may reference; the output is a
:class:`SafetyResult` that carries enum codes only, never text (NFR-P4).

Data comes from ``data/walk_words.yaml`` (generic nouns, verb allowlist, blocklist,
time-of-day words), ``data/lexicon.yaml`` (concrete-detail aliases across scripts) and
``data/hedges.yaml`` (``entry_hedge_words``, for FR-M5). All text is NFC-normalised,
curly apostrophes folded and whitespace collapsed before any rule runs.

Rules, in the order their codes are reported
--------------------------------------------
* FR-M6(a) ``VERB_NOT_ALLOWED``: the task verb(s) must be on the allowlist. See
  "Main verb" below.
* FR-M6(b) ``BLOCKLISTED``: any blocklist regex matches (``block_categories`` says which).
* FR-M4 ``NOUN_NOT_PROVENANCED``: every concrete noun is a generic walk word or comes
  from a *referenced* fact's subject/place, alias-aware via the lexicon (koel, koyal and
  कोयल are one entry). Concrete nouns are spaCy NOUN/PROPN tokens plus any lexicon alias
  (multi-word aware; colour and number entries are adjectives/quantities, not nouns, and
  are exempt). ``REFERENCE_UNKNOWN`` for ids not among the supplied facts (or, when
  ``group`` is given, facts of another walker group; those supply no nouns either) and
  ``RETURN_WITHOUT_REFERENCE`` for a return mission with no references. ``facts_by_id`` is
  the in-scope set: the last-7-walks window (FR-M3) is enforced by the query that builds it
  (``hub.db.verified_facts_for_group``), not here.
* FR-M5 ``UNSURE_NOT_QUESTION``: if any referenced fact is unsure, the mission ends with
  "?", the question sentence says "you", and every non-question sentence that mentions
  the unsure subject carries a hedge word ("You thought you heard...").
* ``NOT_ADDRESSED_TO_GROUP``: the mission says "you"/"your" and never singles out one
  person (he/she, Mom/Dad/Papa...). PRD §3: Sair has no per-person attribution.
* FR-M6b ``TIME_OF_DAY``: no time-of-day word or phrase from walk_words.yaml (am/pm only
  after a digit or as a.m./p.m.; "gets dark", "sun goes down", ...), no clock time such as
  5:30, and no bare clock hour after a time preposition (``_BARE_HOUR_RE``: at, around, by,
  after, before, till, until, from + 1-12 as a digit or an English/Hinglish number word,
  optionally "-ish"). The bare hour counts only at a clause edge (followed by punctuation,
  the end, or one of on/in/at/and/or/to/today/tonight/sharp/baje/o'clock), so "by one
  tree" and "after two gates" stay allowed while "around 6 on your path" and "again at 6?"
  are rejected.
* FR-M1/M6(d) ``MISSION_TOO_LONG`` (> 30 words) and ``MULTIPLE_TASKS``.
* FR-M3 ``MISSION_REPEATED``: equal to, or a near-duplicate of, a recent mission.

Main verb (FR-M6a, FR-M1)
-------------------------
The text is split into sentences on ``. ? !``; at most 2 are allowed. Each sentence is
parsed with spaCy ``en_core_web_sm`` (loaded once, lazily, from the installed package)
and classified:

1. **Question** (ends with "?"): a task sentence. Its task verbs are the top-most verbs
   whose subject is "you" ("Can you *find* out?", "Do you *hear* it?"), plus their
   conjuncts. Verbs with another subject describe the world ("Is it still flowering?")
   and copula/auxiliary-only questions ("Is it there again?") have no task verb.
2. **Context** (declarative with an explicit subject): either the subject is not "you"
   ("The gulmohar was in flower.") or it is "you" and the verb is past tense ("Last time
   you *heard* a koel", "You *thought* you heard..."). Context verbs are not task verbs.
3. **Imperative** (everything else, including "you" + non-past): a task sentence. Its
   task verbs are the ROOT (whatever its POS tag; the small model often tags a leading
   "Notice" as a noun) plus its conj/parataxis verbs.

Every task verb must be an allowlisted verb (any inflection). Verbs *subordinate* to a
task verb whose subject is "you", or open complements of a task verb ("try to spot"),
are actions too: they must be allowlisted or one of a few benign motion/state verbs
(``_BENIGN_SUBORDINATE``: "the gates you *pass*", "as you *walk*"). Exactly one task
sentence is required (none → ``VERB_NOT_ALLOWED``; more → ``MULTIPLE_TASKS``), and it may
carry one task verb, optionally preceded by "stand"/"pause" ("Pause by the gate and
listen" is one task). The rule is deliberately strict: a false reject costs only one
regeneration or a fixed fallback (FR-M6).

``provenance=False`` disables only the FR-M4 noun check. It exists solely for the PRD
§13.4 headline contamination runs (filter OFF in both arms). Production never passes it.
"""

from __future__ import annotations

import difflib
import functools
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from hub.schemas import BlockCategory, Certainty, MissionPlan, MissionType, ReasonCode, VerifiedFact

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

__all__ = ["SafetyResult", "check_fallback_text", "check_mission", "mentions_time_of_day"]

MAX_MISSION_WORDS = 30
MAX_SENTENCES = 2
REPEAT_SIMILARITY = 0.9  # difflib ratio on normalised text; >= this is a near-duplicate

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SPACY_MODEL = "en_core_web_sm"

# Allowed and frequent in context sentences, but missing from walk_words.generic_nouns.
# Reported upstream as candidates for generic_nouns; kept minimal on purpose.
_STRUCTURAL_NOUNS = frozenset(
    {
        "time",
        "times",
        "day",
        "days",
        "week",
        "weeks",
        "today",
        "yesterday",
        "start",
        "lot",
        "lots",
        "bit",
        "while",
    }
)
_WEEKDAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)
# Verbs allowed in clauses subordinate to the task ("the gates you pass", "as you walk").
_BENIGN_SUBORDINATE = frozenset(
    {
        "walk",
        "pass",
        "go",
        "come",
        "return",
        "head",
        "turn",
        "reach",
        "stroll",
        "move",
        "leave",
        "start",
        "finish",
        "end",
        "be",
        "have",
        "do",
        "get",
        "like",
        "want",
        "think",
        "remember",
        "wonder",
        "know",
        "seem",
        "can",
        "will",
        "may",
        "might",
    }
)
_PREAMBLE_VERBS = frozenset({"stand", "pause"})
_IRREGULAR = {
    "see": {"saw", "seen"},
    "hear": {"heard"},
    "find": {"found"},
    "stand": {"stood"},
    "smell": {"smelt"},
    "spot": {"spotted", "spotting"},
}
_EXEMPT_LEXICON_CATEGORIES = frozenset({"colour", "number"})
_FUNCTION_POS = frozenset(
    {"ADV", "ADP", "AUX", "VERB", "PRON", "DET", "CCONJ", "SCONJ", "PART", "INTJ"}
)
_NON_OPENER_POS = frozenset(
    {"PRON", "DET", "ADP", "ADV", "ADJ", "SCONJ", "CCONJ", "NUM", "AUX", "PART", "INTJ", "PUNCT"}
)
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "expl"})
_PAST_TAGS = frozenset({"VBD", "VBN"})

_YOU_RE = re.compile(r"\byou(?:r|rs|rself|rselves|'re|'ve|'ll|'d)?\b", re.IGNORECASE)
_INDIVIDUAL_RE = re.compile(
    r"\b(?:he|she|him|her|his|hers|himself|herself|mom|mum|mummy|mommy|mother|maa|mama|"
    r"ammi|amma|dad|daddy|papa|father|abba|abbu|appa|baba|husband|wife|grandma|grandpa|"
    r"nana|nani|dada|dadi)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\d\s*(?:am|pm)\b|\b[ap]\.\s?m\.", re.IGNORECASE)
_HOUR_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "ek|do|teen|char|chaar|paanch|panch|chhe|chhah|chheh|che|saat|aath|nau|das|"
    "gyarah|gyaarah|barah|baarah"
)
_BARE_HOUR_RE = (
    r"\b(?:at|around|by|after|before|till|until|from|about)\s+"
    rf"(?:(?:1[0-2]|0?[1-9])|(?:{_HOUR_WORDS}))(?:\s?ish)?\b"
    r"(?=\s*(?:[.,;:!?)]|$)|\s+(?:on|in|at|and|or|to|today|tonight|sharp|baje|o'clock)\b)"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")
_TOKEN_RE = re.compile(r"[^\s.,!?;:\"()\[\]{}]+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

_REASON_ORDER: tuple[ReasonCode, ...] = (
    ReasonCode.VERB_NOT_ALLOWED,
    ReasonCode.BLOCKLISTED,
    ReasonCode.NOUN_NOT_PROVENANCED,
    ReasonCode.REFERENCE_UNKNOWN,
    ReasonCode.RETURN_WITHOUT_REFERENCE,
    ReasonCode.UNSURE_NOT_QUESTION,
    ReasonCode.NOT_ADDRESSED_TO_GROUP,
    ReasonCode.TIME_OF_DAY,
    ReasonCode.MISSION_TOO_LONG,
    ReasonCode.MULTIPLE_TASKS,
    ReasonCode.MISSION_REPEATED,
)


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of the filter: enum codes only (NFR-P4)."""

    passed: bool
    reasons: tuple[ReasonCode, ...]
    block_categories: tuple[BlockCategory, ...]


# ------------------------------------------------------------------- normalisation


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def _words(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(_norm(text).lower()) if any(c.isalnum() for c in w)]


def _variants(word: str) -> set[str]:
    """The word plus naive singular forms (gates -> gate, benches -> bench)."""
    out = {word}
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.add(word[:-2])
    if word.endswith("s") and len(word) > 2:
        out.add(word[:-1])
    return out


def _latin_term_re(term: str) -> str:
    return r"(?<!\w)" + re.escape(term) + r"(?!\w)"


# --------------------------------------------------------------------- data loading


def _load_yaml(name: str) -> dict[str, Any]:
    with (_DATA_DIR / name).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{name}: expected a mapping")
    return data


def _inflections(base: str) -> set[str]:
    forms = {base}
    forms.add(base + "es" if base.endswith(("ch", "sh", "s", "x")) else base + "s")
    if base.endswith("ee"):
        forms.add(base + "ing")
    elif base.endswith("e"):
        forms |= {base + "d", base[:-1] + "ing"}
    else:
        forms |= {base + "ed", base + "ing"}
    return forms | _IRREGULAR.get(base, set())


@dataclass(frozen=True)
class _Rules:
    generic_nouns: frozenset[str]
    verb_lemmas: frozenset[str]
    verb_forms: frozenset[str]
    blocklist: tuple[tuple[BlockCategory, tuple[re.Pattern[str], ...]], ...]
    time_re: re.Pattern[str]
    hedge_re: re.Pattern[str]
    alias_re: re.Pattern[str]
    alias_entries: Mapping[str, frozenset[str]]  # normalised alias -> lexicon entry ids
    entry_aliases: Mapping[str, tuple[str, ...]]  # entry id -> its normalised aliases
    entry_category: Mapping[str, str]


@functools.cache
def _rules() -> _Rules:
    ww = _load_yaml("walk_words.yaml")
    lex = _load_yaml("lexicon.yaml")
    hedges = _load_yaml("hedges.yaml")

    generic = frozenset(_norm(str(n)).lower() for n in ww["generic_nouns"])
    lemmas = frozenset(_norm(str(v)).lower() for v in ww["verb_allowlist"])
    forms = frozenset(f for v in lemmas for f in _inflections(v))

    blocklist: list[tuple[BlockCategory, tuple[re.Pattern[str], ...]]] = []
    for item in ww["blocklist"]:
        category = BlockCategory(item["id"])
        pats = tuple(
            re.compile(unicodedata.normalize("NFC", str(p)), re.IGNORECASE)
            for p in item["patterns"]
        )
        blocklist.append((category, pats))

    time_parts: list[str] = [_CLOCK_RE.pattern, _BARE_HOUR_RE]
    for raw in ww["time_of_day_words"]:
        word = _norm(str(raw)).lower()
        if word in ("am", "pm"):
            continue  # covered by _CLOCK_RE: only after a digit or as a.m./p.m.
        time_parts.append(re.escape(word) if _DEVANAGARI_RE.search(word) else _latin_term_re(word))
    time_re = re.compile("|".join(time_parts), re.IGNORECASE)

    hedge_words = sorted({_norm(str(h)).lower() for h in hedges["entry_hedge_words"]}, key=len)
    hedge_re = re.compile("|".join(_latin_term_re(h) for h in reversed(hedge_words)), re.IGNORECASE)

    alias_entries: dict[str, set[str]] = {}
    entry_aliases: dict[str, tuple[str, ...]] = {}
    entry_category: dict[str, str] = {}
    for entry in lex["entries"]:
        eid = str(entry["id"])
        aliases = tuple(_norm(str(a)).lower() for a in entry["aliases"])
        entry_aliases[eid] = aliases
        entry_category[eid] = str(entry["category"])
        for alias in aliases:
            alias_entries.setdefault(alias, set()).add(eid)
    ordered = sorted(alias_entries, key=len, reverse=True)  # longest (multi-word) first
    alias_re = re.compile(
        "|".join(re.escape(a) if _DEVANAGARI_RE.search(a) else _latin_term_re(a) for a in ordered),
        re.IGNORECASE,
    )
    return _Rules(
        generic_nouns=generic,
        verb_lemmas=lemmas,
        verb_forms=forms,
        blocklist=tuple(blocklist),
        time_re=time_re,
        hedge_re=hedge_re,
        alias_re=alias_re,
        alias_entries={k: frozenset(v) for k, v in alias_entries.items()},
        entry_aliases=entry_aliases,
        entry_category=entry_category,
    )


@functools.cache
def _nlp() -> Language:
    import spacy  # local import: loaded once, lazily, from the installed package

    return spacy.load(_SPACY_MODEL, exclude=["ner"])


# ------------------------------------------------------------------- small helpers


def mentions_time_of_day(text: str) -> bool:
    """FR-M6b: True if the text names a time of day or a clock time."""
    return _rules().time_re.search(_norm(text)) is not None


def _blocked(text: str) -> tuple[BlockCategory, ...]:
    norm = _norm(text)
    return tuple(cat for cat, pats in _rules().blocklist if any(p.search(norm) for p in pats))


def _lexicon_hits(text: str) -> list[tuple[int, int, frozenset[str]]]:
    rules = _rules()
    return [
        (m.start(), m.end(), rules.alias_entries[m.group(0).lower()])
        for m in rules.alias_re.finditer(text)
        if m.group(0).lower() in rules.alias_entries
    ]


def _entries_in(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        for _, _, ids in _lexicon_hits(_norm(t).lower()):
            out |= ids
    return out


def _alias_words(entry_ids: Iterable[str]) -> set[str]:
    rules = _rules()
    return {w for eid in entry_ids for a in rules.entry_aliases[eid] for w in _words(a)}


def _is_you(tok: Token) -> bool:
    return tok.lower_ == "you"


def _subjects(tok: Token) -> list[Token]:
    return [c for c in tok.children if c.dep_ in _SUBJ_DEPS]


def _has_you_subject(tok: Token) -> bool:
    return any(_is_you(s) for s in _subjects(tok))


def _is_past(tok: Token) -> bool:
    if tok.tag_ in _PAST_TAGS:
        return True
    return any(c.dep_ in ("aux", "auxpass") and c.tag_ in _PAST_TAGS for c in tok.children)


def _conjuncts(heads: list[Token]) -> list[Token]:
    out: list[Token] = []
    stack = list(heads)
    while stack:
        tok = stack.pop(0)
        if tok in out:
            continue
        out.append(tok)
        stack.extend(c for c in tok.children if c.dep_ in ("conj", "parataxis") and _verbish(c))
    return out


def _verbish(tok: Token) -> bool:
    """A verb, or a noun-tagged token used as one ("... and count the gates")."""
    if tok.pos_ == "VERB":
        return True
    if tok.pos_ not in ("NOUN", "PROPN"):
        return False
    return tok.lower_ in _rules().verb_forms or any(
        c.dep_ in ("dobj", "dative") for c in tok.children
    )


def _stray_imperatives(doc: Doc, task: list[Token]) -> list[Token]:
    """Bare-form conjuncts the parser hung under a subordinate verb ("the gates you pass
    and listen for birds"): no subject of their own, so they are further task verbs."""
    return [
        t
        for t in doc
        if t not in task
        and t.dep_ == "conj"
        and t.tag_ == "VB"
        and not _subjects(t)
        and any(a in task for a in t.ancestors)
    ]


@dataclass
class _Sentence:
    doc: Doc
    is_question: bool
    is_task: bool
    task_verbs: list[Token]
    opener_override: bool = False


def _classify(doc: Doc) -> _Sentence:
    text = doc.text.rstrip()
    roots = [t for t in doc if t.dep_ == "ROOT"] or [doc[0]]
    root = roots[0]
    if text.endswith("?"):
        you_verbs = [t for t in doc if t.pos_ != "AUX" and _has_you_subject(t)]
        heads = [t for t in you_verbs if not any(a in you_verbs for a in t.ancestors)]
        if not heads and root.pos_ == "VERB" and not _subjects(root):
            heads = [root]
        opener = doc[0]
        if opener not in heads and opener.tag_ == "VB" and not _subjects(opener):
            heads.insert(0, opener)  # "Listen closely, is that a bird near you?"
        verbs = _conjuncts(heads)
        return _Sentence(doc, True, True, verbs + _stray_imperatives(doc, verbs))
    subjects = _subjects(root)
    if subjects and (not any(_is_you(s) for s in subjects) or _is_past(root)):
        return _Sentence(doc, False, False, [])
    if subjects:
        # "you" + non-past. The small model sometimes hangs a leading imperative off a
        # later "you" clause ("Pause ... at one spot you usually pass"); if the sentence
        # opens with a content word before that subject, the opener is the task verb.
        first = doc[0]
        if (
            first.pos_ not in _NON_OPENER_POS
            and first.lower_ not in _STRUCTURAL_NOUNS | _WEEKDAYS
            and first.i < min(s.i for s in subjects)
        ):
            verbs = _conjuncts([first])
            return _Sentence(doc, False, True, verbs + _stray_imperatives(doc, verbs), True)
    verbs = _conjuncts(roots)
    return _Sentence(doc, False, True, verbs + _stray_imperatives(doc, verbs))


def _verb_allowed(tok: Token) -> bool:
    rules = _rules()
    return tok.lower_ in rules.verb_forms or tok.lemma_.lower() in rules.verb_lemmas


def _subordinate_actions(sent: _Sentence) -> list[Token]:
    """Verbs below the task whose doer is "you", plus open complements of task verbs."""
    task = set(sent.task_verbs)
    out: list[Token] = []
    for tok in sent.doc:
        if tok in task or tok.pos_ != "VERB":
            continue
        below_task = sent.opener_override or any(a in task for a in tok.ancestors)
        if (below_task and _has_you_subject(tok)) or (tok.dep_ == "xcomp" and tok.head in task):
            out.append(tok)
    return out


# --------------------------------------------------------------------- the checks


def _verb_and_task_codes(sentences: list[_Sentence]) -> set[ReasonCode]:
    codes: set[ReasonCode] = set()
    tasks = [s for s in sentences if s.is_task]
    if len(sentences) > MAX_SENTENCES or len(tasks) > 1:
        codes.add(ReasonCode.MULTIPLE_TASKS)
    if not tasks:
        codes.add(ReasonCode.VERB_NOT_ALLOWED)
    for sent in tasks:
        if not all(_verb_allowed(v) for v in sent.task_verbs):
            codes.add(ReasonCode.VERB_NOT_ALLOWED)
        for sub in _subordinate_actions(sent):
            if not (_verb_allowed(sub) or sub.lemma_.lower() in _BENIGN_SUBORDINATE):
                codes.add(ReasonCode.VERB_NOT_ALLOWED)
        lemmas = {v.lemma_.lower() if _verb_allowed(v) else v.lower_ for v in sent.task_verbs}
        main = lemmas - _PREAMBLE_VERBS if lemmas - _PREAMBLE_VERBS else lemmas
        if len(main) > 1:
            codes.add(ReasonCode.MULTIPLE_TASKS)
    return codes


def _noun_provenance_ok(sentences: list[_Sentence], facts: Sequence[VerifiedFact]) -> bool:
    rules = _rules()
    fact_texts = [t for f in facts for t in (f.subject, f.place) if t]
    referenced = _entries_in(fact_texts)
    allowed = set(rules.generic_nouns) | _STRUCTURAL_NOUNS | _WEEKDAYS
    for word in [w for t in fact_texts for w in _words(t)] + sorted(_alias_words(referenced)):
        allowed |= _variants(word)

    for sent in sentences:
        skip = set(sent.task_verbs)
        for tok in sent.doc:
            if tok in skip or tok.pos_ not in ("NOUN", "PROPN"):
                continue
            if tok.lower_ in rules.verb_forms:
                continue
            candidates = _variants(tok.lower_) | {tok.lemma_.lower()}
            if not candidates & allowed:
                return False
        text = sent.doc.text.lower()
        for start, end, ids in _lexicon_hits(text):
            if all(rules.entry_category[i] in _EXEMPT_LEXICON_CATEGORIES for i in ids):
                continue
            alias = text[start:end]
            if " " not in alias and not _DEVANAGARI_RE.search(alias):
                span = sent.doc.char_span(start, end, alignment_mode="expand")
                if span is not None and all(t.pos_ in _FUNCTION_POS for t in span):
                    continue  # an ambiguous alias used as a function word ("how well")
            if ids & referenced:
                continue
            if all(_variants(w) & allowed for w in _words(alias)):
                continue
            return False
    return True


def _unsure_ok(sentences: list[_Sentence], unsure: Sequence[VerifiedFact], text: str) -> bool:
    if not text.rstrip().endswith("?"):
        return False
    if not _YOU_RE.search(sentences[-1].doc.text):
        return False
    entries = _entries_in(f.subject for f in unsure)
    terms = {v for f in unsure for w in _words(f.subject) for v in _variants(w)}
    hedge_re = _rules().hedge_re
    for sent in sentences:
        if sent.is_question:
            continue
        stext = sent.doc.text.lower()
        mentions = bool(
            {w for w in _words(stext)} & terms
            or any(ids & entries for _, _, ids in _lexicon_hits(stext))
        )
        if mentions and not hedge_re.search(stext):
            return False
    return True


def _norm_for_repeat(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in _norm(text).lower()).split())


def _is_repeat(text: str, recent: Sequence[str]) -> bool:
    a = _norm_for_repeat(text)
    for other in recent:
        b = _norm_for_repeat(other)
        if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_SIMILARITY:
            return True
    return False


def _group_key(group: str) -> str:
    return " ".join(_norm(group).casefold().split())


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT_RE.split(text) if any(c.isalnum() for c in s)]


def _check(
    raw_text: str,
    *,
    mission_type: MissionType,
    references: Sequence[str],
    facts_by_id: Mapping[str, VerifiedFact],
    recent_missions: Sequence[str],
    provenance: bool,
    group: str | None = None,
) -> SafetyResult:
    text = _norm(raw_text)
    codes: set[ReasonCode] = set()
    parts = _split_sentences(text) or [text or "."]
    sentences = [_classify(doc) for doc in _nlp().pipe(parts)]

    codes |= _verb_and_task_codes(sentences)

    blocked = _blocked(text)
    if blocked:
        codes.add(ReasonCode.BLOCKLISTED)

    scope = _group_key(group) if group is not None else None
    known = [
        facts_by_id[r]
        for r in references
        if r in facts_by_id and (scope is None or _group_key(facts_by_id[r].group) == scope)
    ]
    if len(known) != len(references):
        codes.add(ReasonCode.REFERENCE_UNKNOWN)
    if mission_type is MissionType.RETURN and not references:
        codes.add(ReasonCode.RETURN_WITHOUT_REFERENCE)
    if provenance and not _noun_provenance_ok(sentences, known):
        codes.add(ReasonCode.NOUN_NOT_PROVENANCED)

    unsure = [f for f in known if f.certainty is Certainty.UNSURE]
    if unsure and not _unsure_ok(sentences, unsure, text):
        codes.add(ReasonCode.UNSURE_NOT_QUESTION)

    if not _YOU_RE.search(text) or _INDIVIDUAL_RE.search(text):
        codes.add(ReasonCode.NOT_ADDRESSED_TO_GROUP)
    if mentions_time_of_day(text):
        codes.add(ReasonCode.TIME_OF_DAY)
    if len(_words(text)) > MAX_MISSION_WORDS:
        codes.add(ReasonCode.MISSION_TOO_LONG)
    if recent_missions and _is_repeat(text, recent_missions):
        codes.add(ReasonCode.MISSION_REPEATED)

    reasons = tuple(c for c in _REASON_ORDER if c in codes)
    return SafetyResult(passed=not reasons, reasons=reasons, block_categories=blocked)


# ----------------------------------------------------------------------- public API


def check_mission(
    plan: MissionPlan,
    facts_by_id: Mapping[str, VerifiedFact],
    *,
    recent_missions: Sequence[str] = (),
    provenance: bool = True,
    group: str | None = None,
) -> SafetyResult:
    """Run every FR-M1/M4/M5/M6/M6b rule on a planner output.

    ``facts_by_id`` holds the verified facts the planner was given (the in-scope set: the
    group's facts from its last 7 walks, FR-M3, as returned by
    ``hub.db.verified_facts_for_group``); only those in ``plan.references`` supply nouns.
    When ``group`` is given, a referenced fact of another walker group is out of scope:
    ``REFERENCE_UNKNOWN``, and it supplies no nouns (FR-M2). ``provenance=False`` is for
    PRD §13.4 headline contamination runs only and is never passed in production.
    """
    return _check(
        plan.text,
        mission_type=plan.type,
        references=plan.references,
        facts_by_id=facts_by_id,
        recent_missions=recent_missions,
        provenance=provenance,
        group=group,
    )


def check_fallback_text(text: str) -> SafetyResult:
    """Same rules for hand-written fallback and seed missions: no references, so every
    concrete noun must be a generic walk word. No repeat check."""
    return _check(
        text,
        mission_type=MissionType.SEED,
        references=(),
        facts_by_id={},
        recent_missions=(),
        provenance=True,
    )
