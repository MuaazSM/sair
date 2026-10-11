"""Generate synthetic voice-note transcripts with gold labels (PRD §12.1, synthetic only).

For every seed in ``data/seeds.yaml`` Gemma (Ollama ``/api/chat``, ``think:false``,
``format`` = JSON schema) writes ~15-20 transcripts. The gold *labels* (kind, subject,
certainty, place) come from the seed; the model only supplies the transcript, one exact
span per fact it says, and the mission span. Every item is then validated with the same
code the hub uses (``hub.grounding``). Failures are dropped and counted, never repaired.

Output text is synthetic; it is still never sent anywhere but the loopback Ollama port.

Usage::

    uv run python -m training.gen_transcripts --seeds s001,s002,s003 [--variants 15-20] [--force]
    uv run python -m training.gen_transcripts --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from hub.grounding import (
    Lexicon,
    _find_bounded,
    _has_cue,
    build_template_entry,
    check_entry,
    extract_details,
    load_lexicon,
    normalise,
)
from hub.schemas import (
    Certainty,
    EntryOrigin,
    Fact,
    MissionResult,
    WriterOutput,
)
from training.common import (
    CACHE_DIR,
    ITEMS_PATH,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    Seed,
    load_seeds,
    read_jsonl,
    write_jsonl,
)

LEAK_TOKENS = ("mission_span", "mission:", "item 0", "item 1", "span:", "transcript", "{", "}")
NOT_AT_END = ("looked like", "sounded like")
TOPUP_START = 30  # variant indices >= this are top-up variants (length-leak-balanced)
R_SCRIPT = "script_mismatch"
POSITIONS = ("start", "middle", "end")
HARD_PERCENT = 20
DETAIL_REQUESTS: dict[str, list[str]] = {
    "english": [
        "write that it was a beautiful sunny day",
        "add the colour of the flowers",
        "say how many birds there were",
    ],
    "hinglish": [
        "likh dena ki bahut dhoop khili thi",
        "phoolon ka rang bhi likh dena",
        "likh dena ki kitne pakshi the",
    ],
    "devanagari_heavy": [
        "लिख देना कि बहुत धूप खिली थी",
        "फूलों का रंग भी लिख देना",
        "लिख देना कि कितने पक्षी थे",
    ],
}
DETAIL_REQUESTS["mixed"] = [DETAIL_REQUESTS[k][i] for i, k in enumerate(DETAIL_REQUESTS)]
MAX_WORDS = 120
MIN_WORDS = 10
MIN_WORDS_VERY_SHORT = 3
DEDUPE_THRESHOLD = 0.9
TEMPERATURE = 0.8
NUM_PREDICT = 700

CUES: dict[str, list[str]] = {
    "english": ["maybe", "I think", "probably", "looked like", "sounded like"],
    "hinglish": ["shayad", "lagta hai", "pata nahi", "ho sakta hai"],
    "devanagari_heavy": ["शायद", "लगता है", "पता नहीं", "हो सकता है"],
}
CUES["mixed"] = CUES["english"][:3] + CUES["hinglish"][:2] + CUES["devanagari_heavy"][:2]

LANG_TEXT = {
    "english": "plain, natural spoken English.",
    "hinglish": (
        "Hinglish: Hindi and English mixed, written in Latin (Roman) script, "
        "like 'aaj hawa kaafi thandi thi, bahut achha laga'. No Devanagari."
    ),
    "devanagari_heavy": (
        "mostly Hindi written in Devanagari script, with a few English words mixed in "
        "(for example 'walk', 'mission')."
    ),
    "mixed": (
        "a mix: some English sentences, some Hindi in Roman script, and a few phrases "
        "in Devanagari."
    ),
}

OUTCOME_TEXT = {
    "done": "They DID the mission and say so in one short phrase.",
    "partly": "They only PARTLY did the mission and say so in one short phrase.",
    "not_done": "They did NOT do the mission (forgot, or no time) and say so in one short phrase.",
    "not_mentioned": "They never mention the mission at all.",
}

# Drop reasons that come from the grounding code are reported with their enum value.
R_PARSE = "parse_error"
R_MISSING_SPAN = "missing_span"
R_WORDS = "word_count"
R_HEDGE_MISMATCH = "hedge_label_mismatch"
R_UNSURE_NO_CUE = "unsure_without_cue"
R_MISSION_SPAN = "mission_span_not_in_transcript"
R_NAME = "likely_proper_noun"
R_LURE = "lure_leaked"
R_OMITTED = "omitted_item_leaked"
R_DUP = "near_duplicate"
R_SELF_HEDGE = "model_added_hedge"
R_HEDGE_POS = "hedge_position_mismatch"
R_NUMBERS = "numbers_missing"
R_DETAIL = "requested_detail_stated"
R_PROMPT_LEAK = "prompt_leak"
R_TEMPLATE = "template_check_failed"

SPAN_REASONS = {
    "span_not_in_transcript": "span_not_in_transcript",
    "subject_not_in_span": "subject_not_in_span",
    "place_not_in_span": "place_not_in_span",
}


# --------------------------------------------------------------------------- variants


@dataclass
class Variant:
    seed_id: str
    idx: int
    included: list[int]  # indexes into seed["noticed"] said in this transcript
    style: list[str]
    two_speaker_labels: bool
    hedge_cue: dict[int, tuple[str, str]] = field(default_factory=dict)  # item -> (cue, position)
    target_words: tuple[int, int] = (20, 60)
    number_item: int | None = None  # seed noticed-index whose segment must state a count
    hard: bool = False
    hard_kind: str = ""  # "detail" | "bait"
    topup: str = ""  # "" | "long_few" | "short_many" (top-up variants only)
    hard_text: str = ""  # the unsaid detail requested (detail) or the lure hinted at (bait)


def target_position(seed_id: str, idx: int, item_idx: int) -> str:
    """Hedge position target, cycling start/middle/end deterministically."""
    return POSITIONS[(zlib.crc32(seed_id.encode()) + idx + item_idx) % 3]


def item_seed(seed_id: str, idx: int) -> int:
    return zlib.crc32(f"{seed_id}:{idx}".encode()) & 0x7FFFFFFF


def make_variant(seed: Seed, idx: int) -> Variant:
    rng = random.Random(item_seed(seed["id"], idx))
    base = list(seed.get("style") or [])
    noticed = seed.get("noticed") or []
    included = [i for i in range(len(noticed)) if rng.random() < 0.85]
    if noticed and not included:
        included = [rng.randrange(len(noticed))]

    topup = idx >= TOPUP_START
    topup_mode = ""
    if topup:
        # Break the length/fact-count correlation: long notes with 0-2 facts, short notes
        # with 3+ facts (only possible when the seed has 3+ noticed items).
        base = [t for t in base if t not in ("very_short", "long")]
        if len(noticed) >= 3 and rng.random() < 0.5:
            topup_mode = "short_many"
            k = rng.randint(3, len(noticed))
            included = sorted(rng.sample(range(len(noticed)), k))
            base.append("very_short")
        else:
            topup_mode = "long_few"
            k = rng.randint(0, min(2, len(noticed)))
            included = sorted(rng.sample(range(len(noticed)), k))
            base += ["long", "off_topic"]
    very_short = "very_short" in base
    style: list[str] = []
    for tag in base:
        if (
            tag in ("very_short", "long", "numbers", "request_detail", "hedges")
            or rng.random() < 0.85
        ):
            style.append(tag)
    if not very_short:
        for tag in ("filler", "false_start", "off_topic"):
            if tag not in style and rng.random() < 0.12:
                style.append(tag)
        if "two_speakers" not in style and rng.random() < 0.10:
            style.append("two_speakers")
    if any(noticed[i]["certainty"] == "unsure" for i in included) and "hedges" not in style:
        style.append("hedges")

    mix = seed["language_mix"]
    hedge_cue: dict[int, tuple[str, str]] = {}
    for i in included:
        if noticed[i]["certainty"] == "unsure":
            pos = target_position(seed["id"], idx, i)
            pool = [c for c in CUES[mix] if pos != "end" or c not in NOT_AT_END]
            hedge_cue[i] = (rng.choice(pool), pos)
    style = [t for t in style if t != "request_detail"]  # replaced by the hard mechanism
    number_item: int | None = None
    if "numbers" in style:
        if included:
            number_item = included[rng.randrange(len(included))]
        else:
            style.remove("numbers")
    hard = zlib.crc32(f"hard:{seed['id']}:{idx}".encode()) % 100 < HARD_PERCENT
    hard_kind, hard_text = "", ""
    if hard:
        lures = [x for x in (seed.get("lures") or []) if x]
        if lures and rng.random() < 0.5:
            hard_kind, hard_text = "bait", rng.choice(lures)
        else:
            hard_kind, hard_text = "detail", rng.choice(DETAIL_REQUESTS[mix])
    if topup_mode == "short_many":
        words = (10, 26)
    elif very_short:
        words = (3, 15)
    elif "long" in style:
        words = (60, 110)
    else:
        words = (20, 60)
    return Variant(
        seed_id=seed["id"],
        idx=idx,
        included=included,
        style=style,
        two_speaker_labels=rng.random() < 0.6,
        hedge_cue=hedge_cue,
        target_words=words,
        number_item=number_item,
        hard=hard,
        hard_kind=hard_kind,
        hard_text=hard_text,
        topup=topup_mode,
    )


def variant_count(seed_id: str, lo: int, hi: int) -> int:
    return lo + zlib.crc32(f"n:{seed_id}".encode()) % (hi - lo + 1)


# --------------------------------------------------------------------------- prompt


POSITION_TEXT = {
    "start": (
        'we will add the hedge "{cue}" at the very START of this sentence ourselves, so write '
        "the sentence WITHOUT any hedge word (4-8 words), as a sentence that reads naturally "
        "after it"
    ),
    "middle": (
        'we will insert the hedge "{cue}" in the MIDDLE of this sentence ourselves, so write '
        "it WITHOUT any hedge word as TWO parts of 2-4 words each and similar length: role "
        "item_{n}_a (starts the sentence and contains the thing's name) and role item_{n}_b "
        '(the rest), for example a = "the bird", b = "was flying high"'
    ),
    "end": (
        'we will add the hedge "{cue}" at the very END of this sentence ourselves, so write '
        "the sentence WITHOUT any hedge word (4-8 words), as a sentence that reads naturally "
        "before it"
    ),
}

STYLE_TEXT = {
    "filler": 'Include spoken fillers such as "umm", "achha", "haan" or "you know".',
    "false_start": (
        'Include one false start or self-correction (e.g. "we saw, no wait, I mean...").'
    ),
    "off_topic": (
        "Include one short off-topic remark (about dinner, the neighbour's wedding, a "
        "relative's visit). No names."
    ),
    "numbers": (
        "Include a number said as a word or digit inside one of the item sentences, "
        "if it fits naturally."
    ),
    "request_detail": (
        "At the end, ask the diary to add a nice detail that was never described, for "
        'example "write that it was a lovely sunny evening" or "put that it was very peaceful".'
    ),
}


def _alias_index(lex: Lexicon) -> dict[str, tuple[list[str], list[str]]]:
    """Lexicon id -> (latin aliases, devanagari aliases)."""
    idx: dict[str, tuple[list[str], list[str]]] = {}
    for key, eid in lex.latin.items():
        idx.setdefault(eid, ([], []))[0].append(key)
    for alias, eid in lex.deva:
        idx.setdefault(eid, ([], []))[1].append(alias)
    return idx


def name_hints(subject: str, mix: str, lex: Lexicon | None = None) -> list[str]:
    """Words the model may use for a subject in this language mix (from the frozen lexicon)."""
    lex = lex or load_lexicon()
    ids = sorted(extract_details(subject, lex, inclusive=True) & set(lex.categories))
    index = _alias_index(lex)
    hints: list[str] = []
    for eid in ids:
        latin, deva = index.get(eid, ([], []))
        if mix == "english":
            continue
        if mix == "devanagari_heavy":
            hints += deva[:3]
        elif mix == "hinglish":
            hints += latin[:4]
        else:
            hints += latin[:2] + deva[:2]
    return list(dict.fromkeys(hints))


def build_prompt(seed: Seed, v: Variant) -> str:
    noticed = seed.get("noticed") or []
    group = seed["group"]
    who = "I" if group in ("Mom", "Dad") else "we"
    speaker = group if group in ("Mom", "Dad") else "Mom and Dad"
    lines: list[str] = [
        f"You write realistic SYNTHETIC voice-note transcripts for a training set. {speaker} "
        f"just came back from an evening walk and dictate a short diary note. Speak as "
        f'"{who}". Write ONE transcript.',
        "",
        f"Setting: {seed['place_type']}, {seed['season']}.",
        f"Language: {LANG_TEXT[seed['language_mix']]}",
        f"Length: between {v.target_words[0]} and {v.target_words[1]} words.",
    ]
    if "two_speakers" in v.style:
        if v.two_speaker_labels:
            lines.append('Two speakers talk in turns, each turn starting with "Dad:" or "Mom:".')
        else:
            lines.append("Two speakers talk in turns, but write NO speaker labels.")
    lines.append("")
    if v.included:
        lines.append(
            "Things they noticed. Each one gets its OWN segment (role item_0, item_1, ...): one "
            "short sentence that contains the thing's name (in English or the Hindi word for it):"
        )
        for n, i in enumerate(v.included):
            item = noticed[i]
            place = f", {item['place']}" if item.get("place") else ""
            role = f"item_{n}"
            if item["certainty"] == "unsure":
                cue, pos = v.hedge_cue[i]
                how = "they are NOT sure about it; " + POSITION_TEXT[pos].format(cue=cue, n=n)
                if pos == "middle":
                    role = f"item_{n}_a + item_{n}_b"
            else:
                how = (
                    "they are sure: state it plainly, with NO hedge words (no maybe, I think, "
                    "shayad, lagta, kuch, jaisa, jaise)"
                )
            if i == v.number_item:
                how += (
                    ". This sentence MUST state a count of it, e.g. 'teen kauve', "
                    "'two dogs', 'पाँच तोते' (in the note's language)"
                )
            hints = name_hints(item["subject"], seed["language_mix"])
            say = f" Words you may use for it: {', '.join(hints)}." if hints else ""
            lines.append(f'  role {role}: "{item["subject"]}" ({item["kind"]}{place}); {how}.{say}')
        lines.append(
            "Do NOT mention any other bird, animal, plant, sound, smell, weather, person "
            "or object that they noticed. Never invent names of people or places."
        )
    else:
        lines.append(
            "They noticed nothing in particular ('nothing special'). Do not mention any "
            "specific bird, animal, plant, sound, smell, weather or object."
        )
    lures = [x for x in (seed.get("lures") or []) if x]
    if lures:
        lines.append("Never mention these (they were not seen): " + "; ".join(lures) + ".")
    lines.append("")
    lines.append(f'Today\'s mission was: "{seed["mission"]["text"]}"')
    lines.append(OUTCOME_TEXT[seed["mission_outcome"]])
    for tag in v.style:
        if tag in STYLE_TEXT:
            lines.append(STYLE_TEXT[tag])
    if "very_short" in v.style:
        lines.append('It is a very short, low-effort note (for example "bas, kuch khaas nahi").')
    lines.append(
        "Start directly with the walk. Do not greet anyone, call anyone by name or say who is back."
    )
    if v.hard and v.hard_kind == "detail":
        lines.append(
            'Add one extra "other" segment where they ask the diary to add a detail they never '
            f'described, for example: "{v.hard_text}". Do not describe that detail yourself.'
        )
    elif v.hard and v.hard_kind == "bait":
        lines.append(
            f'Add one extra "other" segment that hints at "{v.hard_text}" without naming or '
            'describing it, for example "pata nahi uska colour kya tha" or "I forgot how it '
            'looked".'
        )
    lines.append(
        "Use no real person names, no real place names, no brand names. Only the words "
        '"Mom" and "Dad" may be capitalised mid-sentence.'
    )
    lines.append("")
    lines.append(
        f"IMPORTANT: the whole transcript must be written in {LANG_TEXT[seed['language_mix']]}"
    )
    mission_note = (
        'One segment with role "mission" says what happened with the mission, in your own '
        "short words (never the mission text itself)."
        if seed["mission_outcome"] != "not_mentioned"
        else 'Use no "mission" segment.'
    )
    lines.append(
        "Write the note as a list of segments in speaking order. Roles: item_N (see above), "
        '"mission", and "other" for everything else (greetings, fillers, off-topic). '
        f"{mission_note}"
    )
    lines.append(
        'Reply as JSON: {"segments": [{"role": "other", "text": "..."}, ...]}. '
        "Use each item role exactly once."
    )
    return "\n".join(lines)


def item_roles(seed: Seed, v: Variant) -> list[str]:
    roles: list[str] = []
    for n, i in enumerate(v.included):
        if i in v.hedge_cue and v.hedge_cue[i][1] == "middle":
            roles += [f"item_{n}_a", f"item_{n}_b"]
        else:
            roles.append(f"item_{n}")
    roles.append("other")
    if seed["mission_outcome"] != "not_mentioned":
        roles.append("mission")
    return roles


def item_schema(roles: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "minItems": 1,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": roles},
                        "text": {"type": "string"},
                    },
                    "required": ["role", "text"],
                },
            }
        },
        "required": ["segments"],
    }


# --------------------------------------------------------------------------- ollama


def call_ollama(
    prompt: str, seed_int: int, roles: list[str], client: httpx.Client
) -> dict[str, Any]:
    body = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": item_schema(roles),
        "options": {
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "seed": seed_int,
            "num_predict": NUM_PREDICT,
            "num_ctx": 4096,
        },
    }
    r = client.post(f"{OLLAMA_HOST}/api/chat", json=body)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data


# --------------------------------------------------------------------------- validation


_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’]*")
_SENT_END = set('.!?…।\n:"“‘(—-')
_NAME_OK = {"i", "mom", "dad"}


def _lex_tokens(lex: Lexicon) -> set[str]:
    return {t for key in lex.latin for t in key.split()}


def likely_names(text: str, lex: Lexicon | None = None) -> list[str]:
    """Capitalised Latin tokens that are not sentence-initial, Mom/Dad/I or lexicon words."""
    lex = lex or load_lexicon()
    known = _lex_tokens(lex)
    found: list[str] = []
    for m in _NAME_TOKEN.finditer(text):
        tok = m.group()
        if not tok[0].isupper() or tok.casefold() in _NAME_OK:
            continue
        j = m.start() - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        if j < 0 or text[j] in _SENT_END:
            continue
        if tok.casefold().replace("’", "'") in known:
            continue
        found.append(tok)
    return found


def word_count(text: str) -> int:
    return len(text.split())


def hedge_position(span: str, lex: Lexicon | None = None) -> str | None:
    """Midpoint of the first hedge cue / span length: <0.33 start, <=0.66 middle, else end."""
    lex = lex or load_lexicon()
    s = normalise(span)
    best: tuple[int, int] | None = None
    for cue in list(lex.hedge_latin) + list(lex.hedge_deva):
        for m in re.finditer(r"(?<![\w])" + re.escape(cue) + r"(?![\w])", s):
            if best is None or m.start() < best[0]:
                best = (m.start(), m.end())
            break
    if best is None:
        return None
    frac = (best[0] + best[1]) / 2 / max(len(s), 1)
    if frac < 0.33:
        return "start"
    return "middle" if frac <= 0.66 else "end"


_DEVA_LETTER = re.compile(r"[\u0900-\u097F]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_LABEL = re.compile(r"^\s*(?:Mom|Dad)\s*:\s*")
_ROLE = re.compile(r"item_(\d+)(?:_([ab]))?")


def _cue_sets() -> tuple[frozenset[str], tuple[str, ...]]:
    lex = load_lexicon()
    return lex.hedge_latin, lex.hedge_deva


def assemble(seed: Seed, v: Variant, parsed: dict[str, Any]) -> dict[str, Any] | str:
    """Build the gold item from the model's parsed JSON, or return a drop reason.

    The model writes role-tagged segments; the transcript is their concatenation and each
    fact/mission span is the text of its segment (minus any "Mom:"/"Dad:" label).
    """
    segs = parsed.get("segments")
    if not isinstance(segs, list):
        return R_PARSE
    parts: dict[tuple[int, str], str] = {}  # (item, "" | "a" | "b") -> label-free text
    labels: dict[int, str] = {}
    first_at: dict[int, int] = {}
    mission_span: str | None = None
    pieces: list[tuple[str, int | None, str]] = []  # (kind, item, text) in speaking order
    for seg in segs:
        if not (isinstance(seg, dict) and isinstance(seg.get("text"), str)):
            return R_PARSE
        text = seg["text"].strip()
        if not text:
            continue
        role = seg.get("role")
        m = _ROLE.fullmatch(role) if isinstance(role, str) else None
        label_m = _LABEL.match(text)
        body = _LABEL.sub("", text).strip()
        if m:
            n, part = int(m.group(1)), m.group(2) or ""
            if (n, part) in parts:
                continue
            parts[(n, part)] = body
            labels.setdefault(n, label_m.group() if label_m else "")
            if n not in first_at:
                first_at[n] = len(pieces)
                pieces.append(("item", n, ""))
        elif role == "mission" and mission_span is None:
            mission_span = body
            pieces.append(("text", None, text))
        else:
            pieces.append(("text", None, text))
    if any(_has_cue(normalise(t), *_cue_sets()) for t in parts.values()):
        return R_SELF_HEDGE
    spans: dict[int, str] = {}
    for n, i in enumerate(v.included):
        hc = v.hedge_cue.get(i)
        if hc is None:
            span = parts.get((n, ""), "")
        else:
            cue, pos = hc
            if pos == "middle":
                a_, b_ = parts.get((n, "a"), ""), parts.get((n, "b"), "")
                span = f"{a_.rstrip()} {cue} {b_.lstrip()}" if a_ and b_ else ""
            else:
                t = parts.get((n, ""), "")
                tail = t[-1] if t and t[-1] in ".!?।" else ""
                core = t.rstrip(".!?। ")
                if not core:
                    span = ""
                elif pos == "start":
                    span = f"{cue} {core}{tail}"
                else:
                    span = f"{core} {cue}{tail}"
        spans[n] = span
    texts: list[str] = []
    for kind, pn, text in pieces:
        if kind == "text":
            texts.append(text)
        elif pn is not None and spans.get(pn):
            texts.append(f"{labels.get(pn, '')}{spans[pn]}".strip())
    by_item = spans
    transcript = ("\n" if v.two_speaker_labels and "two_speakers" in v.style else " ").join(texts)
    noticed = seed.get("noticed") or []
    facts: list[dict[str, Any]] = []
    for n, i in enumerate(v.included):
        span = by_item.get(n, "")
        if not span:
            return R_MISSING_SPAN
        it = noticed[i]
        facts.append(
            {
                "span": span,
                "kind": it["kind"],
                "subject": it["subject"],
                "certainty": it["certainty"],
                "place": it.get("place"),
            }
        )
    result = seed["mission_outcome"]
    m_span: str | None = None
    if result != "not_mentioned":
        m_span = mission_span or ""
        if not m_span:
            return R_MISSION_SPAN
    return {
        "item_id": f"{seed['id']}-{v.idx:02d}",
        "seed_id": seed["id"],
        "group": seed["group"],
        "mission_text": seed["mission"]["text"],
        "transcript": transcript.strip(),
        "gold": {"facts": facts, "mission_result": result, "mission_span": m_span},
        "meta": {
            "language_mix": seed["language_mix"],
            "style": list(v.style),
            "hard": False,
            "lures": list(seed.get("lures") or []),
            "hedge_positions": [],
            "gen": {
                "hedge_targets": [v.hedge_cue[i][1] for i in v.included if i in v.hedge_cue],
                "number_fact": v.included.index(v.number_item)
                if v.number_item is not None
                else None,
                "topup": v.topup,
                "hard_kind": v.hard_kind,
                "hard_text": v.hard_text,
            },
        },
    }


def _lure_words(lure: str) -> list[str]:
    lure = lure.strip()
    if not lure or len(lure.split()) > 2 or lure.split()[0].lower() in {"any", "a", "an", "the"}:
        return []
    return [lure]


def validate_item(
    item: dict[str, Any], seed: Seed, v: Variant | None = None, lex: Lexicon | None = None
) -> str | None:
    """Return None if the item is valid, else a drop reason code. Never repairs."""
    lex = lex or load_lexicon()
    transcript: str = item["transcript"]
    gold = item["gold"]
    t_norm = normalise(transcript)

    n_words = word_count(transcript)
    short_ok = "very_short" in item["meta"]["style"]
    if n_words > MAX_WORDS or n_words < (MIN_WORDS_VERY_SHORT if short_ok else MIN_WORDS):
        return R_WORDS

    if likely_names(transcript, lex):
        return R_NAME
    low = transcript.casefold()
    if any(tok in low for tok in LEAK_TOKENS) or normalise(item["mission_text"] or "") in t_norm:
        return R_PROMPT_LEAK

    facts = [Fact.model_validate(f) for f in gold["facts"]]
    for f in facts:
        span_n = normalise(f.span)
        hedged = _has_cue(span_n, lex.hedge_latin, lex.hedge_deva)
        if hedged and f.certainty is Certainty.SURE:
            return R_HEDGE_MISMATCH
        if f.certainty is Certainty.UNSURE and not hedged:
            return R_UNSURE_NO_CUE

    gen = item["meta"].get("gen") or {}
    if gen.get("topup") and item["meta"]["language_mix"] == "devanagari_heavy":
        deva = len(_DEVA_LETTER.findall(transcript))
        latin = len(_LATIN_LETTER.findall(transcript))
        if deva + latin == 0 or deva / (deva + latin) < 0.5:
            return R_SCRIPT
    unsure = [f for f in facts if f.certainty is Certainty.UNSURE]
    for f, target in zip(unsure, gen.get("hedge_targets") or [], strict=False):
        if hedge_position(f.span, lex) != target:
            return R_HEDGE_POS
    nf = gen.get("number_fact")
    if nf is not None:
        keys = extract_details(facts[nf].span, lex, inclusive=True)
        if not (keys & set(lex.id_values) or any(k.startswith("num:") for k in keys)):
            return R_NUMBERS
    if gen.get("hard_kind") == "detail":
        req = extract_details(gen.get("hard_text", ""), lex, inclusive=True)
        for f in facts:
            stated = extract_details(f.span, lex, inclusive=True) - extract_details(
                f.subject, lex, inclusive=True
            )
            if stated & req:
                return R_DETAIL

    result = MissionResult(gold["mission_result"])
    m_span = gold["mission_span"]
    if result is not MissionResult.NOT_MENTIONED and not _find_bounded(
        t_norm, normalise(m_span or "")
    ):
        return R_MISSION_SPAN

    # Lures must not be said; omitted seed items must not leak into the transcript.
    for lure in item["meta"].get("lures", []):
        for w in _lure_words(lure):
            if _find_bounded(t_norm, normalise(w)):
                return R_LURE
    if v is not None:
        said = extract_details(transcript, lex, inclusive=True)
        noticed = seed.get("noticed") or []
        kept_keys: set[str] = set()
        for i in v.included:
            kept_keys |= extract_details(noticed[i]["subject"], lex, inclusive=True)
        for i in range(len(noticed)):
            if i in v.included:
                continue
            gone = extract_details(noticed[i]["subject"], lex, inclusive=True) - kept_keys
            if gone and gone & said:
                return R_OMITTED

    # Grounding: spans in transcript, subject/place in span (checks 1-3), then a
    # template entry built from the gold facts must pass checks 4-5 (and 7).
    group = item["group"]
    template = build_template_entry(group, facts, result is MissionResult.DONE)
    out = WriterOutput(facts=facts, mission_result=result, mission_span=m_span, entry=template)
    check = check_entry(out, transcript, group, origin=EntryOrigin.TEMPLATE, lex=lex)
    if check.dropped_facts:
        for code in check.reasons:
            if code.value in SPAN_REASONS:
                return SPAN_REASONS[code.value]
            if code.value == "hedge_certainty_mismatch":
                return R_HEDGE_MISMATCH
        return R_TEMPLATE
    if not check.passed or check.mission_result is not result:
        return R_TEMPLATE
    return None


def finalise_meta(item: dict[str, Any], seed: Seed, lex: Lexicon | None = None) -> None:
    """Fill hedge positions and the hard flag."""
    lex = lex or load_lexicon()
    meta = item["meta"]
    meta["hedge_positions"] = [
        p
        for f in item["gold"]["facts"]
        if f["certainty"] == "unsure" and (p := hedge_position(f["span"], lex))
    ]
    meta["hard"] = bool(meta["gen"]["hard_kind"])


def similarity(a: str, b: str) -> float:
    na, nb = normalise(a), normalise(b)
    sm = SequenceMatcher(None, na, nb, autojunk=False)
    if sm.real_quick_ratio() < DEDUPE_THRESHOLD or sm.quick_ratio() < DEDUPE_THRESHOLD:
        return 0.0
    return sm.ratio()


def dedupe(items: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the first of any near-identical pair. Returns (kept, dropped item ids)."""
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for it in items:
        if any(similarity(it["transcript"], k["transcript"]) >= DEDUPE_THRESHOLD for k in kept):
            dropped.append(it["item_id"])
        else:
            kept.append(it)
    return kept, dropped


# --------------------------------------------------------------------------- pipeline


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def generate_one(
    seed: Seed, v: Variant, client: httpx.Client, *, force: bool, cache_root: Path = CACHE_DIR
) -> tuple[dict[str, Any], float]:
    """Return (cache record, seconds spent calling the model; 0.0 on a cache hit)."""
    path = cache_root / seed["id"] / f"{v.idx}.json"
    prompt = build_prompt(seed, v)
    sha = _sha(prompt + OLLAMA_MODEL + str(TEMPERATURE))
    if path.exists() and not force:
        rec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("prompt_sha") == sha:
            return rec, 0.0
    t0 = time.monotonic()
    try:
        resp = call_ollama(
            prompt,
            item_seed(seed["id"], v.idx),
            item_roles(seed, v),
            client,
        )
        raw = resp.get("message", {}).get("content", "")
        done = resp.get("done_reason")
    except httpx.HTTPError as exc:
        raw, done = "", f"http_error:{type(exc).__name__}"
    dt = time.monotonic() - t0
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    rec = {
        "prompt_sha": sha,
        "variant": asdict(v) | {"hedge_cue": {str(k): list(x) for k, x in v.hedge_cue.items()}},
        "raw": raw,
        "done_reason": done,
        "parsed": parsed,
        "seconds": round(dt, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec, dt


def process_seed(
    seed: Seed,
    lo: int,
    hi: int,
    client: httpx.Client,
    *,
    force: bool,
    lex: Lexicon,
    cache_root: Path = CACHE_DIR,
    indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], list[float]]:
    drops: Counter[str] = Counter()
    secs: list[float] = []
    cand: list[dict[str, Any]] = []
    idxs = list(indices) if indices is not None else list(range(variant_count(seed["id"], lo, hi)))
    n = len(idxs)
    for idx in idxs:
        v = make_variant(seed, idx)
        rec, dt = generate_one(seed, v, client, force=force, cache_root=cache_root)
        if dt:
            secs.append(dt)
        if not isinstance(rec["parsed"], dict):
            drops[R_PARSE] += 1
            continue
        built = assemble(seed, v, rec["parsed"])
        if isinstance(built, str):
            drops[built] += 1
            continue
        reason = validate_item(built, seed, v, lex)
        if reason:
            drops[reason] += 1
            continue
        finalise_meta(built, seed, lex)
        cand.append(built)
    kept, dup = dedupe(cand)
    drops[R_DUP] += len(dup)
    drops["_generated"] = n
    drops["_kept"] = len(kept)
    return kept, drops, secs


def merge_topup(items_path: Path, new_items: Sequence[dict[str, Any]]) -> int:
    """Append top-up items to items.jsonl (backup once as items.base.jsonl); dedupe per seed."""
    base = read_jsonl(items_path)
    backup = items_path.with_name("items.base.jsonl")
    if not backup.exists():
        write_jsonl(backup, base)
    known = {i["item_id"] for i in base}
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in [*base, *(n for n in new_items if n["item_id"] not in known)]:
        by_seed[it["seed_id"]].append(it)
    out: list[dict[str, Any]] = []
    for seed_id in dict.fromkeys(i["seed_id"] for i in [*base, *new_items]):
        kept, _ = dedupe(by_seed[seed_id])
        out.extend(kept)
    return write_jsonl(items_path, out)


def parse_range(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", help="comma-separated seed ids")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--variants", default="15-20")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--out", type=Path, default=ITEMS_PATH)
    ap.add_argument(
        "--topup",
        type=int,
        default=0,
        help=f"generate N NEW variants per seed at indices {TOPUP_START}.. (cache kept)",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="with --topup: merge the new items into items.jsonl (backup: items.base.jsonl)",
    )
    args = ap.parse_args(argv)
    if args.topup and args.out == ITEMS_PATH:
        args.out = ITEMS_PATH.with_name("items.topup.jsonl")

    seeds = load_seeds()
    if args.seeds:
        wanted = [s.strip() for s in args.seeds.split(",") if s.strip()]
        by_id = {s["id"]: s for s in seeds}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            print("unknown seed ids:", ",".join(missing), file=sys.stderr)
            return 2
        seeds = [by_id[w] for w in wanted]
    lo, hi = parse_range(args.variants)
    lex = load_lexicon()
    all_items: list[dict[str, Any]] = []
    table: dict[str, Counter[str]] = {}
    all_secs: list[float] = []
    with httpx.Client(timeout=httpx.Timeout(180.0), trust_env=False) as client:
        for seed in seeds:
            kept, drops, secs = process_seed(
                seed,
                lo,
                hi,
                client,
                force=args.force,
                lex=lex,
                indices=range(TOPUP_START, TOPUP_START + args.topup) if args.topup else None,
            )
            all_items.extend(kept)
            table[seed["id"]] = drops
            all_secs.extend(secs)
            print(
                f"{seed['id']}: generated={drops['_generated']} kept={drops['_kept']} "
                f"calls={len(secs)}",
                flush=True,
            )
    n_written = write_jsonl(args.out, all_items)
    print(f"\nwrote {n_written} items to {args.out}")
    if args.topup and args.merge:
        merged = merge_topup(ITEMS_PATH, all_items)
        print(f"merged into {ITEMS_PATH}: {merged} items total")

    reasons: defaultdict[str, int] = defaultdict(int)
    print("\ndrop table (per seed):")
    for sid, c in table.items():
        gen = c["_generated"]
        parts = {k: v for k, v in c.items() if not k.startswith("_") and v}
        print(f"  {sid}: gen={gen} dropped={sum(parts.values())} {parts}")
        for k, v in parts.items():
            reasons[k] += v
    total_gen = sum(c["_generated"] for c in table.values())
    print("\ndrop table (total):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:36s} {v:4d}  {100 * v / max(total_gen, 1):5.1f}%")
    print(f"  kept {n_written}/{total_gen} = {100 * n_written / max(total_gen, 1):.1f}%")
    if all_secs:
        print(
            f"\nthroughput: {len(all_secs)} model calls, "
            f"{sum(all_secs) / len(all_secs):.1f} s/call (mean), total {sum(all_secs):.0f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
