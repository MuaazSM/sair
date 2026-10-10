"""PRD §13.3 metrics as pure functions over per-item records and gold labels.

Every rate is a *ratio of sums* (micro-average): each item contributes a numerator and a
denominator (:class:`Ratio`), the point estimate is ``sum(num) / sum(den)``, and the
bootstrap resamples the same per-item pairs (``eval/bootstrap.py``). Definitions:

* ``unsupported_rate``: share of entries with >= 1 unsupported detail (checker,
  ``CheckResult.unsupported_details``) among items that have a final entry. For "+"
  systems this is the FINAL entry (model or template). Items without an entry
  (unparseable plain-system output, backend error) are excluded and counted by
  ``invalid_rate``.
* ``details_per_entry``: mean number of distinct unsupported details per final entry
  (same denominator as ``unsupported_rate``).
* ``hedge_preservation``: among gold-unsure facts the system *mentions* (a predicted
  verified fact matches it, or its subject appears in the entry, alias-aware), the share
  where a matched predicted fact has certainty=unsure AND every entry sentence that
  mentions the subject carries an entry hedge word (``hedges.yaml entry_hedge_words``).
  Gold-unsure facts that are never mentioned are excluded.
* ``fact_precision`` / ``fact_recall``: predicted = the final check's verified facts
  (what would enter memory). A predicted and a gold fact match when the token-level
  Jaccard of their spans is >= 0.5 (tokens: NFC, casefold, punctuation stripped,
  whitespace split; sets) AND their subjects match alias-aware (same normalised text,
  one contains the other on token boundaries, or they share a lexicon id). One-to-one,
  greedy by descending Jaccard. precision = TP / #predicted, recall = TP / #gold.
* ``valid_json_no_grammar``: share of first-attempt raw outputs that parse into the §9.1
  schema (``parse_writer_output``), on records generated with ``grammar=False`` only.
* ``template_fallback_rate``: share of items whose final entry came from the template.
* ``mission_accuracy``: share of items whose final (check-7 coerced) mission_result equals
  the gold mission_result.
* ``invalid_rate``: share of items with no final entry (unparseable or backend error).
* ``checker_recall_items``: among items a human labelled as having >= 1 unsupported
  detail, the share the checker flagged. ``checker_recall_details``: among human-listed
  details, the share whose canonical key (or normalised surface) the checker also found.
* ``wer`` / ``cer``: corpus-level edit distance / reference length (NFC, casefold,
  punctuation stripped, whitespace collapsed; words = whitespace tokens, chars = Unicode
  code points incl. single spaces, so Devanagari matras count as characters).
* latency: p50/p95 (numpy linear interpolation) per node and ``e2e`` over records.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from eval.items import EvalItem, GoldFact, HumanLabels
from eval.systems import ItemRecord
from hub.grounding import extract_details, load_lexicon, normalise
from hub.schemas import Certainty, Fact

__all__ = [
    "METRICS",
    "Ratio",
    "cer",
    "cer_ratio",
    "checker_recall_details",
    "checker_recall_items",
    "details_per_entry",
    "edit_distance",
    "fact_precision",
    "fact_recall",
    "hedge_preservation",
    "invalid_rate",
    "latency_percentiles",
    "match_facts",
    "mission_accuracy",
    "point",
    "span_jaccard",
    "subjects_match",
    "template_fallback_rate",
    "unsupported_rate",
    "valid_json_no_grammar",
    "wer",
    "wer_ratio",
]

SPAN_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Ratio:
    """One item's contribution to a ratio metric."""

    num: float
    den: float


def point(values: Sequence[Ratio]) -> float | None:
    den = sum(v.den for v in values)
    return None if den == 0 else sum(v.num for v in values) / den


# --------------------------------------------------------------------------- text helpers

_TOKEN_SPLIT = re.compile(r"\s+")


def _clean(text: str) -> str:
    """NFC, casefold, punctuation/symbols -> space, whitespace collapsed."""
    nfc = unicodedata.normalize("NFC", text).casefold()
    chars = [" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in nfc]
    return " ".join("".join(chars).split())


def _tokens(text: str) -> list[str]:
    cleaned = _clean(text)
    return cleaned.split(" ") if cleaned else []


_HEDGE_TOKEN = re.compile(r"[0-9A-Za-zÀ-ɏ]+(?:['’][0-9A-Za-zÀ-ɏ]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?…।॥])\s+")


@lru_cache(maxsize=1)
def _entry_hedges() -> frozenset[str]:
    return load_lexicon().entry_hedges


def _has_entry_hedge(sentence: str) -> bool:
    toks = [
        t.casefold().replace("’", "'")
        for t in _HEDGE_TOKEN.findall(unicodedata.normalize("NFC", sentence))
    ]
    padded = " " + " ".join(toks) + " "
    return any(f" {h} " in padded for h in _entry_hedges())


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(unicodedata.normalize("NFC", text).strip()) if s.strip()]


def _bounded_in(hay: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?<![0-9a-zà-ɏ]){re.escape(needle)}(?![0-9a-zà-ɏ])", hay) is not None


def _lex_ids(text: str) -> set[str]:
    return {k for k in extract_details(text, inclusive=True) if ":" not in k}


# --------------------------------------------------------------------------- fact matching


def span_jaccard(a: str, b: str) -> float:
    """Token-set Jaccard of two spans after :func:`_clean` (0.0 if both empty)."""
    sa, sb = set(_tokens(a)), set(_tokens(b))
    union = sa | sb
    return 0.0 if not union else len(sa & sb) / len(union)


def subjects_match(a: str, b: str) -> bool:
    na, nb = _clean(a), _clean(b)
    if not na or not nb:
        return False
    if na == nb or _bounded_in(na, nb) or _bounded_in(nb, na):
        return True
    return bool(_lex_ids(a) & _lex_ids(b))


def match_facts(
    predicted: Sequence[Fact | GoldFact], gold: Sequence[GoldFact]
) -> list[tuple[int, int]]:
    """One-to-one (pred index, gold index) matches, greedy by descending span Jaccard."""
    cands: list[tuple[float, int, int]] = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            jac = span_jaccard(p.span, g.span)
            if jac >= SPAN_MATCH_THRESHOLD and subjects_match(p.subject, g.subject):
                cands.append((jac, i, j))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int]] = []
    for _, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def _pred_facts(rec: ItemRecord) -> list[Fact]:
    return rec.final_check.facts() if rec.final_check is not None else []


def _entry(rec: ItemRecord) -> str:
    return str(rec.final["entry"]) if rec.final is not None else ""


# --------------------------------------------------------------------------- metrics

Pair = tuple[ItemRecord, EvalItem]


def unsupported_rate(rec: ItemRecord, item: EvalItem) -> Ratio:
    if not rec.has_entry or rec.final_check is None:
        return Ratio(0, 0)
    return Ratio(1.0 if rec.final_check.unsupported_details else 0.0, 1)


def details_per_entry(rec: ItemRecord, item: EvalItem) -> Ratio:
    if not rec.has_entry or rec.final_check is None:
        return Ratio(0, 0)
    return Ratio(float(len(set(rec.final_check.unsupported_details))), 1)


def invalid_rate(rec: ItemRecord, item: EvalItem) -> Ratio:
    return Ratio(0.0 if rec.has_entry else 1.0, 1)


def template_fallback_rate(rec: ItemRecord, item: EvalItem) -> Ratio:
    return Ratio(1.0 if rec.final_origin == "template" else 0.0, 1)


def mission_accuracy(rec: ItemRecord, item: EvalItem) -> Ratio:
    if rec.final_check is None:
        return Ratio(0.0, 1)  # no entry: counted as wrong
    return Ratio(
        1.0 if rec.final_check.mission_result == item.gold.mission_result.value else 0.0, 1
    )


def valid_json_no_grammar(rec: ItemRecord, item: EvalItem) -> Ratio:
    if rec.grammar or not rec.attempts or rec.system == "A":
        return Ratio(0, 0)
    return Ratio(1.0 if rec.attempts[0].parse_ok else 0.0, 1)


def fact_precision(rec: ItemRecord, item: EvalItem) -> Ratio:
    pred = _pred_facts(rec)
    return Ratio(float(len(match_facts(pred, item.gold.facts))), float(len(pred)))


def fact_recall(rec: ItemRecord, item: EvalItem) -> Ratio:
    pred = _pred_facts(rec)
    return Ratio(float(len(match_facts(pred, item.gold.facts))), float(len(item.gold.facts)))


def _subject_in_sentence(subject: str, sentence: str) -> bool:
    if _bounded_in(normalise(sentence), normalise(subject)):
        return True
    ids = _lex_ids(subject)
    return bool(ids and ids & _lex_ids(sentence))


def hedge_preservation(rec: ItemRecord, item: EvalItem) -> Ratio:
    gold_unsure = [(j, g) for j, g in enumerate(item.gold.facts) if g.certainty is Certainty.UNSURE]
    if not gold_unsure or not rec.has_entry:
        return Ratio(0, 0)
    pred = _pred_facts(rec)
    matched = {j: i for i, j in match_facts(pred, item.gold.facts)}
    sentences = _sentences(_entry(rec))
    num = den = 0
    for j, g in gold_unsure:
        mention_sents = [s for s in sentences if _subject_in_sentence(g.subject, s)]
        if j not in matched and not mention_sents:
            continue  # not mentioned
        den += 1
        unsure_marked = j in matched and pred[matched[j]].certainty is Certainty.UNSURE
        hedged = all(_has_entry_hedge(s) for s in mention_sents)
        if unsure_marked and hedged:
            num += 1
    return Ratio(float(num), float(den))


MetricFn = Callable[[ItemRecord, EvalItem], Ratio]

METRICS: dict[str, MetricFn] = {
    "unsupported_rate": unsupported_rate,
    "details_per_entry": details_per_entry,
    "hedge_preservation": hedge_preservation,
    "fact_precision": fact_precision,
    "fact_recall": fact_recall,
    "template_fallback_rate": template_fallback_rate,
    "mission_accuracy": mission_accuracy,
    "invalid_rate": invalid_rate,
    "valid_json_no_grammar": valid_json_no_grammar,
}


# --------------------------------------------------------------------------- human labels


def checker_recall_items(pairs: Iterable[tuple[bool, bool]]) -> Ratio:
    """``pairs`` = (human says unsupported, checker flagged) per item → summed Ratio."""
    num = den = 0
    for human, checker in pairs:
        if human:
            den += 1
            num += 1 if checker else 0
    return Ratio(float(num), float(den))


def _detail_keys(surface: str) -> set[str]:
    keys = extract_details(surface, inclusive=True)
    return keys or {f"surface:{_clean(surface)}"}


def checker_recall_details(human: Sequence[str], checker: Sequence[str]) -> Ratio:
    """Per item: of the human-listed unsupported details, how many the checker also found
    (matched by canonical key, alias-aware; surface text when no key exists)."""
    checker_keys: set[str] = set()
    for c in checker:
        checker_keys |= _detail_keys(c)
    num = sum(1 for h in human if _detail_keys(h) & checker_keys)
    return Ratio(float(num), float(len(human)))


def human_item_label(labels: HumanLabels) -> bool | None:
    if labels.has_unsupported is not None:
        return labels.has_unsupported
    if labels.unsupported is not None:
        return bool(labels.unsupported)
    return None


# --------------------------------------------------------------------------- WER / CER


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance (substitution, insertion, deletion all cost 1)."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer_ratio(ref: str, hyp: str) -> Ratio:
    r, h = _tokens(ref), _tokens(hyp)
    return Ratio(float(edit_distance(r, h)), float(len(r)))


def cer_ratio(ref: str, hyp: str) -> Ratio:
    r, h = list(_clean(ref)), list(_clean(hyp))
    return Ratio(float(edit_distance(r, h)), float(len(r)))


def wer(ref: str, hyp: str) -> float:
    v = wer_ratio(ref, hyp)
    if v.den == 0:
        return 0.0 if v.num == 0 else float("inf")
    return v.num / v.den


def cer(ref: str, hyp: str) -> float:
    v = cer_ratio(ref, hyp)
    if v.den == 0:
        return 0.0 if v.num == 0 else float("inf")
    return v.num / v.den


# --------------------------------------------------------------------------- latency


def latency_percentiles(records: Sequence[ItemRecord]) -> dict[str, dict[str, float | int]]:
    """p50/p95 in ms per node (and ``e2e``) across records that ran that node."""
    by_node: dict[str, list[float]] = {}
    for rec in records:
        for node, ms in rec.latency.items():
            by_node.setdefault(node, []).append(float(ms))
    out: dict[str, dict[str, float | int]] = {}
    for node, vals in sorted(by_node.items()):
        arr = np.asarray(vals, dtype=float)
        out[node] = {
            "n": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }
    return out
