"""Shortcut audit of the synthetic training data (PRD §12.2). Must pass before full training.

    uv run python training/audit.py                      # train.jsonl (+ sft_*.jsonl if present)
    uv run python training/audit.py --strict             # non-zero exit on any failure
    uv run python training/audit.py --items training/data/train.jsonl \
        --sft training/data/sft_train.jsonl

Prints a table of rules (value, threshold, PASS/FAIL/INFO). Synthetic data only; prints
aggregates only, never transcript, span or entry text (subjects are reported only as a
count and a max share, never by name).

Rules and thresholds (documented here, recorded in the run output):

R1  length_leak_facts   Spearman rho(transcript words, gold fact count) <= 0.60.
R2  length_leak_entry   Spearman rho(transcript words, gold entry words) <= 0.60 (needs SFT
                        gold entries; INFO if absent).
R3  length_predictor    A trivial length-only predictor of the fact count (transcript words
                        cut into 5 quantile bins, predict each bin's modal count, fitted and
                        scored on the same data, which flatters it) must not beat the majority
                        class baseline by more than 0.25 accuracy.
R4  fact_count_spread   every gold fact count 0..6 occurs at least once.
R5  subject_share       no gold subject (normalised) appears in more than 5% of items.
R6  hedge_positions     among hedged gold spans, start / middle / end each >= 15%.
R7  hinglish_meta       >= 30% of items have meta.language_mix == hinglish.
R8  hinglish_script     >= 30% of items are Hinglish-dominant by script: Latin letters >= 90%
                        of letters AND romanised-Hindi function words >= 15% of Latin tokens.
R9  devanagari_meta     >= 10% of items have meta.language_mix == devanagari_heavy.
R10 devanagari_script   >= 10% of items are Devanagari-heavy by script: Devanagari >= 50% of
                        letters.
R11 mission_dropout     share of items with mission_text null in [0.15, 0.30] (~20%).
R12 hard_share          share of items with meta.hard in [0.15, 0.40] (~20%; the generator's
                        hard flag also counts any hedged item, so the band is wider above).
R13 template_gold_share share of SFT rows whose gold is the template: INFO (reported only;
                        warn above 0.40).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.tinker_common import DATA_DIR, read_jsonl, script_stats, word_count

RHO_MAX = 0.60
PREDICTOR_MARGIN_MAX = 0.25
SUBJECT_SHARE_MAX = 0.05
HEDGE_POS_MIN = 0.15
HINGLISH_MIN = 0.30
DEVANAGARI_MIN = 0.10
DROPOUT_BAND = (0.15, 0.30)
HARD_BAND = (0.15, 0.40)
TEMPLATE_WARN = 0.40
HINGLISH_LATIN_MIN = 0.90
HINGLISH_ROMAN_HINDI_MIN = 0.15
DEVA_HEAVY_MIN = 0.50


@dataclass
class Rule:
    name: str
    value: str
    threshold: str
    status: str  # PASS | FAIL | INFO | WARN


# --------------------------------------------------------------------------- stats


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rho with average ranks for ties; None if undefined (n<3 or constant)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return float(cov / (vx * vy) ** 0.5)


def length_predictor_margin(lengths: Sequence[int], labels: Sequence[int], bins: int = 5) -> float:
    """Accuracy of a quantile-bin modal predictor minus the majority-class accuracy."""
    n = len(labels)
    if n == 0:
        return 0.0
    majority = Counter(labels).most_common(1)[0][1] / n
    order = sorted(range(n), key=lambda i: lengths[i])
    correct = 0
    for b in range(bins):
        idx = order[b * n // bins : (b + 1) * n // bins]
        if not idx:
            continue
        mode = Counter(labels[i] for i in idx).most_common(1)[0][1]
        correct += mode
    return correct / n - majority


def _norm_subject(s: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def hedge_position(span: str) -> str | None:
    """Where the first hedge cue sits in a span: start, middle or end (None: no cue).

    Same definition as the generator's ``meta.hedge_positions`` (used when meta lacks it).
    """
    import re

    from hub.grounding import load_lexicon, normalise

    lex = load_lexicon()
    s = normalise(span)
    best: tuple[int, int] | None = None
    for cue in list(lex.hedge_latin) + list(lex.hedge_deva):
        m = re.search(r"(?<![\w])" + re.escape(cue) + r"(?![\w])", s)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end())
    if best is None:
        return None
    start, end = best
    if start <= 1 or not s[:start].strip(" .,!?;:।"):
        return "start"
    if not s[end:].strip(" .,!?;:।"):
        return "end"
    return "middle"


# --------------------------------------------------------------------------- audit


def _band(name: str, v: float, lo: float, hi: float) -> Rule:
    ok = lo <= v <= hi
    return Rule(name, f"{v:.3f}", f"[{lo:.2f}, {hi:.2f}]", "PASS" if ok else "FAIL")


def _min(name: str, v: float, lo: float) -> Rule:
    return Rule(name, f"{v:.3f}", f">= {lo:.2f}", "PASS" if v >= lo else "FAIL")


def audit(items: list[dict[str, Any]], sft_rows: list[dict[str, Any]] | None = None) -> list[Rule]:
    rules: list[Rule] = []
    n = len(items)
    if n == 0:
        return [Rule("items", "0", ">= 1", "FAIL")]
    t_words = [word_count(str(it["transcript"])) for it in items]
    n_facts = [len(it["gold"].get("facts") or []) for it in items]

    rho = spearman(t_words, n_facts)
    rules.append(
        Rule(
            "length_leak_facts",
            "n/a" if rho is None else f"{rho:.3f}",
            f"<= {RHO_MAX:.2f}",
            "INFO" if rho is None else ("PASS" if rho <= RHO_MAX else "FAIL"),
        )
    )

    entry_words: dict[str, int] = {}
    sources: Counter[str] = Counter()
    for row in sft_rows or []:
        meta = row.get("meta") or {}
        sources["template" if meta.get("source") == "template" else "model"] += 1
        try:
            gold = json.loads(row["messages"][-1]["content"])
            entry_words[str(meta.get("item_id"))] = word_count(str(gold["entry"]))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    pairs = [
        (tw, entry_words[str(it["item_id"])])
        for it, tw in zip(items, t_words, strict=True)
        if str(it["item_id"]) in entry_words
    ]
    if len(pairs) >= 3:
        rho_e = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        rules.append(
            Rule(
                "length_leak_entry",
                "n/a" if rho_e is None else f"{rho_e:.3f} (n={len(pairs)})",
                f"<= {RHO_MAX:.2f}",
                "INFO" if rho_e is None else ("PASS" if rho_e <= RHO_MAX else "FAIL"),
            )
        )
    else:
        rules.append(Rule("length_leak_entry", "no SFT entries", f"<= {RHO_MAX:.2f}", "INFO"))

    margin = length_predictor_margin(t_words, n_facts)
    rules.append(
        Rule(
            "length_predictor",
            f"{margin:+.3f}",
            f"<= +{PREDICTOR_MARGIN_MAX:.2f} over majority",
            "PASS" if margin <= PREDICTOR_MARGIN_MAX else "FAIL",
        )
    )

    counts = Counter(n_facts)
    missing = [k for k in range(7) if counts.get(k, 0) == 0]
    rules.append(
        Rule(
            "fact_count_spread",
            " ".join(f"{k}:{counts.get(k, 0)}" for k in range(7)),
            "each of 0..6 >= 1",
            "PASS" if not missing else "FAIL",
        )
    )

    subj_items: Counter[str] = Counter()
    for it in items:
        subj_items.update({_norm_subject(str(f["subject"])) for f in it["gold"].get("facts") or []})
    top = subj_items.most_common(1)[0][1] / n if subj_items else 0.0
    rules.append(
        Rule(
            "subject_share",
            f"max={top:.3f} distinct={len(subj_items)}",
            f"<= {SUBJECT_SHARE_MAX:.2f}",
            "PASS" if top <= SUBJECT_SHARE_MAX else "FAIL",
        )
    )

    pos: Counter[str] = Counter()
    for it in items:
        meta_pos = (it.get("meta") or {}).get("hedge_positions")
        if meta_pos is not None:
            pos.update(str(p) for p in meta_pos)
        else:
            for f in it["gold"].get("facts") or []:
                if f.get("certainty") == "unsure" and (p := hedge_position(str(f["span"]))):
                    pos[p] += 1
    total_pos = sum(pos.values())
    shares = {k: (pos[k] / total_pos if total_pos else 0.0) for k in ("start", "middle", "end")}
    ok_pos = total_pos > 0 and all(v >= HEDGE_POS_MIN for v in shares.values())
    rules.append(
        Rule(
            "hedge_positions",
            " ".join(f"{k}={v:.2f}" for k, v in shares.items()) + f" (n={total_pos})",
            f"each >= {HEDGE_POS_MIN:.2f}",
            "PASS" if ok_pos else "FAIL",
        )
    )

    mixes = Counter(str((it.get("meta") or {}).get("language_mix")) for it in items)
    stats = [script_stats(str(it["transcript"])) for it in items]
    hing_script = (
        sum(
            1
            for s in stats
            if s["latin_ratio"] >= HINGLISH_LATIN_MIN
            and s["roman_hindi_ratio"] >= HINGLISH_ROMAN_HINDI_MIN
        )
        / n
    )
    deva_script = sum(1 for s in stats if s["deva_ratio"] >= DEVA_HEAVY_MIN) / n
    rules.append(_min("hinglish_meta", mixes["hinglish"] / n, HINGLISH_MIN))
    rules.append(_min("hinglish_script", hing_script, HINGLISH_MIN))
    rules.append(_min("devanagari_meta", mixes["devanagari_heavy"] / n, DEVANAGARI_MIN))
    rules.append(_min("devanagari_script", deva_script, DEVANAGARI_MIN))

    dropout = sum(1 for it in items if not it.get("mission_text")) / n
    rules.append(_band("mission_dropout", dropout, *DROPOUT_BAND))
    hard = sum(1 for it in items if (it.get("meta") or {}).get("hard")) / n
    rules.append(_band("hard_share", hard, *HARD_BAND))

    if sources:
        share = sources["template"] / sum(sources.values())
        rules.append(
            Rule(
                "template_gold_share",
                f"{share:.3f} (n={sum(sources.values())})",
                f"report; warn > {TEMPLATE_WARN:.2f}",
                "WARN" if share > TEMPLATE_WARN else "INFO",
            )
        )
    else:
        rules.append(Rule("template_gold_share", "no SFT rows", "report", "INFO"))
    rules.append(Rule("items", str(n), "report", "INFO"))
    rules.append(
        Rule(
            "language_mix", " ".join(f"{k}={v}" for k, v in sorted(mixes.items())), "report", "INFO"
        )
    )
    return rules


def print_table(rules: list[Rule]) -> None:
    w = max(len(r.name) for r in rules)
    wv = max(len(r.value) for r in rules)
    print(f"{'rule':{w}}  {'value':{wv}}  {'threshold':28}  status")
    for r in rules:
        print(f"{r.name:{w}}  {r.value:{wv}}  {r.threshold:28}  {r.status}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PRD §12.2 shortcut audit")
    ap.add_argument("--items", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument(
        "--sft",
        type=Path,
        nargs="*",
        default=None,
        help="SFT files (default: training/data/sft_train.jsonl, sft_dev.jsonl if present)",
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on any FAIL")
    ap.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    a = ap.parse_args(argv)
    if not a.items.is_file():
        print("missing items file")
        return 2
    items = read_jsonl(a.items)
    sft_paths = (
        a.sft
        if a.sft is not None
        else [p for p in (DATA_DIR / "sft_train.jsonl", DATA_DIR / "sft_dev.jsonl") if p.is_file()]
    )
    sft_rows = [r for p in sft_paths for r in read_jsonl(p)]
    rules = audit(items, sft_rows)
    print_table(rules)
    failed = [r.name for r in rules if r.status == "FAIL"]
    print(f"\nFAILED: {', '.join(failed)}" if failed else "\nall rules pass")
    if a.json:
        a.json.write_text(
            json.dumps([r.__dict__ for r in rules], indent=1) + "\n", encoding="utf-8"
        )
    return 1 if (a.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
