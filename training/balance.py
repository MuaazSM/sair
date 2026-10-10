"""Deterministic balancing of the TRAIN split only (PRD §12.2 shortcut audit).

Reads ``training/data/train.unbalanced.jsonl`` (written by ``training.split``), writes the
balanced ``training/data/train.jsonl`` and a drop log of item ids and reason codes. Dev and
test are never read or written here.

Steps (all deterministic; ties are broken by a hash of the item id):

1. Subject cap: while any gold subject occurs in more than ``SUBJECT_CAP`` of the items,
   drop one item containing the most over-represented subject (non-hard items first).
2. Length leak: while Spearman(transcript words, gold fact count) > ``RHO_MAX``, drop the
   item that contributes most to the correlation (long notes with many facts, then short
   notes with no facts), non-hard first. Capped by ``MAX_LEAK_DROP_SHARE``; if the cap is
   reached the script stops and says so rather than loosen anything.

Usage::

    uv run python -m training.balance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.common import DATA_DIR, read_jsonl, write_jsonl

SUBJECT_CAP = 0.05
RHO_MAX = 0.60
RHO_TARGET = 0.59  # stop slightly below the audit threshold so rounding cannot flip it
MAX_LEAK_DROP_SHARE = 0.25

Item = dict[str, Any]


def norm_subject(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def tie(item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big")


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


def n_words(item: Item) -> int:
    return len(str(item["transcript"]).split())


def n_facts(item: Item) -> int:
    return len(item["gold"].get("facts") or [])


def _contributions(items: Sequence[Item]) -> tuple[float | None, list[float]]:
    """Spearman rho and each item's covariance contribution."""
    n = len(items)
    if n < 3:
        return None, [0.0] * n
    rx = _ranks([n_words(i) for i in items])
    ry = _ranks([n_facts(i) for i in items])
    mx, my = sum(rx) / n, sum(ry) / n
    contrib = [(a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)]
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None, contrib
    return sum(contrib) / (vx * vy) ** 0.5, contrib


def rho(items: Sequence[Item]) -> float | None:
    return _contributions(items)[0]


def subject_counts(items: Sequence[Item]) -> Counter[str]:
    c: Counter[str] = Counter()
    for it in items:
        c.update({norm_subject(str(f["subject"])) for f in it["gold"].get("facts") or []})
    return c


def cap_subjects(
    items: Sequence[Item], cap: float = SUBJECT_CAP
) -> tuple[list[Item], list[tuple[str, str]]]:
    """Drop items until no subject exceeds ``cap`` of the remaining items."""
    kept = list(items)
    dropped: list[tuple[str, str]] = []
    while kept:
        counts = subject_counts(kept)
        if not counts:
            break
        subject, top = counts.most_common(1)[0]
        # strict audit rule: top / n <= cap
        if top / len(kept) <= cap:
            break
        cands = [
            i
            for i in kept
            if subject in {norm_subject(str(f["subject"])) for f in i["gold"].get("facts") or []}
        ]
        victim = min(cands, key=lambda i: (bool(i["meta"].get("hard")), tie(str(i["item_id"]))))
        kept.remove(victim)
        dropped.append((str(victim["item_id"]), "subject_cap"))
    return kept, dropped


def cut_length_leak(
    items: Sequence[Item], max_share: float = MAX_LEAK_DROP_SHARE
) -> tuple[list[Item], list[tuple[str, str]], bool]:
    """Drop the largest rank-covariance contributors until rho <= RHO_TARGET.

    Returns (kept, dropped, ok); ok is False if the drop cap was hit first.
    """
    kept = list(items)
    dropped: list[tuple[str, str]] = []
    limit = int(len(items) * max_share)
    while True:
        r, contrib = _contributions(kept)
        if r is None or r <= RHO_TARGET:
            return kept, dropped, True
        if len(dropped) >= limit:
            return kept, dropped, False
        idx = max(
            range(len(kept)),
            key=lambda k: (
                round(contrib[k], 6),
                not kept[k]["meta"].get("hard"),
                tie(str(kept[k]["item_id"])),
            ),
        )
        dropped.append((str(kept[idx]["item_id"]), "length_leak"))
        del kept[idx]


def balance(items: Sequence[Item]) -> tuple[list[Item], list[tuple[str, str]], bool]:
    kept, d1 = cap_subjects(items)
    kept, d2, ok = cut_length_leak(kept)
    # removing items can push a subject back over the cap: re-run once, then re-check rho
    kept, d3 = cap_subjects(kept)
    return kept, d1 + d2 + d3, ok


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=DATA_DIR / "train.unbalanced.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--log", type=Path, default=DATA_DIR / "train.balance.json")
    args = ap.parse_args(argv)
    items = read_jsonl(args.inp)
    before = rho(items)
    kept, dropped, ok = balance(items)
    write_jsonl(args.out, kept)
    reasons = Counter(r for _, r in dropped)
    after = rho(kept)
    args.log.write_text(
        json.dumps(
            {"in": len(items), "out": len(kept), "reasons": dict(reasons), "dropped": dropped},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"train {len(items)} -> {len(kept)}; dropped {dict(reasons)}")
    print(f"length_leak_facts rho {fmt(before)} -> {fmt(after)}")
    top = subject_counts(kept).most_common(1)
    if top:
        print(f"max subject share {top[0][1] / len(kept):.3f}")
    if not ok:
        print("STOP: length-leak drop cap reached; rho still above target", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
