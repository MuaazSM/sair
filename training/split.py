"""Mission-text dropout and the by-seed train/dev/test split (PRD §12.1.4-5).

Dropout definition (documented decision): with the mission text removed from the writer's
input the writer cannot know what the mission was, so the gold label becomes
``mission_result = not_mentioned`` and ``mission_span = null``. The facts are unchanged.
The selection is deterministic per item (a hash of item_id and the RNG seed), about 20%.

Split: whole seeds are assigned to train/dev/test 80/10/10 with a fixed RNG seed, so no
seed scenario appears in two splits. ``test`` is also written to
``eval/data/synthetic/test.jsonl`` for the eval harness.

Usage::

    uv run python -m training.split [--in training/data/items.jsonl] [--rng 1729]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from training.common import DATA_DIR, EVAL_SYNTH_DIR, ITEMS_PATH, read_jsonl, write_jsonl

RNG_SEED = 1729
DROPOUT_FRACTION = 0.20
SPLIT_FRACTIONS = (("train", 0.8), ("dev", 0.1), ("test", 0.1))

Item = dict[str, Any]


def split_seeds(seed_ids: Iterable[str], rng_seed: int = RNG_SEED) -> dict[str, str]:
    """Assign each seed id to train/dev/test (80/10/10) deterministically."""
    ids = sorted(set(seed_ids))
    random.Random(rng_seed).shuffle(ids)
    n = len(ids)
    n_dev = max(1, round(n * SPLIT_FRACTIONS[1][1])) if n >= 3 else 0
    n_test = max(1, round(n * SPLIT_FRACTIONS[2][1])) if n >= 3 else 0
    out: dict[str, str] = {}
    for i, sid in enumerate(ids):
        if i < n_test:
            out[sid] = "test"
        elif i < n_test + n_dev:
            out[sid] = "dev"
        else:
            out[sid] = "train"
    return out


def dropout_selected(item_id: str, rng_seed: int = RNG_SEED) -> float:
    """Stable pseudo-random number in [0, 1) for an item."""
    h = hashlib.sha256(f"{rng_seed}:{item_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def apply_dropout(
    items: Sequence[Item], fraction: float = DROPOUT_FRACTION, rng_seed: int = RNG_SEED
) -> list[Item]:
    """Return copies where ~fraction of items lose the mission text (and the mission gold)."""
    out: list[Item] = []
    for it in items:
        new = copy.deepcopy(it)
        drop = dropout_selected(it["item_id"], rng_seed) < fraction
        new["meta"]["mission_dropout"] = drop
        if drop:
            new["mission_text"] = None
            new["gold"]["mission_result"] = "not_mentioned"
            new["gold"]["mission_span"] = None
        out.append(new)
    return out


def run(items: Sequence[Item], rng_seed: int = RNG_SEED) -> dict[str, list[Item]]:
    assignment = split_seeds((i["seed_id"] for i in items), rng_seed)
    items = apply_dropout(items, rng_seed=rng_seed)
    splits: dict[str, list[Item]] = {"train": [], "dev": [], "test": []}
    for it in items:
        splits[assignment[it["seed_id"]]].append(it)
    return splits


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ITEMS_PATH)
    ap.add_argument("--rng", type=int, default=RNG_SEED)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--eval-out", type=Path, default=EVAL_SYNTH_DIR / "test.jsonl")
    args = ap.parse_args(argv)

    items = read_jsonl(args.inp)
    if not items:
        print("no items", file=sys.stderr)
        return 1
    splits = run(items, args.rng)
    for name, rows in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        if name == "train":
            # balance.py reads this copy and rewrites train.jsonl (idempotent re-runs)
            write_jsonl(args.out_dir / "train.unbalanced.jsonl", rows)
    write_jsonl(args.eval_out, splits["test"])

    total = len(items)
    print(f"{total} items, {len({i['seed_id'] for i in items})} seeds, rng={args.rng}")
    for name, rows in splits.items():
        seeds = len({r["seed_id"] for r in rows})
        drop = sum(1 for r in rows if r["meta"]["mission_dropout"])
        hard = sum(1 for r in rows if r["meta"]["hard"])
        mix = Counter(r["meta"]["language_mix"] for r in rows)
        print(
            f"  {name:5s} items={len(rows):5d} seeds={seeds:3d} "
            f"dropout={drop} ({100 * drop / max(len(rows), 1):.1f}%) "
            f"hard={hard} ({100 * hard / max(len(rows), 1):.1f}%) mix={dict(mix)}"
        )
    allmix = Counter(i["meta"]["language_mix"] for i in items)
    print("  language_mix overall:", {k: f"{100 * v / total:.1f}%" for k, v in allmix.items()})
    print(f"  wrote {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
